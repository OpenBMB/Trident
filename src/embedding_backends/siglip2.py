#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
embedding_backends/siglip2.py

SigLIP 2 backend (baseline; default checkpoint google/siglip2-large-patch16-384).
SigLIP 2 succeeds SigLIP: sigmoid loss instead of a softmax contrastive loss,
plus decoder loss / global-local & masked prediction / resolution adaptation
(see https://huggingface.co/blog/siglip2). The fixed-resolution
"large-patch16-384" variant is used (not naflex), so no naflex-specific
arguments such as max_num_patches are needed.

Differences from jina_clip_v2 / clip_openai:
  1. The model is loaded through AutoModel (Siglip2Model), which avoids
     depending on whether `Siglip2Model` is exported at the top level of the
     installed transformers version.
  2. The model card requires `padding="max_length", max_length=64` for text
     (SigLIP/SigLIP2 are trained with text padded/truncated to 64 tokens; other
     settings run but produce wrong embeddings and much worse retrieval).
  3. get_image_features / get_text_features return either a pooled tensor or a
     ModelOutput with `.pooler_output` depending on the version;
     `getattr(out, "pooler_output", out)` handles both.
  4. The sigmoid training loss does not change inference: cosine similarity of
     get_*_features outputs is used as for regular CLIP.

Shared with clip_openai.py:
  - no joint text+image encoding; records with both use common.clip_style_fuse;
  - processor pixel_values are float32 and are cast to the model dtype.

Dependencies: transformers (a recent version with Siglip2 support); no
trust_remote_code.

Usage:
    python src/eval/embed_jsonl_unified_multigpu.py \
        --model_type siglip2 \
        --checkpoint google/siglip2-large-patch16-384 \
        --role query --input_jsonl ... --output_dir ...
"""

import numpy as np
import torch

from .base import EmbeddingBackend
from .common import l2_normalize, parse_torch_dtype, to_numpy_f32

# SigLIP/SigLIP2 are trained with text padded/truncated to this length; inference must
# use the same length, otherwise text embeddings drift and retrieval degrades.
_TEXT_MAX_LENGTH = 64


class Siglip2Backend(EmbeddingBackend):
    # CLIP-based (contrastive dual-tower) model; GR-CLIP mean-shift calibration only
    # applies to this family (see base.EmbeddingBackend.is_clip_based).
    is_clip_based = True

    supports_fused_text_image = False
    max_images_per_record = 1

    def default_instruction(self, role: str) -> str:
        # The SigLIP2 text tower is not instruction-tuned; no prefix is added.
        return ""

    def load(self) -> None:
        from transformers import AutoModel, AutoProcessor

        checkpoint = self.args_dict["checkpoint"]
        dtype = parse_torch_dtype(self.args_dict.get("dtype", "float32"))
        attn_impl = self.args_dict.get("attn_implementation")

        model_kwargs: dict = {"torch_dtype": dtype}
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl

        print(f"[worker] loading SigLIP2: {checkpoint} (dtype={dtype})", flush=True)

        self.model = AutoModel.from_pretrained(checkpoint, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()

        # The SigLIP2 text tower uses the Gemma tokenizer (vocab=256000), which also trips the
        # transformers "vocab_size > 100000 -> suspected Mistral regex" false positive
        # (#42591/#44736; see altclip.py). Passing fix_mistral_regex=False only skips the
        # warning branch and does not change tokenization.
        self.processor = AutoProcessor.from_pretrained(
            checkpoint, fix_mistral_regex=False
        )

        print("[worker] SigLIP2 ready", flush=True)

    def build_inputs_cpu(self, items: list[dict]):
        from .common import load_images_parallel

        image_workers = self.args_dict.get("image_load_workers", 8)
        role = self.args_dict["role"]

        image_paths = [
            (item.get("images") or [None])[0] if item.get("images") else None
            for item in items
        ]
        pil_images = load_images_parallel(image_paths, max_workers=image_workers)

        prepared = []
        for item, pil_image in zip(items, pil_images):
            prefix = item.get("instruction")
            if prefix is None:
                prefix = self.default_instruction(role)

            raw_text = item.get("text")
            text_value = (prefix + raw_text) if raw_text else None

            prepared.append({"text": text_value, "pil": pil_image})
        return prepared

    @staticmethod
    def _unwrap(out):
        # get_*_features return types differ across transformers versions (see note 3
        # in the module docstring); normalize to a tensor.
        return getattr(out, "pooler_output", out)

    def _encode_text(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(
            text=texts,
            padding="max_length",
            max_length=_TEXT_MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        embeds = self._unwrap(self.model.get_text_features(**inputs))
        return to_numpy_f32(embeds)

    def _encode_image(self, images: list) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.model.dtype)
        embeds = self._unwrap(self.model.get_image_features(pixel_values=pixel_values))
        return to_numpy_f32(embeds)

    @torch.inference_mode()
    def compute_from_inputs(self, prepared, items, role):
        text_idx, text_batch = [], []
        image_idx, image_batch = [], []

        for i, entry in enumerate(prepared):
            if entry["text"]:
                text_idx.append(i)
                text_batch.append(entry["text"])
            if entry["pil"] is not None:
                image_idx.append(i)
                image_batch.append(entry["pil"])

        text_vecs: dict[int, np.ndarray] = {}
        image_vecs: dict[int, np.ndarray] = {}

        if text_batch:
            text_embeds = self._encode_text(text_batch)
            for offset, i in enumerate(text_idx):
                text_vecs[i] = text_embeds[offset]

        if image_batch:
            image_embeds = self._encode_image(image_batch)
            for offset, i in enumerate(image_idx):
                image_vecs[i] = image_embeds[offset]

        embeddings: list[np.ndarray] = []
        for i in range(len(prepared)):
            has_text = i in text_vecs
            has_image = i in image_vecs

            if has_text and has_image:
                # GR-CLIP: use the base-class gr_fuse instead of clip_style_fuse. With mean
                # calibration, fused documents must subtract e_bar_I / e_bar_T from each
                # component before interpolation (see common.clip_style_fuse); without
                # calibration both are identical.
                embeddings.append(
                    self.gr_fuse(text_vecs[i], image_vecs[i], role, index=i)
                )
            elif has_image:
                embeddings.append(l2_normalize(image_vecs[i]))
            else:
                embeddings.append(l2_normalize(text_vecs[i]))

        array = np.stack(embeddings, axis=0).astype(np.float32)
        token_infos = [(-1, -1, -1) for _ in items]
        return array, token_infos