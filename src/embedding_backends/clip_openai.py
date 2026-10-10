#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
embedding_backends/clip_openai.py

OpenAI CLIP backend (baseline; default checkpoint openai/clip-vit-large-patch14,
768-d). The standard dual-tower CLIP implementation in transformers; text and
images go through their own towers and get_text_features / get_image_features
return projected embeddings.

Notes:
  1. CLIPModel outputs are not normalized; compute_from_inputs applies
     l2_normalize, consistent with the other backends (cosine retrieval).
  2. The CLIP tokenizer context length is 77 (max_position_embeddings=77) and
     longer inputs fail, so truncation=True is passed (max_length is left to the
     tokenizer's model_max_length).
  3. CLIP has no joint text+image encoding; records with both use
     common.clip_style_fuse (same as jina_clip_v2).
  4. CLIPImageProcessor outputs float32 pixel_values; they are cast to
     self.model.dtype before the forward (fp16/bf16 models would fail otherwise).

Dependencies: transformers (CLIPModel / CLIPProcessor); no trust_remote_code.

Usage:
    python src/eval/embed_jsonl_unified_multigpu.py \
        --model_type clip_vit_l14 \
        --checkpoint openai/clip-vit-large-patch14 \
        --role query --input_jsonl ... --output_dir ...

Any standard CLIPModel-compatible OpenAI / LAION checkpoint also works (e.g.
openai/clip-vit-large-patch14-336, openai/clip-vit-base-patch32).
"""

import numpy as np
import torch

from .base import EmbeddingBackend
from .common import l2_normalize, parse_torch_dtype, to_numpy_f32


class OpenAICLIPBackend(EmbeddingBackend):
    # CLIP-based (contrastive dual-tower) model; GR-CLIP mean-shift calibration only
    # applies to this family (see base.EmbeddingBackend.is_clip_based).
    is_clip_based = True

    supports_fused_text_image = False
    max_images_per_record = 1

    def default_instruction(self, role: str) -> str:
        # The CLIP text tower is not instruction-tuned; no query/document prefix is added,
        # consistent with the official usage.
        return ""

    def load(self) -> None:
        from transformers import CLIPModel, CLIPProcessor

        checkpoint = self.args_dict["checkpoint"]
        dtype = parse_torch_dtype(self.args_dict.get("dtype", "float32"))
        attn_impl = self.args_dict.get("attn_implementation")

        model_kwargs: dict = {"torch_dtype": dtype}
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl

        print(f"[worker] loading OpenAI CLIP: {checkpoint} (dtype={dtype})", flush=True)

        self.model = CLIPModel.from_pretrained(checkpoint, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()

        self.processor = CLIPProcessor.from_pretrained(checkpoint)

        print("[worker] OpenAI CLIP ready", flush=True)

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

    def _encode_text(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(
            text=texts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        embeds = self.model.get_text_features(**inputs)
        return to_numpy_f32(embeds)

    def _encode_image(self, images: list) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.model.dtype)
        embeds = self.model.get_image_features(pixel_values=pixel_values)
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
