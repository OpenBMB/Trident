#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
embedding_backends/altclip.py

AltCLIP backend (baseline; default checkpoint BAAI/AltCLIP). AltCLIP replaces
OpenAI CLIP's text tower with a multilingual XLM-R encoder (aligned with the
original CLIP image-text space via teacher learning + contrastive learning) and
keeps the standard CLIP ViT image tower. transformers provides dedicated
AltCLIPModel / AltCLIPProcessor classes.

Differences from clip_openai.py:
  1. Classes are AltCLIPModel / AltCLIPProcessor (same interface as CLIP, but not
     interchangeable in from_pretrained).
  2. The XLM-R text tower supports far more than CLIP's 77 tokens; only
     truncation=True is passed and the tokenizer's own model_max_length applies.
  3. The text tower is multilingual, so non-English text needs no translation.

Shared with clip_openai.py:
  - no joint text+image encoding; records with both use common.clip_style_fuse;
  - CLIP image preprocessing outputs float32 pixel_values, cast to the model dtype.

Dependencies: transformers (AltCLIPModel / AltCLIPProcessor); no trust_remote_code.

Usage:
    python src/eval/embed_jsonl_unified_multigpu.py \
        --model_type altclip \
        --checkpoint BAAI/AltCLIP \
        --role query --input_jsonl ... --output_dir ...
"""

import numpy as np
import torch

from .base import EmbeddingBackend
from .common import l2_normalize, parse_torch_dtype, to_numpy_f32


class AltCLIPBackend(EmbeddingBackend):
    # CLIP-based (contrastive dual-tower) model; GR-CLIP mean-shift calibration only
    # applies to this family (see base.EmbeddingBackend.is_clip_based).
    is_clip_based = True

    supports_fused_text_image = False
    max_images_per_record = 1

    def default_instruction(self, role: str) -> str:
        # The AltCLIP text tower is not instruction-tuned either; no prefix is added.
        return ""

    def load(self) -> None:
        from transformers import AltCLIPModel, AltCLIPProcessor

        checkpoint = self.args_dict["checkpoint"]
        dtype = parse_torch_dtype(self.args_dict.get("dtype", "float32"))
        attn_impl = self.args_dict.get("attn_implementation")

        model_kwargs: dict = {"torch_dtype": dtype}
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl

        print(f"[worker] loading AltCLIP: {checkpoint} (dtype={dtype})", flush=True)

        self.model = AltCLIPModel.from_pretrained(checkpoint, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()

        # The AltCLIP text tower is XLM-R with vocab 250002, above the transformers
        # "vocab_size > 100000 -> suspected Mistral regex" false-positive threshold
        # (huggingface/transformers#42591, #44736). Passing fix_mistral_regex=False only
        # skips the warning branch and does not change tokenization. Do NOT set it to True:
        # that would patch the XLM-R tokenizer with the Mistral tekken regex and break it.
        self.processor = AltCLIPProcessor.from_pretrained(
            checkpoint, fix_mistral_regex=False
        )

        print("[worker] AltCLIP ready", flush=True)

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