#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
embedding_backends/jina_v5_omni.py

jina-embeddings-v5-omni backend (baseline).
"""

import numpy as np
import torch

from .base import EmbeddingBackend
from .common import l2_normalize, load_images_parallel, to_numpy_f32


class JinaV5OmniBackend(EmbeddingBackend):
    supports_fused_text_image = False
    max_images_per_record = 1

    VISION_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

    def default_instruction(self, role: str) -> str:
        return "Query: " if role == "query" else "Document: "

    def load(self) -> None:
        import warnings

        from transformers import AutoModel, AutoProcessor

        checkpoint = self.args_dict["checkpoint"]
        self.task = self.args_dict.get("jina_task") or "retrieval"
        self.dim = self.args_dict.get("dim")

        print(
            f"[worker] loading jina-embeddings-v5-omni: {checkpoint} "
            f"(default_task={self.task})",
            flush=True,
        )

        # _register_vllm() in modeling_jina_embeddings_v5_omni.py tries to import a
        # vllm_qwen3vl_audio submodule to register a vLLM backend. We use plain
        # transformers inference, so the missing module only triggers a harmless UserWarning.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*vLLM registration failed.*",
                category=UserWarning,
            )
            self.model = AutoModel.from_pretrained(
                checkpoint,
                trust_remote_code=True,
                default_task=self.task,
            ).eval()

        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            checkpoint, trust_remote_code=True
        )

        print("[worker] jina-v5-omni ready", flush=True)

    def build_inputs_cpu(self, items: list[dict]):
        # Load images in parallel first, then build the prefix / prompt (the role is read from
        # self.args_dict, which worker_main updates in place for each job in multi-job mode).
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
            text_prompt = (prefix + raw_text) if raw_text else None

            image_prompt = None
            if pil_image is not None:
                image_prompt = prefix + self.VISION_PLACEHOLDER

            prepared.append(
                {
                    "text_prompt": text_prompt,
                    "image_prompt": image_prompt,
                    "pil_image": pil_image,
                }
            )
        return prepared

    @torch.inference_mode()
    def compute_from_inputs(self, prepared, items, role):
        text_idx, text_prompts = [], []
        image_idx = []

        for i, entry in enumerate(prepared):
            if entry["text_prompt"] is not None:
                text_idx.append(i)
                text_prompts.append(entry["text_prompt"])
            if entry["image_prompt"] is not None:
                image_idx.append(i)

        text_vecs: dict[int, np.ndarray] = {}
        image_vecs: dict[int, np.ndarray] = {}

        def _embed_kwargs():
            kwargs = {}
            if self.dim:
                kwargs["truncate_dim"] = self.dim
            return kwargs

        if text_prompts:
            # Truncation is fine for text-only inputs: there are no image placeholders to
            # align, so overly long inputs only lose text tokens.
            inputs = self.processor(
                text=text_prompts, padding=True, truncation=True,
                return_tensors="pt",
            ).to(self.model.device)
            vecs = self.model.embed(**inputs, **_embed_kwargs())
            vecs = to_numpy_f32(vecs)
            for offset, i in enumerate(text_idx):
                text_vecs[i] = vecs[offset]

        for i in image_idx:
            entry = prepared[i]
            # Truncation must be disabled explicitly: this processor's ProcessingKwargs has a
            # small default max_length (independent of tokenizer.model_max_length = 131072).
            # Without truncation=False, part of the expanded <|image_pad|> placeholders would be
            # cut, so the number of image tokens would not match the vision patches
            # ("Mismatch in `image` token count...").
            inputs = self.processor(
                images=entry["pil_image"], text=entry["image_prompt"],
                truncation=False,
                return_tensors="pt",
            ).to(self.model.device)
            vec = self.model.embed(**inputs, **_embed_kwargs())
            image_vecs[i] = to_numpy_f32(vec)[0]

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
