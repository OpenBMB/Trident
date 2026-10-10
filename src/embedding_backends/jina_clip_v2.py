#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
embedding_backends/jina_clip_v2.py

jina-clip-v2 backend (baseline): multilingual multimodal CLIP model (864M
parameters; the text tower is the jina-embeddings-v3 backbone supporting 89
languages; 512x512 images; text and image embeddings share one 1024-d space
with Matryoshka truncation).

The official API has no joint text+image encoding; encode_text / encode_image
return separate embeddings, so records with both use CLIP-style late fusion
(common.clip_style_fuse).

Notes:
  1. The checkpoint is loaded with AutoModel.from_pretrained(trust_remote_code=True);
     torch_dtype is passed explicitly from the global --dtype.
  2. task/prompt: jina-clip-v2 has a single query-side LoRA adapter on the text
     tower, "retrieval.query" (which wraps query text with "Represent the query
     for retrieving evidence documents: "). There is no "retrieval.passage"
     adapter; task=None means default weights. Queries therefore use
     task="retrieval.query" and documents use task=None.
  3. encode_text / encode_image take `truncate_dim`; the global --dim is mapped to it.
  4. Whether encode_text accepts the `task` keyword differs across remote-code
     versions. It is probed once in load(), with a TypeError fallback that calls
     without `task`, so remote-code version differences do not crash the worker.
"""

import numpy as np
import torch

from .base import EmbeddingBackend
from .common import l2_normalize, parse_torch_dtype, to_numpy_f32

# Only the query side of jina-clip-v2's text tower has this LoRA task adapter
# (text_config.hf_model_config_kwargs.lora_adaptations contains only "retrieval.query");
# documents use None. Do not invent a "retrieval.passage" task.
_QUERY_TASK = "retrieval.query"


class JinaClipV2Backend(EmbeddingBackend):
    # CLIP-based (contrastive dual-tower) model; GR-CLIP mean-shift calibration only
    # applies to this family (see base.EmbeddingBackend.is_clip_based).
    is_clip_based = True

    supports_fused_text_image = False
    max_images_per_record = 1

    def default_instruction(self, role: str) -> str:
        # Instructions are handled by the model's internal LoRA task adapter (see above);
        # no instruction text needs to be prepended here.
        return ""

    def load(self) -> None:
        from .common import ensure_mistral_regex_disabled

        # The released config.json of jina-clip-v2 has no transformers_version field and the
        # text tower (XLM-RoBERTa Large) has a large vocabulary, so transformers 4.57+/5.x
        # misdetects it as "suspected Mistral regex" (see common.ensure_mistral_regex_disabled).
        # The tokenizer is loaded inside trust_remote_code model code, so the check must be
        # short-circuited before loading the model.
        ensure_mistral_regex_disabled()

        from transformers import AutoModel

        checkpoint = self.args_dict["checkpoint"]
        dtype = parse_torch_dtype(self.args_dict.get("dtype", "bfloat16"))
        self.dim = self.args_dict.get("dim")

        print(f"[worker] loading jina-clip-v2: {checkpoint}", flush=True)

        self.model = AutoModel.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        self.model.to(self.device)
        self.model.eval()

        # Probe once in load() whether encode_text accepts a `task` keyword argument,
        # instead of try/except on every batch.
        import inspect

        try:
            sig = inspect.signature(self.model.encode_text)
            self._encode_text_supports_task = "task" in sig.parameters
        except (TypeError, ValueError):
            # Some remote-code implementations have no inspectable signature; assume
            # support and rely on the TypeError fallback at call time.
            self._encode_text_supports_task = True

        print(
            f"[worker] jina-clip-v2 ready, "
            f"encode_text(task=...) supported={self._encode_text_supports_task}",
            flush=True,
        )

    def build_inputs_cpu(self, items: list[dict]):
        from .common import load_images_parallel

        # Load all images of the batch in parallel.
        image_workers = self.args_dict.get("image_load_workers", 8)
        image_paths = [
            (item.get("images") or [None])[0] if item.get("images") else None
            for item in items
        ]
        pil_images = load_images_parallel(image_paths, max_workers=image_workers)

        prepared = []
        for item, pil in zip(items, pil_images):
            prepared.append({"text": item.get("text"), "pil": pil})
        return prepared

    def _encode_text(self, texts: list[str], role: str) -> np.ndarray:
        kwargs = {}
        if self.dim:
            kwargs["truncate_dim"] = self.dim

        if role == "query" and self._encode_text_supports_task:
            kwargs["task"] = _QUERY_TASK

        try:
            embeds = self.model.encode_text(texts, **kwargs)
        except TypeError:
            # Fallback: the installed remote code does not accept `task`; call without it
            # and remember the result to avoid repeated TypeErrors in this process.
            self._encode_text_supports_task = False
            kwargs.pop("task", None)
            embeds = self.model.encode_text(texts, **kwargs)

        return to_numpy_f32(embeds)

    def _encode_image(self, images: list) -> np.ndarray:
        kwargs = {}
        if self.dim:
            kwargs["truncate_dim"] = self.dim
        embeds = self.model.encode_image(images, **kwargs)
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
            text_embeds = self._encode_text(text_batch, role)
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
