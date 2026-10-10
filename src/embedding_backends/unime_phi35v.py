#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
embedding_backends/unime_phi35v.py

UniME-Phi3.5-V-4.2B backend (baseline).

Unlike dedicated embedding models, this is a Phi-3.5-Vision multimodal LLM:
following the official demo, a prompt asks the model to summarize a sentence /
an image in one word, and the L2-normalized last-layer hidden state of the last
token is used as the embedding. Therefore:

  - text and images use different prompt templates and separate forwards; there
    is no official joint text+image prompt, so supports_fused_text_image = False
    and records with both use CLIP-style late fusion (clip_style_fuse);
  - at most one image per record (the official template has a single <|image_1|>);
  - it depends on modelscope's AutoProcessor / AutoModelForCausalLM
    (`pip install modelscope`).

Batching: text prompts of a batch are encoded in one forward. The image branch
**cannot** be batched: when images are given, Phi-3.5-V's processor.__call__
searches `texts` for <|image_N|> placeholders and treats it as a single
conversation string (not a batch of independent samples). Passing a list (even
of length 1) raises "expected string or bytes-like object, got 'list'". Images
are therefore processed one by one exactly like the official demo; this is a
limitation of the model's processor.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .base import EmbeddingBackend
from .common import l2_normalize, load_images_parallel, parse_torch_dtype


class UniMEPhi35VBackend(EmbeddingBackend):
    supports_fused_text_image = False
    max_images_per_record = 1

    # The two prompt templates of the official demo, kept verbatim.
    IMAGE_PROMPT_TEMPLATE = (
        "<|user|>\n<|image_1|>\nSummary above image in one word: <|end|>\n"
        "<|assistant|>\n"
    )
    TEXT_PROMPT_TEMPLATE = (
        "<|user|>\n{text}\nSummary above sentence in one word: <|end|>\n"
        "<|assistant|>\n"
    )

    def default_instruction(self, role: str) -> str:
        return ""

    @staticmethod
    def _patch_dynamic_cache_compat() -> None:
        """
        Compatibility fix: UniME-Phi3.5-V is loaded with trust_remote_code=True and
        runs the checkpoint's own (older) modeling_phi3_v.py, which calls
        `past_key_values.get_usable_length(seq_length)`, a method of the old
        Cache/DynamicCache API (transformers ~4.36-4.44). Newer transformers removed
        get_usable_length (and get_max_length / seen_tokens) in favor of
        get_seq_length(), so the remote code fails with:
            AttributeError: 'DynamicCache' object has no attribute 'get_usable_length'

        Instead of downgrading transformers globally (which could break other
        backends), a process-local monkey patch adds the old methods back on
        DynamicCache, delegating to the new API. Methods are only patched if missing.
        """
        from transformers.cache_utils import DynamicCache

        if not hasattr(DynamicCache, "get_usable_length"):

            def get_usable_length(self, new_seq_length=0, layer_idx=0):
                # Old semantics: DynamicCache has no length limit, so the usable length
                # is the current cached sequence length.
                return self.get_seq_length(layer_idx)

            DynamicCache.get_usable_length = get_usable_length

        if not hasattr(DynamicCache, "get_max_length"):

            def get_max_length(self):
                # The new API expresses "no limit" via get_max_cache_shape()
                # (always None for DynamicCache).
                if hasattr(self, "get_max_cache_shape"):
                    return self.get_max_cache_shape()
                return None

            DynamicCache.get_max_length = get_max_length

        if not hasattr(DynamicCache, "seen_tokens"):
            DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())

    def load(self) -> None:
        # modelscope is a heavy dependency; only imported when this backend is used.
        from modelscope import AutoModelForCausalLM, AutoProcessor

        self._patch_dynamic_cache_compat()

        checkpoint = self.args_dict["checkpoint"]
        dtype = parse_torch_dtype(self.args_dict.get("dtype", "float16"))
        # The official demo uses flash_attention_2; override with
        # --attn_implementation eager where it is unavailable.
        attn_impl = self.args_dict.get("attn_implementation") or "flash_attention_2"

        print(
            f"[worker] loading UniME-Phi3.5-V: {checkpoint} "
            f"(dtype={dtype}, attn_implementation={attn_impl})",
            flush=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            checkpoint, trust_remote_code=True
        )
        # The official demo forces left padding: for decoder-only models taking the last
        # token's representation, padding must be on the left so the last valid token
        # is aligned across samples of different lengths.
        self.processor.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            device_map=self.device,
            trust_remote_code=True,
            torch_dtype=dtype,
            _attn_implementation=attn_impl,
        )
        self.model.eval()

        print("[worker] UniME-Phi3.5-V ready", flush=True)

    def render_prompt(self, text, images, instruction, role) -> str:
        if images:
            return self.IMAGE_PROMPT_TEMPLATE
        return self.TEXT_PROMPT_TEMPLATE.format(text=text or "")

    def build_inputs_cpu(self, items: list[dict]):
        # Load the images of a batch in parallel, consistent with the other backends.
        image_workers = self.args_dict.get("image_load_workers", 8)
        image_paths = [
            (item.get("images") or [None])[0] if item.get("images") else None
            for item in items
        ]
        pil_images = load_images_parallel(image_paths, max_workers=image_workers)

        prepared = []
        for item, pil_image in zip(items, pil_images):
            prepared.append({"text": item.get("text"), "pil_image": pil_image})
        return prepared

    def _forward_last_token_embedding(self, model_inputs: dict) -> np.ndarray:
        """Run one forward, take the last-layer hidden state at the last token, and normalize."""
        model_inputs = {
            key: (value.to(self.model.device) if isinstance(value, torch.Tensor) else value)
            for key, value in model_inputs.items()
        }
        outputs = self.model(
            **model_inputs, output_hidden_states=True, return_dict=True
        )
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        last_hidden = F.normalize(last_hidden, dim=-1)
        return last_hidden.detach().to(torch.float32).cpu().numpy()

    @torch.inference_mode()
    def compute_from_inputs(self, prepared, items, role):
        text_idx, text_prompts = [], []
        image_idx, image_list = [], []

        for i, entry in enumerate(prepared):
            if entry["text"]:
                text_idx.append(i)
                text_prompts.append(self.TEXT_PROMPT_TEMPLATE.format(text=entry["text"]))
            if entry["pil_image"] is not None:
                image_idx.append(i)
                image_list.append(entry["pil_image"])

        text_vecs: dict[int, np.ndarray] = {}
        image_vecs: dict[int, np.ndarray] = {}

        if text_prompts:
            inputs_text = self.processor(
                text=text_prompts,
                images=None,
                return_tensors="pt",
                padding=True,
            )
            text_embeds = self._forward_last_token_embedding(dict(inputs_text))
            for offset, i in enumerate(text_idx):
                text_vecs[i] = text_embeds[offset]

        if image_list:
            # Phi-3.5-V's processor.__call__ treats `texts` as a single conversation string when
            # images are given (re.findall over <|image_N|>), not as a batch of independent samples;
            # a list (even of length 1) raises "expected string or bytes-like object, got 'list'".
            # Each image is therefore processed with its own processor call (text as a plain string)
            # and its own forward, exactly like the official demo.
            image_embed_list = []
            for pil_image in image_list:
                inputs_image = self.processor(
                    text=self.IMAGE_PROMPT_TEMPLATE,
                    images=[pil_image],
                    return_tensors="pt",
                    padding=True,
                )
                single_embed = self._forward_last_token_embedding(dict(inputs_image))
                image_embed_list.append(single_embed[0])
            for offset, i in enumerate(image_idx):
                image_vecs[i] = image_embed_list[offset]

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