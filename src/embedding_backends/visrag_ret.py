#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
embedding_backends/visrag_ret.py

VisRAG-Ret (openbmb/VisRAG-Ret) backend (baseline). A document embedding model
built on MiniCPM-V 2.0 (SigLIP vision encoder + MiniCPM-2B language model),
loaded from modelscope with trust_remote_code=True (custom VisRAG_Ret class).

Fusion: the official README / demo only shows all-text batches (image = [None]*bs)
or all-image batches (text = ['']*bs), not records with both.

However, the source (modeling_visrag_ret.py::VisRAG_Ret.forward / prepare_context)
supports joint text+image encoding per record:
    text_, image_ = inputs
    content = text_
    if image_:
        content = <image placeholder> + "\n" + content
i.e. the image of each record is optional; when present, the image placeholder
is prepended to the text and both go through the same forward. Therefore
supports_fused_text_image = True and records with both are jointly encoded in
one forward instead of CLIP-style late fusion.

Instruction prefix (as in the official demo): only queries get the prefix
"Represent this query for retrieving relevant documents: "; documents (image
only or text+image) get no prefix.

Dependencies:
    pip install modelscope
(modelscope is a heavy dependency and is only imported inside load(), when this
backend is actually selected.)
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .base import EmbeddingBackend
from .common import l2_normalize, load_images_parallel, parse_torch_dtype


def weighted_mean_pooling(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Weighted mean pooling kept verbatim from the official demo: later valid tokens
    get larger weights (the cumsum of attention_mask increases with position).
    """
    attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
    s = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
    d = attention_mask_.sum(dim=1, keepdim=True).float()
    reps = s / d
    return reps


class VisragRetBackend(EmbeddingBackend):
    # See the module docstring: the underlying forward supports joint text+image
    # encoding per record, so no late fusion is needed.
    supports_fused_text_image = True
    max_images_per_record = 1

    def default_instruction(self, role: str) -> str:
        # As in the official demo: only queries get the retrieval prefix; documents
        # (image only / text+image) get none.
        return (
            "Represent this query for retrieving relevant documents: "
            if role == "query"
            else ""
        )

    def load(self) -> None:
        # modelscope is a heavy dependency; only imported when this backend is used.
        from modelscope import AutoModel, AutoTokenizer

        checkpoint = self.args_dict["checkpoint"]
        dtype = parse_torch_dtype(self.args_dict.get("dtype", "bfloat16"))

        print(f"[worker] loading VisRAG-Ret: {checkpoint} (dtype={dtype})", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            checkpoint,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        # The official demo uses .cuda(); .to(self.device) is used instead for the
        # multi-GPU worker scheduling (each subprocess is bound to one GPU).
        self.model = self.model.to(self.device)
        self.model.eval()

        # max_inp_length of forward() defaults to 2048; can be overridden from the command line.
        self.max_inp_length = self.args_dict.get("visrag_max_inp_length", 2048)

        print("[worker] VisRAG-Ret ready", flush=True)

    def build_inputs_cpu(self, items: list[dict]):
        # Load the images of a batch in parallel, consistent with the other backends
        # (pure CPU / network I/O, safe in prefetch threads).
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

            raw_text = item.get("text") or ""
            # VisRAG_Ret.forward checks isinstance(text_, str), so this must always be a
            # string (never None): image-only records pass an empty string.
            text_value: str = (prefix or "") + raw_text

            prepared.append({"text": text_value, "pil_image": pil_image})
        return prepared

    @torch.inference_mode()
    def compute_from_inputs(self, prepared, items, role):
        texts = [entry["text"] for entry in prepared]
            # The `image` argument of forward() is a flat list with one image or None per
            # record, not List[List[PIL.Image]] (the internal fused_tokenize format;
            # prepare_context wraps a single image as [image_]).
        images = [entry["pil_image"] for entry in prepared]

        outputs = self.model(
            text=texts,
            image=images,
            tokenizer=self.tokenizer,
            max_inp_length=self.max_inp_length,
        )
        attention_mask = outputs.attention_mask
        hidden = outputs.last_hidden_state

        reps = weighted_mean_pooling(hidden, attention_mask)
        embeddings = F.normalize(reps, p=2, dim=1).detach().to(torch.float32).cpu().numpy()

        array = l2_normalize(embeddings)

        if array.shape[0] != len(items):
            raise ValueError(
                f"VisRAG-Ret returned {array.shape[0]} embeddings "
                f"for {len(items)} inputs."
            )

        token_infos = [(-1, -1, -1) for _ in items]
        return array, token_infos
