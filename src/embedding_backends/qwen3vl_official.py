#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
embedding_backends/qwen3vl_official.py

Backend for the official Qwen3-VL-Embedding models (baseline). Not to be confused
with trident_qwen3vl, which loads checkpoints trained with this repository.

Requires the official `qwen3_vl_embedding.py` (class Qwen3VLEmbedder) from the
Qwen3-VL-Embedding release: place it in `src/` or pass its path with
--qwen3vl_official_script.
"""

import os
import sys

import numpy as np
import torch

from .base import EmbeddingBackend
from .common import l2_normalize, parse_torch_dtype


class Qwen3VLOfficialBackend(EmbeddingBackend):
    supports_fused_text_image = True
    # Supports --matryoshka_dims: several truncation dims from one forward (see base.py).
    # The official Qwen3-VL-Embedding supports MRL: take the first d dims and L2-normalize
    # each segment (same as trident_qwen3vl).
    supports_matryoshka = True
    max_images_per_record = 1

    def default_instruction(self, role: str) -> str:
        return "Represent the user's input."

    def load(self) -> None:
        checkpoint = self.args_dict["checkpoint"]
        script_path = self.args_dict.get("qwen3vl_official_script")

        if script_path:
            if not os.path.isfile(script_path):
                raise FileNotFoundError(
                    f"Path given by --qwen3vl_official_script does not exist: {script_path}"
                )

            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "qwen3_vl_embedding_official", script_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[union-attr]
            Qwen3VLEmbedder = module.Qwen3VLEmbedder

            print(
                f"[worker] loaded Qwen3VLEmbedder from explicit path: {script_path}",
                flush=True,
            )
        else:
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if script_dir not in sys.path:
                sys.path.insert(0, script_dir)

            try:
                from qwen3_vl_embedding import Qwen3VLEmbedder
            except ImportError as exc:
                raise ImportError(
                    f"Could not find qwen3_vl_embedding.py (with Qwen3VLEmbedder) in "
                    f"{script_dir}. Place the official file in that directory, or pass "
                    f"its path explicitly with --qwen3vl_official_script."
                ) from exc

            print(
                f"[worker] loaded Qwen3VLEmbedder from same directory: {script_dir}",
                flush=True,
            )

        dtype = parse_torch_dtype(self.args_dict.get("dtype", "bfloat16"))

        print(
            f"[worker] loading Qwen3-VL-Embedding (official): {checkpoint}",
            flush=True,
        )

        kwargs = {"model_name_or_path": checkpoint, "torch_dtype": dtype}
        attn_impl = self.args_dict.get("attn_implementation")
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl

        self.model = Qwen3VLEmbedder(**kwargs)

        underlying = getattr(self.model, "model", None) or getattr(
            self.model, "backbone", None
        )
        if underlying is not None and hasattr(underlying, "to"):
            underlying.to(self.device)

        # The full dimension is only needed with --matryoshka_dims; otherwise nothing extra is done.
        if self.args_dict.get("matryoshka_dims"):
            self.embedding_dim = self._detect_embedding_dim()
            self.setup_matryoshka(self.embedding_dim)

        print(
            f"[worker] Qwen3-VL-Embedding (official) ready"
            + (f", embedding_dim={self.embedding_dim}" if self.embedding_dim else ""),
            flush=True,
        )

    def _detect_embedding_dim(self) -> int:
        """
        The official Qwen3VLEmbedder has no common "output dimension" attribute
        (it differs across checkpoint sizes), so one real forward on a very short
        text determines it from the actual output. This costs one batch=1 forward
        per worker at load time.
        """
        array = self._forward_raw(
            [{"text": "hello"}], self.default_instruction("doc")
        )
        if array.ndim != 2 or array.shape[0] != 1 or array.shape[1] <= 0:
            raise RuntimeError(
                f"Failed to probe the qwen3vl_official output dimension; warm-up gave shape={array.shape}"
            )
        return int(array.shape[1])

    @torch.inference_mode()
    def _forward_raw(self, prepared, instruction) -> np.ndarray:
        """One forward of the official process(); returns float32 numpy (before our own L2 normalization)."""
        try:
            embeddings = self.model.process(prepared, instruction=instruction)
        except TypeError:
            embeddings = self.model.process(prepared)

        if isinstance(embeddings, torch.Tensor):
            return embeddings.detach().float().cpu().numpy()
        return np.asarray(embeddings, dtype=np.float32)

    def build_inputs_cpu(self, items: list[dict]):
        prepared = []
        for item in items:
            images = item.get("images") or []
            entry: dict = {}
            if item.get("text"):
                entry["text"] = item["text"]
            if images:
                entry["image"] = images[0]
            prepared.append(entry)
        return prepared

    @torch.inference_mode()
    def compute_from_inputs(self, prepared, items, role):
        instruction = None
        for item in items:
            if item.get("instruction"):
                instruction = item["instruction"]
                break
        if instruction is None:
            instruction = self.default_instruction(role)

        array = self._forward_raw(prepared, instruction)

        if self.matryoshka_pack_dims:
            # One forward, all Matryoshka dims packed; the driver unpacks them when merging.
            # Truncation is done on the raw float32 embeddings, each segment L2-normalized.
            array = self.matryoshka_pack(array)
        else:
            array = l2_normalize(array)

        if array.shape[0] != len(items):
            raise ValueError(
                f"Model returned {array.shape[0]} embeddings "
                f"for {len(items)} inputs."
            )

        token_infos = [(-1, -1, -1) for _ in items]
        return array, token_infos
