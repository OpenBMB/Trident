"""Abstract encoder base class.

**A new model only needs to implement two methods**:
    build_inputs(records, role) -> Dict[str, Tensor]     # data -> model inputs
    encode_features(**inputs)   -> Tensor [B, D]         # model inputs -> embeddings

Everything else (pooling, projection head, L2 normalization, Matryoshka
truncation, freezing, LoRA, save/load, batched inference) is provided by the base class.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.schema import Record
from ..utils.misc import count_params, get_logger, human
from .pooling import build_pooler

logger = get_logger(__name__)


class BaseEmbedder(nn.Module):
    CONFIG_NAME = "mmemb_model.json"
    EXTRA_NAME = "mmemb_extra.bin"

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg: Dict[str, Any] = dict(cfg)
        self.normalize: bool = bool(self.cfg.get("normalize", True))
        self.pooler = build_pooler(self.cfg.get("pooling", "last_token"))
        self.projection: Optional[nn.Module] = None
        self._hidden_size: Optional[int] = None

    # ---------------- must be implemented by subclasses ----------------
    def build_inputs(self, records: Sequence[Record], role: str = "query") -> Dict[str, Any]:
        raise NotImplementedError

    def encode_features(self, **inputs) -> torch.Tensor:
        raise NotImplementedError

    # ---------------- shared functionality ----------------
    def _init_projection(self, hidden_size: int) -> None:
        self._hidden_size = hidden_size
        out_dim = self.cfg.get("embed_dim")
        if out_dim and int(out_dim) != hidden_size:
            bias = bool(self.cfg.get("projection_bias", False))
            self.projection = nn.Linear(hidden_size, int(out_dim), bias=bias)
            nn.init.normal_(self.projection.weight, std=hidden_size**-0.5)
            if bias:
                nn.init.zeros_(self.projection.bias)
            logger.info("Projection head enabled: %d -> %d", hidden_size, int(out_dim))

    @property
    def embedding_dim(self) -> int:
        if self.projection is not None:
            return self.projection.out_features
        assert self._hidden_size is not None, "subclasses must call _init_projection() in __init__"
        return self._hidden_size

    def post_pool(self, pooled: torch.Tensor) -> torch.Tensor:
        if self.projection is not None:
            pooled = self.projection(pooled.to(self.projection.weight.dtype))
        pooled = pooled.float()  # the loss is always computed in fp32 for numerical stability
        if self.normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled

    def forward(self, **inputs) -> torch.Tensor:
        return self.encode_features(**inputs)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # ---------------- inference helpers ----------------
    @torch.no_grad()
    def encode(
        self,
        records: Sequence[Record],
        role: str = "query",
        batch_size: int = 8,
        dim: Optional[int] = None,
        show_progress: bool = False,
    ) -> torch.Tensor:
        self.eval()
        outs: List[torch.Tensor] = []
        rng = range(0, len(records), batch_size)
        if show_progress:
            try:
                from tqdm.auto import tqdm

                rng = tqdm(rng, desc=f"encode[{role}]")
            except ImportError:
                pass
        for i in rng:
            chunk = list(records[i : i + batch_size])
            inputs = self.build_inputs(chunk, role=role)
            inputs = {
                k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in inputs.items()
            }
            emb = self.encode_features(**inputs)
            outs.append(emb.detach().float().cpu())
        embs = torch.cat(outs, dim=0) if outs else torch.empty(0)
        if dim:
            embs = F.normalize(embs[:, :dim], p=2, dim=-1) if self.normalize else embs[:, :dim]
        return embs

    # ---------------- parameter freezing ----------------
    def apply_freeze(self, freeze_patterns: Sequence[str] = (), unfreeze_patterns: Sequence[str] = ()) -> None:
        if freeze_patterns:
            for name, p in self.named_parameters():
                if any(re.search(pat, name) for pat in freeze_patterns):
                    p.requires_grad_(False)
        if unfreeze_patterns:
            for name, p in self.named_parameters():
                if any(re.search(pat, name) for pat in unfreeze_patterns):
                    p.requires_grad_(True)
        stat = count_params(self)
        logger.info(
            "Parameters total=%s trainable=%s (%.2f%%)",
            human(stat["total"]),
            human(stat["trainable"]),
            100.0 * stat["trainable"] / max(stat["total"], 1),
        )

    # ---------------- save / load ----------------
    def _save_backbone(self, save_dir: str) -> None:
        raise NotImplementedError

    def save_pretrained(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        self._save_backbone(save_dir)
        extra = {k: v for k, v in self.state_dict().items() if not k.startswith("backbone.")}
        if extra:
            torch.save(extra, os.path.join(save_dir, self.EXTRA_NAME))
        with open(os.path.join(save_dir, self.CONFIG_NAME), "w", encoding="utf-8") as f:
            json.dump(self.cfg, f, ensure_ascii=False, indent=2)
        logger.info("Encoder saved to %s", save_dir)

    def load_extra(self, save_dir: str) -> None:
        path = os.path.join(save_dir, self.EXTRA_NAME)
        if os.path.isfile(path):
            state = torch.load(path, map_location="cpu")
            missing, unexpected = self.load_state_dict(state, strict=False)
            logger.info("Loaded extra weights: %d tensors (unexpected=%d)", len(state), len(unexpected))
