"""Pooling strategies, all registered in POOLERS; select with `model.pooling: last_token`.

The implementation does not depend on padding_side (positions are correct for
both left and right padding).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..registry import POOLERS


def _last_index(attention_mask: torch.Tensor) -> torch.Tensor:
    """Index of the last non-padding token of each sequence; correct for left and right padding."""
    positions = torch.arange(attention_mask.size(1), device=attention_mask.device)
    positions = positions.unsqueeze(0).expand_as(attention_mask)
    masked = torch.where(attention_mask.bool(), positions, torch.full_like(positions, -1))
    return masked.max(dim=1).values.clamp(min=0)


def _first_index(attention_mask: torch.Tensor) -> torch.Tensor:
    big = attention_mask.size(1)
    positions = torch.arange(big, device=attention_mask.device).unsqueeze(0).expand_as(attention_mask)
    masked = torch.where(attention_mask.bool(), positions, torch.full_like(positions, big))
    return masked.min(dim=1).values.clamp(max=big - 1)


class BasePooler(nn.Module):
    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@POOLERS.register("last_token")
class LastTokenPooler(BasePooler):
    """Official Qwen3-Embedding / Qwen3-VL-Embedding pooling: the last valid token."""

    def forward(self, hidden, attention_mask):
        idx = _last_index(attention_mask)
        return hidden[torch.arange(hidden.size(0), device=hidden.device), idx]


@POOLERS.register("mean")
class MeanPooler(BasePooler):
    def forward(self, hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)


@POOLERS.register("cls")
class FirstTokenPooler(BasePooler):
    def forward(self, hidden, attention_mask):
        idx = _first_index(attention_mask)
        return hidden[torch.arange(hidden.size(0), device=hidden.device), idx]


@POOLERS.register("weighted_mean")
class WeightedMeanPooler(BasePooler):
    """SGPT-style position-weighted mean: later tokens get larger weights."""

    def forward(self, hidden, attention_mask):
        mask = attention_mask.to(hidden.dtype)
        weights = torch.cumsum(mask, dim=1) * mask
        weights = weights.unsqueeze(-1)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-6)


@POOLERS.register("max")
class MaxPooler(BasePooler):
    def forward(self, hidden, attention_mask):
        neg_inf = torch.finfo(hidden.dtype).min
        pad = attention_mask.unsqueeze(-1) == 0  # do not use ~ on long tensors (bitwise not)
        masked = hidden.masked_fill(pad, neg_inf)
        return masked.max(dim=1).values


def build_pooler(name: str, **kwargs) -> BasePooler:
    return POOLERS.build(name, **kwargs)
