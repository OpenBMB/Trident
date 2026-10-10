"""Loss interface contract.

**Every loss receives the same `LossInput`**, which contains both local tensors
and tensors gathered across GPUs, so switching between InfoNCE / distillation /
multi-positive / Matryoshka / ranking losses requires no changes to the trainer,
model or data.

Tensor conventions (D = embedding dim, B = samples per GPU, W = number of GPUs,
Nq = query views, Np = positive views, G = Np + number of hard negatives):
    q      [B*Nq, D]     local query embeddings ([B, D] when Nq=1)
    d      [B, G, D]     local doc embeddings; d[:, :Np] are positives, d[:, Np:] hard negatives
    q_all  [B*W*Nq, D]   queries gathered across GPUs
    d_all  [B*W, G, D]   docs gathered across GPUs
    offset               row of the first local query in q_all = rank * B * Nq
    doc_offset           row of the first local sample in d_all = rank * B (defaults to offset)
    q_valid   [B, Nq]    False = view duplicated by the collator for shape alignment; excluded from the loss
    d_valid   [B, Np]    same, for positive slots
    q_valid_all / d_valid_all   cross-GPU versions of the two above

For single-view data (Nq = Np = 1) these conventions reduce to standard InfoNCE inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class LossInput:
    q: torch.Tensor
    d: torch.Tensor
    q_all: torch.Tensor
    d_all: torch.Tensor
    offset: int = 0
    world_size: int = 1
    meta: Dict[str, Any] = field(default_factory=dict)
    # ------- multi-view / multi-positive fields (defaults = single view) -------
    num_query_views: int = 1
    num_positives: int = 1
    doc_offset: Optional[int] = None
    q_valid: Optional[torch.Tensor] = None
    d_valid: Optional[torch.Tensor] = None
    q_valid_all: Optional[torch.Tensor] = None
    d_valid_all: Optional[torch.Tensor] = None
    # View validity of hard-negative slots [B, R, Np] (only with multiview_negatives=true).
    # R = hard negatives per sample, Np = views per hard negative (same shape as positives).
    neg_valid: Optional[torch.Tensor] = None
    neg_valid_all: Optional[torch.Tensor] = None

    @property
    def batch_size(self) -> int:
        """Number of local samples B (note: query rows are B*Nq)."""
        return self.d.size(0)

    @property
    def num_queries(self) -> int:
        return self.q.size(0)

    @property
    def group_size(self) -> int:
        return self.d.size(1)

    @property
    def num_negatives(self) -> int:
        return self.group_size - self.num_positives

    @property
    def d_start(self) -> int:
        """Index of the first local sample in d_all."""
        return int(self.offset if self.doc_offset is None else self.doc_offset)

    @property
    def is_multiview(self) -> bool:
        return self.num_query_views > 1 or self.num_positives > 1

    @property
    def teacher_scores(self) -> Optional[torch.Tensor]:
        return self.meta.get("teacher_scores")


@dataclass
class LossOutput:
    loss: torch.Tensor
    metrics: Dict[str, float] = field(default_factory=dict)


class BaseLoss(nn.Module):
    """Subclasses only implement forward(LossInput) -> LossOutput.

    A loss is an nn.Module, so it may hold learnable parameters (e.g. a learnable
    temperature), which are handled by the optimizer and DDP.
    """

    def forward(self, x: LossInput) -> LossOutput:  # type: ignore[override]
        raise NotImplementedError
