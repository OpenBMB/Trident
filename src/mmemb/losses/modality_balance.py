"""Modality-balance loss (dispersion penalty on k x k similarity blocks).

Note: this is **not** part of the main Trident objective. It is the explicit
"Modality Balance Loss" studied in Appendix G of the paper ("with Modality
Balance Loss", w = 200). By default it is enabled with `weight: 0`, i.e. it is
only computed for logging (`balance/*` metrics) and does not affect training.

## What it does

Block-diagonal InfoNCE only requires positive cells of a k x k block to score
higher than negatives; it does not constrain **how balanced the cells inside a
block are**. In multimodal data some modality pairs naturally score higher
(e.g. text-query <-> text-doc vs. text-query <-> image-doc), so retrieval is
dominated by the modality that is easiest to score high.

This loss directly reduces **the dispersion inside the k x k cosine-similarity
matrix of the same (query, doc) pair**, pulling the nq x npos values of
cos(q_i, d_j) towards each other. For a 1x3 layout (one query, three positive
views) it makes cos(q, d_txt), cos(q, d_img), cos(q, d_fused) as close as possible.

## Scope

  scope: positives   reduce dispersion only in the positive k x k block of each matched pair.
  scope: all         positive blocks + negative blocks "query vs. other docs in the batch".
                     Cross-sample negative blocks are also nq x npos; for single-view hard
                     negatives the block degenerates to "nq query views vs. one negative
                     vector" (query-side modality bias). Negative blocks use **local
                     in-batch** docs (this is a regularizer; global scale is not needed).

## Centers (configured separately for positives / negatives)

  pos_center: mean (default)  pull every cell towards the block mean (variance minimization);
                              symmetric: high-scoring modalities move down, low-scoring ones up.
  pos_center: max             only lift cells **below the block maximum**; the maximum itself is
                              not pulled down (always detached): "lift weak modalities to the
                              level of the strong one".

  neg_center: mean (default)  same as above, for negative blocks.
  neg_center: min             mirror of max: only push cells **above the block minimum** down;
                              the minimum (the best-separated negative) is not lifted.

`pos_center=max` + `neg_center=min` only move cells away from the "already
well-learned" end and never move that end itself, so they never directly
oppose the main InfoNCE objective.

`center` (deprecated) is still accepted and is treated as `pos_center` only.

## Usage (as an optional module of `infonce`)

    loss:
      type: infonce
      modules:
        modality_balance:
          enable: true        # true by default
          weight: 200         # 0 = log only, no gradient (ablation control)
          scope: all          # positives | all
          pos_center: max     # mean | max -- positive blocks
          neg_center: min     # mean | min -- negative blocks (scope=all only)

**Ablation convention**: with `weight: 0` the term is still computed and logged
to TensorBoard, but does not take part in backprop (computed under
`torch.no_grad()`), so the `balance/*` curves of runs with and without the term
can be plotted together.

It can also be used standalone (`type: modality_balance`) as a pure
regularizer without a retrieval objective (mainly useful for unit tests).

## Notes

* The term is only meaningful for **multi-view / multi-positive data** (nq>1 or
  npos>1); for single-view data the block is 1x1, the value is always 0 and no
  balance metrics are logged.
* `balance/pos_range` is the mean max-min within positive blocks, and
  `balance/neg_cross_range` / `balance/neg_hard_range` are the mean max-min within
  negative blocks. Their magnitude before training indicates how strong the
  modality bias is; they should decrease steadily once the term is enabled. These
  linear-scale range metrics are better suited for judging the bias (and for
  ablation plots) than `balance/loss` (variance scale, structurally small).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from ..registry import LOSSES
from ..utils.misc import get_logger
from .base import BaseLoss, LossInput, LossOutput

logger = get_logger(__name__)

_VALID_POS_CENTER = ("mean", "max")
_VALID_NEG_CENTER = ("mean", "min")
_VALID_METRIC = ("variance", "std", "pairwise")
_VALID_SCOPE = ("positives", "all")

# Metric prefix: groups all balance metrics together in TensorBoard / wandb
PREFIX = "balance/"


def _block_dispersion(
    sim: torch.Tensor,
    valid: torch.Tensor,
    center: str,
    detach_center: bool,
    metric: str,
) -> Tuple[torch.Tensor, float]:
    """Mean dispersion over a batch of blocks (differentiable).

    sim / valid: [M blocks, K], where K is the number of cells of a flattened block.
    Dispersion is computed over **valid cells** of each row (block); only blocks with
    at least 2 valid cells are counted (dispersion is undefined otherwise).

    center:
      mean -- pull towards the block mean (symmetric).
      max  -- only lift cells below the block maximum; the maximum is detached.
              Suited for positives: weak modalities catch up with the strong one.
      min  -- only push cells above the block minimum down; the minimum is detached.
              Suited for negatives: high negatives move towards the best-separated one.

    Returns (scalar loss, diagnostic: mean max-min range over valid blocks).
    """
    sim = sim.float()
    valid = valid.bool()
    valid_f = valid.to(sim.dtype)
    n = valid_f.sum(dim=-1)                       # [M] valid cells per block
    ok = n >= 2                                   # dispersion needs at least 2 cells
    n_safe = n.clamp(min=1.0)

    ssum = (sim * valid_f).sum(dim=-1)
    mean = ssum / n_safe

    if center == "max":
        neg_inf = torch.finfo(sim.dtype).min
        mx = sim.masked_fill(~valid, neg_inf).amax(dim=-1).detach()  # do not pull the maximum down
        deficit = (mx.unsqueeze(-1) - sim).clamp(min=0.0) * valid_f
        var = (deficit ** 2).sum(dim=-1) / n_safe
    elif center == "min":
        pos_inf = torch.finfo(sim.dtype).max
        mn = sim.masked_fill(~valid, pos_inf).amin(dim=-1).detach()  # do not lift the minimum
        excess = (sim - mn.unsqueeze(-1)).clamp(min=0.0) * valid_f
        var = (excess ** 2).sum(dim=-1) / n_safe
    else:  # mean
        c = mean.detach() if detach_center else mean
        centered = (sim - c.unsqueeze(-1)) * valid_f
        var = (centered ** 2).sum(dim=-1) / n_safe

    if metric == "pairwise":
        # mean pairwise squared difference of valid cells = 2n/(n-1) * population variance
        var = var * (2.0 * n_safe / (n_safe - 1.0).clamp(min=1.0))

    per_block = var
    if metric == "std":
        per_block = (var.clamp(min=0.0) + 1e-12).sqrt()

    ok_f = ok.to(sim.dtype)
    denom = ok_f.sum().clamp(min=1.0)
    loss = (per_block * ok_f).sum() / denom

    with torch.no_grad():
        neg_inf = torch.finfo(sim.dtype).min
        pos_inf = torch.finfo(sim.dtype).max
        blk_max = sim.masked_fill(~valid, neg_inf).amax(dim=-1)
        blk_min = sim.masked_fill(~valid, pos_inf).amin(dim=-1)
        rng = ((blk_max - blk_min) * ok_f).sum() / denom
    return loss, float(rng.item())


@LOSSES.register("modality_balance")
class ModalityBalanceLoss(BaseLoss):
    """Reduce dispersion inside k x k cosine-similarity blocks to remove modality score bias."""

    def __init__(
        self,
        scope: str = "positives",           # positives | all
        pos_center: str = "mean",           # mean | max  -- center for positive blocks
        neg_center: str = "mean",           # mean | min  -- center for negative blocks (scope=all only)
        center: Optional[str] = None,       # deprecated; equivalent to setting pos_center
        detach_center: bool = False,        # center=mean only: whether to detach the center
        metric: str = "variance",           # variance | std | pairwise
        neg_weight: float = 1.0,            # weight of negative blocks relative to positive blocks (scope=all)
        include_hard_negatives: bool = True,  # scope=all: also regularize own hard negatives
        prefix: str = PREFIX,               # metric prefix
        **_ignored: Any,                    # absorbs outer keys such as enable / weight / log_every
    ) -> None:
        super().__init__()
        if scope not in _VALID_SCOPE:
            raise ValueError(f"modality_balance scope must be one of {_VALID_SCOPE}")

        if center is not None:
            # `center` only sets pos_center; negative blocks must set neg_center explicitly.
            logger.warning(
                "modality_balance `center` is deprecated; use `pos_center` (positive blocks) "
                "and `neg_center` (negative blocks). `center=%r` is applied to positive "
                "blocks only; set `neg_center` explicitly for negative blocks (default mean).",
                center,
            )
            pos_center = center

        if pos_center not in _VALID_POS_CENTER:
            raise ValueError(f"modality_balance pos_center must be one of {_VALID_POS_CENTER}")
        if neg_center not in _VALID_NEG_CENTER:
            raise ValueError(f"modality_balance neg_center must be one of {_VALID_NEG_CENTER}")
        if metric not in _VALID_METRIC:
            raise ValueError(f"modality_balance metric must be one of {_VALID_METRIC}")

        self.scope = scope
        self.pos_center = pos_center
        self.neg_center = neg_center
        self.detach_center = bool(detach_center)
        self.metric = metric
        self.neg_weight = float(neg_weight)
        self.include_hard_negatives = bool(include_hard_negatives)
        self.prefix = str(prefix or "")
        self._warned_single_view = False

    # ------------------------------------------------------------------
    def _disp(self, sim: torch.Tensor, valid: torch.Tensor, center: str) -> Tuple[torch.Tensor, float]:
        return _block_dispersion(sim, valid, center, self.detach_center, self.metric)

    def _key(self, name: str) -> str:
        return f"{self.prefix}{name}"

    def _hard_negative_valid(
        self, x: LossInput, B: int, nh: int, ref: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """View validity of hard-negative slots [B, nh] (only meaningful with multiview_negatives=true)."""
        nv = x.neg_valid
        if nv is None:
            return None
        nv = nv.to(ref.device).bool()
        if nv.numel() != B * nh:
            return None  # shape mismatch: treat as unavailable and assume all valid (safe degradation)
        return nv.reshape(B, nh)

    def forward(self, x: LossInput) -> LossOutput:  # type: ignore[override]
        zero = x.q.new_zeros(())
        if not x.is_multiview:
            if not self._warned_single_view:
                logger.warning(
                    "modality_balance requires multi-view data "
                    "(data.num_query_views or data.num_positives > 1); "
                    "the data is single-view (1x1 blocks), so the term is always 0."
                )
                self._warned_single_view = True
            return LossOutput(loss=zero, metrics={})

        B, nq, npos = x.batch_size, x.num_query_views, x.num_positives
        dev = x.q.device

        Q = F.normalize(x.q.float().reshape(B, nq, -1), p=2, dim=-1)   # [B, nq, D]
        Dp = F.normalize(x.d[:, :npos].float(), p=2, dim=-1)          # [B, npos, D]
        qv = (
            x.q_valid.to(dev).bool()
            if x.q_valid is not None
            else torch.ones(B, nq, dtype=torch.bool, device=dev)
        )
        dv = (
            x.d_valid.to(dev).bool()
            if x.d_valid is not None
            else torch.ones(B, npos, dtype=torch.bool, device=dev)
        )

        # ---- positive blocks: one nq x npos block per matched pair ----
        Sp = torch.einsum("bid,bjd->bij", Q, Dp)             # [B, nq, npos]
        Mp = qv.unsqueeze(2) & dv.unsqueeze(1)               # [B, nq, npos]
        pos_loss, pos_range = self._disp(
            Sp.reshape(B, nq * npos), Mp.reshape(B, nq * npos), self.pos_center
        )

        metrics: Dict[str, float] = {
            self._key("pos"): float(pos_loss.item()),
            self._key("pos_range"): pos_range,
        }
        total = pos_loss

        # ---- negative blocks (scope=all only) ----
        if self.scope == "all":
            neg_terms = []

            # cross-sample doc blocks: views of query b x positive views of another sample's doc c, nq x npos
            Sc = torch.einsum("bid,cjd->bcij", Q, Dp)        # [B, B, nq, npos]
            offdiag = ~torch.eye(B, dtype=torch.bool, device=dev)
            Mc = (
                qv.view(B, 1, nq, 1)
                & dv.view(1, B, 1, npos)
                & offdiag.view(B, B, 1, 1)
            )
            # diagonal blocks b==c are fully masked (n=0) and skipped by _block_dispersion
            cross_loss, cross_range = self._disp(
                Sc.reshape(B * B, nq * npos), Mc.reshape(B * B, nq * npos), self.neg_center
            )
            neg_terms.append(cross_loss)
            metrics[self._key("neg_cross")] = float(cross_loss.item())
            metrics[self._key("neg_cross_range")] = cross_range

            # own hard negatives: single view, block degenerates to "nq query views -> one negative vector"
            if self.include_hard_negatives and x.group_size > npos:
                Dn = F.normalize(x.d[:, npos:].float(), p=2, dim=-1)  # [B, nh, D]
                nh = Dn.size(1)
                Sh = torch.einsum("bid,bgd->bgi", Q, Dn)             # [B, nh, nq]
                Mh = qv.view(B, 1, nq).expand(B, nh, nq)
                nv = self._hard_negative_valid(x, B, nh, Q)
                if nv is not None:  # hard negatives padded to several views: exclude padded slots
                    Mh = Mh & nv.view(B, nh, 1)
                hn_loss, hn_range = self._disp(
                    Sh.reshape(B * nh, nq), Mh.reshape(B * nh, nq), self.neg_center
                )
                neg_terms.append(hn_loss)
                metrics[self._key("neg_hard")] = float(hn_loss.item())
                metrics[self._key("neg_hard_range")] = hn_range

            neg_loss = torch.stack(neg_terms).mean() if neg_terms else zero
            metrics[self._key("neg")] = float(neg_loss.item())
            total = pos_loss + self.neg_weight * neg_loss

        metrics[self._key("loss")] = float(total.item())
        return LossOutput(loss=total, metrics=metrics)


# Alias: `type: sim_balance` is equivalent to `type: modality_balance`
LOSSES.register("sim_balance")(ModalityBalanceLoss)
LOSSES.register("view_balance")(ModalityBalanceLoss)
