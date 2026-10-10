"""InfoNCE / Multi-Positive View InfoNCE (the Trident training objective).

Configuration has two levels (defaults in configs/base.yaml):

    loss:
      type: infonce
      # 1) core loss parameters (flat)
      temperature / learnable_temperature / min_temperature
      use_inbatch_negatives / inbatch_scope: all | positives_only
      symmetric / symmetric_weight / label_smoothing
      multi_positive_reduction: mean | joint
      # 2) optional modules, all under `modules`
      modules:
        false_negative / matryoshka / modality_balance

Optional modules:

    false_negative   -- false-negative masking (other rows' positives that are
                        also my positives are removed from my denominator)
    matryoshka       -- Matryoshka (multi-dimension) joint training
    modality_balance -- explicit modality-balance regularizer (flattens the
                        dispersion inside each k x k similarity block). With
                        weight=0 it is computed for logging only (no gradient);
                        used for the ablation in Appendix G of the paper.

Multi-positive objective (paper Eq. 4). For a query q with positive-view set P
(text / image / fused views of the same document) and negatives N:

    L(q) = -1/|P| * Σ_{p∈P} log( exp(s_p/τ) / Σ_{j∈P∪N} exp(s_j/τ) )

i.e. all positive views and all negatives share **one** softmax denominator.
This is `multi_positive_reduction: mean` with
`modules.false_negative.mask_sibling_positives: false`. Setting
`mask_sibling_positives: true` instead removes the other positives of the same
sample from each positive's denominator (each positive is normalized
independently against the negatives), which removes the positive-view balance
term of Eq. 6.

The multi-view target matrix has **no hard-coded k**: the block side is
Nq x Np, so k=2 gives 2x2, k=3 gives 3x3, etc., with the same code path.

k=2 (B=2):                       k=3 (B=2, positive columns only):
        d0_a d0_b d1_a d1_b              d0_a d0_b d0_c d1_a d1_b d1_c
  q0_a    1    1    0    0        q0_a     1    1    1    0    0    0
  q0_b    1    1    0    0        q0_b     1    1    1    0    0    0
  q1_a    0    0    1    1        q0_c     1    1    1    0    0    0
  q1_b    0    0    1    1        q1_*     0    0    0    1    1    1
That is, instead of maximizing the diagonal, every k x k block on the
block diagonal is maximized.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from ..utils.misc import get_logger
from .base import BaseLoss, LossInput, LossOutput
from .false_negative import BatchIdentity, FalseNegativeMasker
from .modality_balance import ModalityBalanceLoss
from .spec import canonicalize, describe


logger = get_logger(__name__)


def _neg_inf(t: torch.Tensor) -> float:
    return torch.finfo(t.dtype).min


@LOSSES.register("infonce")
class InfoNCELoss(BaseLoss):
    def __init__(self, **cfg: Any) -> None:
        """The constructor expects a **canonical config** (see losses/spec.py).

        Legacy flat keys (`matryoshka_dims`, `mask_false_negatives`, ...) are
        translated automatically by `spec.canonicalize`.
        """
        super().__init__()
        canonical, warns = canonicalize(cfg)
        for w in warns:
            logger.warning("%s", w)
        self.config = canonical
        mods = canonical["modules"]

        # ---------------- core loss ----------------
        self.base_temperature = float(canonical.get("temperature", 0.02))
        self.min_temperature = float(canonical.get("min_temperature", 0.005))
        self.use_inbatch_negatives = bool(canonical.get("use_inbatch_negatives", True))
        self.inbatch_scope = canonical.get("inbatch_scope", "all")
        self.symmetric = bool(canonical.get("symmetric", False))
        self.symmetric_weight = float(canonical.get("symmetric_weight", 0.5))
        self.label_smoothing = float(canonical.get("label_smoothing", 0.0))
        reduction = canonical.get("multi_positive_reduction", "mean")
        if reduction not in ("mean", "joint"):
            raise ValueError("multi_positive_reduction must be mean or joint")
        self.multi_positive_reduction = reduction

        # ---------------- false-negative masking ----------------
        fn = mods["false_negative"]
        self.mask_sibling_positives = bool(fn["enable"] and fn["mask_sibling_positives"])
        self.fn_masker = FalseNegativeMasker(
            mask_false_negatives=fn["enable"] and fn["mask_positive_ids"],
            mask_same_example=fn["enable"] and fn["mask_same_example"],
            mask_same_group=fn["enable"] and fn["mask_same_group"],
            mask_by_query_id=fn["enable"] and fn["mask_by_query_id"],
        )

        # ---------------- matryoshka ----------------
        mat = mods["matryoshka"]
        self.matryoshka_dims = list(mat["dims"]) if (mat["enable"] and mat["dims"]) else None
        if self.matryoshka_dims:
            w = mat["weights"] or [1.0] * len(self.matryoshka_dims)
            assert len(w) == len(self.matryoshka_dims), "matryoshka.weights must have the same length as dims"
            total = float(sum(w))
            self.matryoshka_weights = [x / total for x in w]

        # ---------------- modality_balance (regularizer / diagnostics) ----------------
        # Convention: with weight=0 the module is **still built and evaluated every step**,
        # just without gradients, so balance/* curves of runs with and without the term
        # can be compared directly; an ablation only changes one number.
        bal = mods["modality_balance"]
        self.balance_weight = float(bal.get("weight", 0.0) or 0.0)
        if self.balance_weight < 0:
            raise ValueError(
                "loss.modules.modality_balance.weight must not be negative "
                "(a negative weight would **encourage** modality imbalance). Use weight: 0 to disable."
            )
        self.balance_log_every = max(1, int(bal.get("log_every", 1) or 1))
        self.balance: Optional[ModalityBalanceLoss] = None
        if bal["enable"]:
            self.balance = ModalityBalanceLoss(
                **{k: v for k, v in bal.items() if k not in ("enable", "weight", "log_every")}
            )
        self._balance_step = 0

        # Diagnostics: fraction of negative cells masked in the last forward (logged to TensorBoard)
        self._last_mask_ratio = 0.0

        init_scale = math.log(1.0 / self.base_temperature)
        if bool(canonical.get("learnable_temperature", False)):
            self.logit_scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        else:
            self.register_buffer("logit_scale", torch.tensor(init_scale, dtype=torch.float32))

    def describe(self) -> str:
        return describe(self.config)

    # ---------------------------------------------------------------- helpers
    @property
    def scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=1.0 / self.min_temperature)

    @staticmethod
    def _slice(t: torch.Tensor, dim: Optional[int]) -> torch.Tensor:
        if dim is None or dim >= t.size(-1):
            return t
        return F.normalize(t[..., :dim], p=2, dim=-1)

    # ------------------------------------------------------------ identity / masking
    def _identity(self, x: LossInput) -> Optional[BatchIdentity]:
        """Build (and cache within one forward) the identity table of this batch.

        Matryoshka calls _one_dim once per dimension; the identity table is
        dimension-independent, so it is built once.
        """
        if not self.fn_masker.enabled:
            return None
        cached = x.meta.get("_fn_identity", "MISS")
        if cached != "MISS":
            return cached
        ident = self.fn_masker.build(
            x.meta,
            n_examples=x.d_all.size(0),
            group_size=x.group_size,
            num_positives=x.num_positives,
            device=x.q.device,
        )
        x.meta["_fn_identity"] = ident
        return ident

    @staticmethod
    def _own_neg_slots(x: LossInput, first_neg: int) -> torch.Tensor:
        """Global flat doc-slot indices of this rank's own hard negatives (same order as d[:, first_neg:])."""
        B, G, dev = x.batch_size, x.group_size, x.q.device
        n_neg = G - first_neg
        base = (torch.arange(B, device=dev) + x.d_start).repeat_interleave(n_neg) * G
        return base + first_neg + torch.arange(n_neg, device=dev).repeat(B)

    def _cand_slots(self, x: LossInput, first_neg: int) -> torch.Tensor:
        """Global flat slot indices of the candidate list for the current inbatch_scope.

        slot index = example_idx * G + slot_in_group, which indexes doc_uids_all
        directly, so both scopes share the same masking code.
        """
        G, dev = x.group_size, x.q.device
        n_ex = x.d_all.size(0)
        if self.inbatch_scope == "positives_only":
            npos = first_neg
            idx = torch.arange(n_ex * npos, device=dev)
            slots = torch.div(idx, npos, rounding_mode="floor") * G + idx % npos
            if G > npos:
                slots = torch.cat([slots, self._own_neg_slots(x, npos)])
            return slots
        return torch.arange(n_ex * G, device=dev)

    def _build_candidates(self, x: LossInput, dim: Optional[int]):
        """Return (candidates [N,D], labels [B], neg_mask [B,N] or None)."""
        B, G = x.batch_size, x.group_size
        q = self._slice(x.q, dim)
        ident = self._identity(x)
        dev = q.device

        if not self.use_inbatch_negatives:
            d = self._slice(x.d, dim)  # [B,G,D]
            logits = torch.einsum("bd,bgd->bg", q, d) * self.scale
            labels = torch.zeros(B, dtype=torch.long, device=dev)
            anchor_ex = torch.arange(B, device=dev) + x.d_start
            slots = anchor_ex.unsqueeze(1) * G + torch.arange(G, device=dev)
            mask = self.fn_masker.own_slots_mask(ident, anchor_ex, slots)
            if mask is not None:
                mask[:, 0] = False  # never mask the gold candidate
                logits = logits.masked_fill(mask, _neg_inf(logits))
            return logits, labels, q, d.reshape(B * G, -1)

        if self.inbatch_scope == "positives_only":
            pos_all = self._slice(x.d_all[:, 0], dim)  # [B*W, D]
            cand = [pos_all]
            labels = torch.arange(B, device=dev) + x.offset
            if G > 1:
                own_neg = self._slice(x.d[:, 1:], dim)
                cand.append(own_neg.reshape(B * (G - 1), own_neg.size(-1)))
            candidates = torch.cat(cand, dim=0)
        else:
            sliced = self._slice(x.d_all, dim)
            candidates = sliced.reshape(-1, sliced.size(-1))
            labels = (torch.arange(B, device=dev) + x.offset) * G

        logits = q @ candidates.t() * self.scale
        anchor_ex = torch.arange(B, device=dev) + x.d_start
        slots = self._cand_slots(x, first_neg=1)
        mask = self.fn_masker.doc_side_mask(ident, anchor_ex, slots)
        if mask is not None:
            gold = torch.zeros_like(mask)
            gold.scatter_(1, labels.view(-1, 1), True)
            mask = mask & ~gold
            logits = logits.masked_fill(mask, _neg_inf(logits))
            self._last_mask_ratio = float(mask.sum().item()) / max(mask.numel() - B, 1)
        return logits, labels, q, candidates

    # ------------------------------------------------------- multi-view / multi-positive
    @staticmethod
    def _ones(n: int, ref: torch.Tensor) -> torch.Tensor:
        return torch.ones(n, dtype=torch.bool, device=ref.device)

    def _valid_or_ones(
        self, mask: Optional[torch.Tensor], rows: int, cols: int, ref: torch.Tensor
    ) -> torch.Tensor:
        if mask is None:
            return torch.ones(rows, cols, dtype=torch.bool, device=ref.device)
        return mask.to(ref.device).bool().view(rows, cols)

    def _mv_reduce(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """InfoNCE with a block-diagonal target.

        logits [A, N] (already scaled by temperature), target [A, N] bool (which
        candidates are positives), weight [A] float (0 = padded anchor, excluded).
        """
        logits = logits.float()
        neg_inf = _neg_inf(logits)

        n_pos = target.sum(dim=1)
        n_pos_f = n_pos.clamp(min=1).float()

        if self.multi_positive_reduction == "joint":
            # -log( Σ_{p∈P} e^{s_p} / Σ_j e^{s_j} ): only the total positive mass must be large
            num = torch.logsumexp(logits.masked_fill(~target, neg_inf), dim=1)
            den = torch.logsumexp(logits, dim=1)
            per_anchor = -(num - den)
        elif self.mask_sibling_positives:
            # Each positive is scored separately with the other positives of the same
            # sample removed from its denominator (independent normalization per positive,
            # no positive-view balance term). Lower bound is 0 (not log|P|).
            neg_lse = torch.logsumexp(logits.masked_fill(target, neg_inf), dim=1, keepdim=True)
            per_pos = -(logits - torch.logaddexp(logits, neg_lse))
            per_anchor = (per_pos * target).sum(dim=1) / n_pos.clamp(min=1)
        else:
            # Multi-Positive View InfoNCE (paper Eq. 4): all positives stay in the shared denominator
            log_prob = F.log_softmax(logits, dim=1)
            per_anchor = -(log_prob * target).sum(dim=1) / n_pos.clamp(min=1)

        if self.label_smoothing > 0:
            uniform = -F.log_softmax(logits, dim=1).mean(dim=1)
            per_anchor = (1 - self.label_smoothing) * per_anchor + self.label_smoothing * uniform

        base_w = weight.float() * (n_pos > 0).float()  # row validity (padded views = 0)
        denom = base_w.sum().clamp(min=1.0)
        loss = (per_anchor * base_w).sum() / denom

        with torch.no_grad():
            pred = logits.argmax(dim=1, keepdim=True)
            hit = target.gather(1, pred).squeeze(1).float()
            acc = (hit * base_w).sum() / denom
            cos = logits / self.scale
            pos_sim = (cos * target).sum() / n_pos.sum().clamp(min=1)
        metrics = {
            "acc": acc.item(),
            "pos_sim": pos_sim.item(),
            "n_cand": float(logits.size(1)),
            "n_pos": float(n_pos.float().mean().item()),
        }
        return loss, metrics

    # All false-negative masking rules live in losses/false_negative.py (FalseNegativeMasker);
    # single-view and multi-view, q->d and d->q paths share the same rules and id table.

    def _mv_query_to_doc(self, x: LossInput, dim: Optional[int]):
        """query -> doc direction: candidates are all (cross-GPU) docs; targets are all positive views of the sample."""
        B, nq, npos, G = x.batch_size, x.num_query_views, x.num_positives, x.group_size
        q = self._slice(x.q, dim)  # [B*nq, D]
        n_query = q.size(0)
        dev = q.device
        q_example = torch.arange(n_query, device=dev) // nq + x.d_start
        q_valid = self._valid_or_ones(x.q_valid, B, nq, q).reshape(-1)

        if not self.use_inbatch_negatives:
            d = self._slice(x.d, dim)  # [B, G, D]
            logits = torch.einsum("bvd,bgd->bvg", q.view(B, nq, -1), d).reshape(n_query, G)
            logits = logits * self.scale
            d_valid = self._valid_or_ones(x.d_valid, B, npos, q)  # [B, npos]
            slot_valid = torch.cat([d_valid, self._ones(B * (G - npos), q).view(B, G - npos)], dim=1)
            slot_valid = slot_valid.repeat_interleave(nq, dim=0)  # [B*nq, G]
            is_pos = (torch.arange(G, device=dev) < npos).unsqueeze(0)
            target = is_pos & slot_valid
            logits = logits.masked_fill(~slot_valid, _neg_inf(logits))
            # Even with only own candidates, guard against "my hard negative is actually my positive"
            own_slots = q_example.unsqueeze(1) * G + torch.arange(G, device=dev)
            own_mask = self.fn_masker.own_slots_mask(self._identity(x), q_example, own_slots)
            if own_mask is not None:
                logits = logits.masked_fill(own_mask & ~target, _neg_inf(logits))
            return logits, target, q_valid.float()

        d_all = self._slice(x.d_all, dim)  # [Ba, G, D]
        n_ex_all = d_all.size(0)
        d_valid_all = self._valid_or_ones(x.d_valid_all, n_ex_all, npos, q)

        if self.inbatch_scope == "positives_only":
            cands = [d_all[:, :npos].reshape(n_ex_all * npos, -1)]
            cand_example = torch.arange(n_ex_all * npos, device=dev) // npos
            cand_valid = d_valid_all.reshape(-1)
            cand_is_pos = self._ones(n_ex_all * npos, q)
            if G > npos:
                own_neg = x.d[:, npos:]
                own_neg = self._slice(own_neg, dim).reshape(B * (G - npos), -1)
                cands.append(own_neg)
                cand_example = torch.cat(
                    [cand_example, torch.full((own_neg.size(0),), -1, dtype=torch.long, device=dev)]
                )
                cand_valid = torch.cat([cand_valid, self._ones(own_neg.size(0), q)])
                cand_is_pos = torch.cat(
                    [cand_is_pos, torch.zeros(own_neg.size(0), dtype=torch.bool, device=dev)]
                )
            candidates = torch.cat(cands, dim=0)
        else:  # all: candidate order = d_all.reshape(-1), aligned with doc_uids_all
            candidates = d_all.reshape(n_ex_all * G, -1)
            idx = torch.arange(n_ex_all * G, device=dev)
            cand_example = idx // G
            cand_is_pos = (idx % G) < npos
            cand_valid = torch.cat(
                [d_valid_all, self._ones(n_ex_all * (G - npos), q).view(n_ex_all, G - npos)], dim=1
            ).reshape(-1)

        logits = q @ candidates.t() * self.scale
        target = (
            (cand_example.unsqueeze(0) == q_example.unsqueeze(1))
            & cand_is_pos.unsqueeze(0)
            & cand_valid.unsqueeze(0)
        )
        logits = logits.masked_fill(~cand_valid.unsqueeze(0), _neg_inf(logits))
        # False-negative masking: sibling groups / same cluster / doc_id in my positive_ids
        fn_mask = self.fn_masker.doc_side_mask(
            self._identity(x), q_example, self._cand_slots(x, first_neg=npos)
        )
        if fn_mask is not None:
            fn_mask = fn_mask & ~target
            logits = logits.masked_fill(fn_mask, _neg_inf(logits))
            self._last_mask_ratio = self.fn_masker.masked_ratio(fn_mask, target)
        return logits, target, q_valid.float()

    def _mv_doc_to_query(self, x: LossInput, dim: Optional[int]):
        """doc -> query direction (symmetric term): candidates are all (cross-GPU) query views."""
        B, nq, npos = x.batch_size, x.num_query_views, x.num_positives
        dev = x.q.device
        anchors = self._slice(x.d[:, :npos], dim).reshape(B * npos, -1)
        cands = self._slice(x.q_all, dim)  # [Ba*nq, D]
        n_cand = cands.size(0)

        anchor_example = torch.arange(B * npos, device=dev) // npos + x.d_start
        cand_example = torch.arange(n_cand, device=dev) // nq
        cand_valid = self._valid_or_ones(x.q_valid_all, n_cand // nq, nq, x.q).reshape(-1)

        logits = anchors @ cands.t() * self.scale
        target = (cand_example.unsqueeze(0) == anchor_example.unsqueeze(1)) & cand_valid.unsqueeze(0)
        logits = logits.masked_fill(~cand_valid.unsqueeze(0), _neg_inf(logits))
        # Symmetric false negatives: candidate query has my query_id / same cluster / contains this doc as a positive
        anchor_slot = (
            (torch.arange(B, device=dev) + x.d_start).repeat_interleave(npos) * x.group_size
            + torch.arange(npos, device=dev).repeat(B)
        )
        sib_mask = self.fn_masker.query_side_mask(
            self._identity(x), anchor_slot, cand_example
        )
        if sib_mask is not None:
            logits = logits.masked_fill(sib_mask & ~target, _neg_inf(logits))
        anchor_valid = self._valid_or_ones(x.d_valid, B, npos, x.q).reshape(-1).float()
        return logits, target, anchor_valid

    def _one_dim_multiview(self, x: LossInput, dim: Optional[int]):
        nq, npos = x.num_query_views, x.num_positives

        logits, target, weight = self._mv_query_to_doc(x, dim)
        loss, metrics = self._mv_reduce(logits, target, weight)
        # Log k so 2x2 / 3x3 / 4x4 runs are easy to compare
        metrics["n_query_views"] = float(nq)
        metrics["n_pos_views"] = float(npos)

        if self.symmetric:
            logits_t, target_t, weight_t = self._mv_doc_to_query(x, dim)
            loss_t, m_t = self._mv_reduce(logits_t, target_t, weight_t)
            loss = (1 - self.symmetric_weight) * loss + self.symmetric_weight * loss_t
            metrics["acc_d2q"] = m_t["acc"]
        return loss, metrics

    # ---------------------------------------------------------------- main
    def _one_dim(self, x: LossInput, dim: Optional[int]):
        if x.is_multiview:
            return self._one_dim_multiview(x, dim)

        logits, labels, q, candidates = self._build_candidates(x, dim)
        loss = F.cross_entropy(logits.float(), labels, label_smoothing=self.label_smoothing)

        if self.symmetric:
            pos_local = self._slice(x.d[:, 0], dim)
            q_all = self._slice(x.q_all, dim)
            logits_t = pos_local @ q_all.t() * self.scale
            labels_t = torch.arange(x.batch_size, device=q.device) + x.offset
            dev = q.device
            anchor_slot = (torch.arange(x.batch_size, device=dev) + x.d_start) * x.group_size
            cand_example = torch.arange(q_all.size(0), device=dev)  # single view: one query per row
            m_t = self.fn_masker.query_side_mask(self._identity(x), anchor_slot, cand_example)
            if m_t is not None:
                gold_t = torch.zeros_like(m_t)
                gold_t.scatter_(1, labels_t.view(-1, 1), True)
                logits_t = logits_t.masked_fill(m_t & ~gold_t, _neg_inf(logits_t))
            loss_t = F.cross_entropy(logits_t.float(), labels_t, label_smoothing=self.label_smoothing)
            loss = (1 - self.symmetric_weight) * loss + self.symmetric_weight * loss_t

        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == labels).float().mean()
            pos_sim = logits.gather(1, labels.view(-1, 1)).mean() / self.scale
        return loss, {"acc": acc.item(), "pos_sim": pos_sim.item(), "n_cand": float(logits.size(1))}

    def forward(self, x: LossInput) -> LossOutput:  # type: ignore[override]
        metrics: Dict[str, float] = {"temperature": float(1.0 / self.scale.item())}
        if not self.matryoshka_dims:
            loss, m = self._one_dim(x, None)
            metrics.update(m)
        else:
            loss = x.q.new_zeros(())
            for w, dim in zip(self.matryoshka_weights, self.matryoshka_dims):
                sub_loss, m = self._one_dim(x, dim)
                loss = loss + w * sub_loss
                metrics[f"loss@{dim}"] = sub_loss.item()
                metrics[f"acc@{dim}"] = m["acc"]
            metrics["acc"] = metrics[f"acc@{max(self.matryoshka_dims)}"]

        metrics["loss_retrieval"] = float(loss.item())
        if self.fn_masker.enabled:
            # Fraction of masked false negatives. Constantly 0 means ids are not configured
            # (or there are no duplicates); persistently >10% suggests heavy row duplication.
            metrics["fn_masked_ratio"] = float(self._last_mask_ratio)

        loss = self._apply_balance(x, loss, metrics)
        metrics["loss_total"] = float(loss.item())

        return LossOutput(loss=loss, metrics=metrics)

    # ------------------------------------------------------- modality_balance
    def _apply_balance(
        self, x: LossInput, loss: torch.Tensor, metrics: Dict[str, float]
    ) -> torch.Tensor:
        """Compute the modality-balance term and add it to the total loss according to `weight`.

        weight > 0 : computed every step, gradients flow, metrics are logged.
        weight == 0: computed under no_grad every `log_every` steps, **metrics only**.
                     Training is bit-identical to not having the term at all, but the
                     curves are still available (used as the ablation control).
        """
        if self.balance is None:
            return loss

        self._balance_step += 1
        w = self.balance_weight

        if w > 0:
            out = self.balance(x)
            loss = loss + w * out.loss
        else:
            if self._balance_step % self.balance_log_every:
                return loss
            with torch.no_grad():
                out = self.balance(x)

        if out.metrics:  # empty for single-view (1x1 block, no dispersion to reduce)
            metrics.update(out.metrics)
            metrics["balance/weight"] = w
            metrics["balance/weighted"] = w * float(out.loss.item())
        return loss


# Alias: `type: mp_infonce` is identical to `type: infonce`
LOSSES.register("mp_infonce")(InfoNCELoss)
