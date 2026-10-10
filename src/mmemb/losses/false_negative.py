"""False-negative masking.

**Why it is needed**: the InfoNCE denominator contains every doc in the batch
(including cross-GPU docs). Whenever another sample's positive is also relevant
to my query, that cell is pushed down as a negative: the model is asked to push
two correct answers apart, which is simply a wrong gradient. The larger the
(global) batch, the more likely such collisions become.

This module gathers all masking rules in one place; both directions share the same ids:

    rule A  mask_same_example  candidate belongs to a sibling view group split from the same row
    rule B  mask_same_group    candidate shares my group_id (exclusive cluster)
    rule C  mask_false_negatives
                              candidate doc_id ∈ my positive_ids
                              (the main rule: non-transitive, exact, also covers hard-negative slots)
    rule D  mask_by_query_id   symmetric direction only: candidate query has my query_id,
                              or that query's positive_ids contain this doc

Rules A/B only apply to **positive slots**: hard negatives chosen by sibling /
same-cluster samples are deliberate hard negatives and should not be masked.
Rule C applies to **all slots**: if someone else's hard negative happens to be
my positive, it is a false negative and must be masked.

Implementation: string ids are mapped to integers and looked up in a boolean
table of shape [n_examples, V] (V = number of distinct doc_ids in the batch),
fully vectorized.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch


def _intern(values: Sequence[Any], table: Dict[Any, int]) -> List[int]:
    """Map string ids to integers; None maps to -1 (unknown, never masked)."""
    out: List[int] = []
    for v in values:
        if v is None:
            out.append(-1)
            continue
        idx = table.get(v)
        if idx is None:
            idx = len(table)
            table[v] = idx
        out.append(idx)
    return out


@dataclass
class BatchIdentity:
    """Identity information of a (cross-GPU merged) batch.

    n_ex = global number of samples, G = group_size (Np positive slots + R hard-negative slots).
    """

    example_ids: torch.Tensor  # [n_ex]      same row -> same id
    group_ids: torch.Tensor  # [n_ex]      exclusive-cluster id
    query_ids: torch.Tensor  # [n_ex]      query-level id
    doc_ids_flat: torch.Tensor  # [n_ex*G]    doc_id of each doc slot (-1 = unknown)
    pos_table: torch.Tensor  # [n_ex, V]   pos_table[e, v] = doc v is a positive of sample e
    group_size: int
    num_positives: int

    @property
    def n_examples(self) -> int:
        return int(self.example_ids.numel())

    def slot_example(self, slot: torch.Tensor) -> torch.Tensor:
        return torch.div(slot, self.group_size, rounding_mode="floor")

    def slot_is_positive(self, slot: torch.Tensor) -> torch.Tensor:
        return (slot % self.group_size) < self.num_positives

    def slot_doc(self, slot: torch.Tensor) -> torch.Tensor:
        """Returns -1 when slot < 0 (unknown candidate, never masked)."""
        safe = slot.clamp(min=0)
        doc = self.doc_ids_flat[safe]
        return torch.where(slot >= 0, doc, torch.full_like(doc, -1))


class FalseNegativeMasker:
    """Build [anchors, candidates] masks according to rules A-D.

    Every switch can be turned off individually; with all switches off, `enabled=False`
    and the whole path is free (no table, no all_gather, no mask).
    """

    def __init__(
        self,
        mask_false_negatives: bool = True,
        mask_same_example: bool = True,
        mask_same_group: bool = True,
        mask_by_query_id: bool = True,
    ) -> None:
        self.mask_false_negatives = bool(mask_false_negatives)
        self.mask_same_example = bool(mask_same_example)
        self.mask_same_group = bool(mask_same_group)
        self.mask_by_query_id = bool(mask_by_query_id)

    @property
    def enabled(self) -> bool:
        return (
            self.mask_false_negatives
            or self.mask_same_example
            or self.mask_same_group
            or self.mask_by_query_id
        )

    # ------------------------------------------------------------------ table construction
    def build(
        self,
        meta: Dict[str, Any],
        n_examples: int,
        group_size: int,
        num_positives: int,
        device: torch.device,
    ) -> Optional[BatchIdentity]:
        """Build a BatchIdentity from meta["identity_all"] (gathered across GPUs by the wrapper).

        Returns None if fields are missing or lengths mismatch (safe degradation: mask nothing rather than mask wrongly).
        """
        if not self.enabled:
            return None
        ident = meta.get("identity_all") or meta.get("identity")
        if not isinstance(ident, dict):
            return None

        ex_uids = ident.get("example_uids")
        if not ex_uids or len(ex_uids) != n_examples:
            return None

        doc_uids = ident.get("doc_uids")
        if not doc_uids or len(doc_uids) != n_examples * group_size:
            return None

        group_uids = ident.get("group_uids") or ex_uids
        query_uids = ident.get("query_uids") or [None] * n_examples
        pos_ids: Sequence[Sequence[str]] = ident.get("positive_ids") or [
            [] for _ in range(n_examples)
        ]

        # doc_id and positive_ids share one integer space so they can be looked up against each other
        doc_table: Dict[Any, int] = {}
        doc_flat = _intern(doc_uids, doc_table)
        pos_rows = [_intern(list(p or []), doc_table) for p in pos_ids]
        vocab = max(len(doc_table), 1)

        pos_table = torch.zeros(n_examples, vocab, dtype=torch.bool, device=device)
        for e, row in enumerate(pos_rows):
            for v in row:
                if v >= 0:
                    pos_table[e, v] = True

        long = lambda xs: torch.tensor(xs, dtype=torch.long, device=device)  # noqa: E731
        return BatchIdentity(
            example_ids=long(_intern(ex_uids, {})),
            group_ids=long(_intern(group_uids, {})),
            query_ids=long(_intern(query_uids, {})),
            doc_ids_flat=long(doc_flat),
            pos_table=pos_table,
            group_size=int(group_size),
            num_positives=int(num_positives),
        )

    # -------------------------------------------------------- query -> doc
    def doc_side_mask(
        self,
        ident: Optional[BatchIdentity],
        anchor_example: torch.Tensor,  # [A] global sample index of each anchor row
        cand_slot: torch.Tensor,  # [N] flat doc-slot index of each candidate (-1 = unknown)
    ) -> Optional[torch.Tensor]:
        """Return an [A, N] bool matrix: True = this candidate is a false negative for the anchor.

        Callers must additionally apply `& ~target` so that true gold candidates are never masked.
        """
        if ident is None:
            return None
        known = cand_slot >= 0
        cand_ex = ident.slot_example(cand_slot.clamp(min=0))
        cand_pos = ident.slot_is_positive(cand_slot.clamp(min=0)) & known

        mask = torch.zeros(
            anchor_example.numel(), cand_slot.numel(), dtype=torch.bool, device=cand_slot.device
        )

        if self.mask_same_example:
            same = ident.example_ids[cand_ex].unsqueeze(0) == ident.example_ids[
                anchor_example
            ].unsqueeze(1)
            mask |= same & cand_pos.unsqueeze(0)

        if self.mask_same_group:
            same = ident.group_ids[cand_ex].unsqueeze(0) == ident.group_ids[
                anchor_example
            ].unsqueeze(1)
            mask |= same & cand_pos.unsqueeze(0)

        if self.mask_false_negatives:
            doc = ident.slot_doc(cand_slot)
            hit = ident.pos_table[anchor_example][:, doc.clamp(min=0)]
            mask |= hit & (doc >= 0).unsqueeze(0)

        return mask

    def own_slots_mask(
        self,
        ident: Optional[BatchIdentity],
        anchor_example: torch.Tensor,  # [B]
        cand_slot: torch.Tensor,  # [B, G] per-row candidates (use_inbatch_negatives=False)
    ) -> Optional[torch.Tensor]:
        """Per-row candidate version: only rule C applies (own hard negative == own positive is a data error)."""
        if ident is None or not self.mask_false_negatives:
            return None
        doc = ident.slot_doc(cand_slot)  # [B, G]
        rows = ident.pos_table[anchor_example]  # [B, V]
        hit = rows.gather(1, doc.clamp(min=0))
        return hit & (doc >= 0)

    # -------------------------------------------------------- doc -> query
    def query_side_mask(
        self,
        ident: Optional[BatchIdentity],
        anchor_slot: torch.Tensor,  # [A] flat index of the anchor (positive doc slot)
        cand_example: torch.Tensor,  # [N] global sample index of each candidate query view
    ) -> Optional[torch.Tensor]:
        """Symmetric direction: candidates are query views, anchors are docs. Returns [A, N]."""
        if ident is None:
            return None
        anchor_ex = ident.slot_example(anchor_slot.clamp(min=0))
        mask = torch.zeros(
            anchor_slot.numel(), cand_example.numel(), dtype=torch.bool, device=anchor_slot.device
        )

        if self.mask_same_example:
            mask |= ident.example_ids[cand_example].unsqueeze(0) == ident.example_ids[
                anchor_ex
            ].unsqueeze(1)

        if self.mask_same_group:
            mask |= ident.group_ids[cand_example].unsqueeze(0) == ident.group_ids[
                anchor_ex
            ].unsqueeze(1)

        if self.mask_by_query_id:
            qa = ident.query_ids[anchor_ex]
            qc = ident.query_ids[cand_example]
            same_q = (qc.unsqueeze(0) == qa.unsqueeze(1)) & (qa.unsqueeze(1) >= 0)
            mask |= same_q

        if self.mask_false_negatives:
            # The candidate query's positive set contains this doc -> it is also a positive pair for me
            doc = ident.slot_doc(anchor_slot)  # [A]
            hit = ident.pos_table[cand_example][:, doc.clamp(min=0)]  # [N, A]
            mask |= hit.t() & (doc >= 0).unsqueeze(1)

        return mask

    # ------------------------------------------------------------- diagnostics
    @staticmethod
    def masked_ratio(mask: Optional[torch.Tensor], target: torch.Tensor) -> float:
        """Fraction of negative cells that were masked (logged to TensorBoard to monitor data quality)."""
        if mask is None:
            return 0.0
        eff = mask & ~target
        denom = float((~target).sum().item()) or 1.0
        return float(eff.sum().item()) / denom
