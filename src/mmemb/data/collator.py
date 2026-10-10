"""Collator: turns a list of Examples into a model-ready batch.

Key design: **the collator knows nothing about the model**. Tokenization and
image preprocessing are delegated to the encoder's `build_inputs(records, role)`,
so switching models (even to text-only or non-shared dual towers) requires no
collator changes.

Output structure (Nq = num_query_views, Np = num_positives, R = num_negatives, G = Np + R):
    {
      "query_inputs": {...},          # B*Nq items, ordered [ex0_v0, ex0_v1, ex1_v0, ...]
      "doc_inputs":   {...},          # B*G items, grouped as [pos_0..pos_{Np-1}, neg...]
      "group_size":   G,
      "num_query_views": Nq,
      "num_positives":   Np,
      "query_valid":  BoolTensor[B, Nq],   # multi-view only; False = padded (duplicated) view
      "doc_valid":    BoolTensor[B, Np],   # multi-view only
      "query_index" / "doc_index": LongTensor  # only when dedupe=True (indices to undo dedup)
      "meta": {"doc_uids": [...], "query_uids": [...], "tasks": [...],
               "example_uids": [...],   # length B; sibling view groups of one row share a uid
               "identity": {            # for false-negative masking (see losses/false_negative.py)
                   "example_uids": [B], "group_uids": [B], "query_uids": [B],
                   "positive_ids": [B][*], "doc_uids": [B*G]},
               "teacher_scores": Tensor|None}
    }

Nq / Np correspond to `k` in the config (`data.multiview_k`) and can be 2, 3, 4, ...
The collator makes no assumption about k; it only pads every sample into a
regular [B, Nq] / [B, G] layout.

With Nq = Np = 1 the output is identical to the single-view layout (no
valid / index fields are added).
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch

from .schema import Example, Record


def _record_key(rec: Record) -> Tuple[str, Tuple[str, ...], str]:
    """Whether two Records would be encoded into the same embedding."""
    return (rec.text or "", tuple(rec.images), rec.instruction or "")


def _dedupe(records: List[Record]) -> Tuple[List[Record], List[int]]:
    uniq: List[Record] = []
    index: List[int] = []
    seen: Dict[Tuple[str, Tuple[str, ...], str], int] = {}
    for rec in records:
        key = _record_key(rec)
        pos = seen.get(key)
        if pos is None:
            pos = len(uniq)
            seen[key] = pos
            uniq.append(rec)
        index.append(pos)
    return uniq, index


class ContrastiveCollator:
    def __init__(
        self,
        encoder,
        num_negatives: int = 1,
        collect_uids: bool = False,
        num_query_views: int = 1,
        num_positives: int = 1,
        dedupe: bool = False,
        collect_example_uids: bool = False,
        collect_identity: bool = True,
        multiview_negatives: bool = False,
    ) -> None:
        self.encoder = encoder
        self.num_negatives = int(num_negatives)
        self.collect_uids = collect_uids
        self.num_query_views = max(1, int(num_query_views))
        self.num_positives = max(1, int(num_positives))
        self.dedupe = bool(dedupe)
        self.multiview_negatives = bool(multiview_negatives)
        # When one row is split into several view groups, sibling groups can land in
        # the same batch; the loss must know they share a source, or they become false negatives.
        self.collect_example_uids = bool(collect_example_uids)
        # Identity info for false-negative masking (example / group / query / doc ids + positive_ids).
        # Only a few lists of Python strings, so it is enabled by default.
        self.collect_identity = bool(collect_identity)

    @property
    def is_multiview(self) -> bool:
        return self.num_query_views > 1 or self.num_positives > 1

    # ------------------------------------------------------------------
    @staticmethod
    def _pad_views(views: Sequence[Record], n: int, what: str) -> Tuple[List[Record], List[bool]]:
        """Pad a view list to a fixed length n: real views are valid=True, duplicates valid=False.

        Views are duplicated rather than left empty because cross-GPU all_gather requires
        identical tensor shapes on every rank, so the batch must be a regular [B, Nq] / [B, G];
        padded positions are excluded with the mask.
        """
        views = list(views)
        if not views:
            raise ValueError(f"Sample has no {what} views")
        out, valid = [], []
        for i in range(n):
            if i < len(views):
                out.append(views[i])
                valid.append(True)
            else:
                out.append(views[i % len(views)])
                valid.append(False)
        return out, valid

    def __call__(self, examples: Sequence[Example]) -> Dict[str, Any]:
        nq, npos, nneg = self.num_query_views, self.num_positives, self.num_negatives
        # With multi-view negatives, each negative group is padded to npos views
        if self.multiview_negatives:
            group_size = npos * (1 + nneg)
        else:
            group_size = npos + nneg
        bsz = len(examples)

        q_records: List[Record] = []
        d_records: List[Record] = []
        q_valid: List[List[bool]] = []
        d_valid: List[List[bool]] = []   # positive slots only -> [B, Np]
        n_valid: List[List[bool]] = []   # negative slots only -> [B*R, Np] (multi-view negatives)

        for ex in examples:
            qv, qm = self._pad_views(ex.query_views, nq, "query")
            q_records.extend(qv)
            q_valid.append(qm)

            pv, pm = self._pad_views(ex.positive_views, npos, "positive")
            d_records.extend(pv)
            d_valid.append(pm)

            negs = list(ex.negatives)[:nneg]
            if len(negs) != nneg:
                raise ValueError(
                    f"Inconsistent number of negatives (expected {nneg}, got {len(negs)}). "
                    "Please check data.negative_fill."
                )
            # negs is List[List[Record]] (each negative is a view group)
            for neg_group in negs:
                if self.multiview_negatives:
                    # Each negative document gets Np views, same shape as the positive document
                    nv, nm = self._pad_views(neg_group, npos, "negative")
                    d_records.extend(nv)
                    n_valid.append(nm)
                else:
                    # Single-view negatives: each negative occupies exactly one column, so the
                    # number of docs per example is Np + R, matching group_size.
                    d_records.append(neg_group[0])

        # Dedup: view padding creates duplicate Records; dedup saves their forward passes
        if self.dedupe:
            q_enc, q_index = _dedupe(q_records)
            d_enc, d_index = _dedupe(d_records)
        else:
            q_enc, q_index = q_records, None
            d_enc, d_index = d_records, None

        batch: Dict[str, Any] = {
            "query_inputs": self.encoder.build_inputs(q_enc, role="query"),
            "doc_inputs": self.encoder.build_inputs(d_enc, role="doc"),
            "group_size": group_size,
        }
        if q_index is not None:
            batch["query_index"] = torch.tensor(q_index, dtype=torch.long)
            batch["doc_index"] = torch.tensor(d_index, dtype=torch.long)

        if self.is_multiview:
            batch["num_query_views"] = nq
            batch["num_positives"] = npos
            batch["query_valid"] = torch.tensor(q_valid, dtype=torch.bool).view(bsz, nq)
            batch["doc_valid"] = torch.tensor(d_valid, dtype=torch.bool).view(bsz, npos)
            if n_valid:
                # [B, R, Np]: which hard-negative views are real (padded views are excluded)
                batch["neg_valid"] = torch.tensor(n_valid, dtype=torch.bool).view(
                    bsz, nneg, npos
                )

        meta: Dict[str, Any] = {"tasks": [ex.task for ex in examples]}
        if self.collect_example_uids:
            meta["example_uids"] = [
                (ex.example_uid if ex.example_uid is not None else f"_ex{i}")
                for i, ex in enumerate(examples)
            ]
        if self.collect_uids:
            meta["query_uids"] = [r.uid for r in q_records]
            meta["doc_uids"] = [r.uid for r in d_records]
        if self.collect_identity:
            # Length contract (relied upon by losses/false_negative.py):
            #   example_uids / group_uids / query_uids / positive_ids -> B
            #   doc_uids -> B * G, in the same order as the flattened doc_inputs
            meta["identity"] = {
                "example_uids": [
                    (ex.example_uid if ex.example_uid is not None else f"_ex{i}")
                    for i, ex in enumerate(examples)
                ],
                "group_uids": [
                    (ex.group_id or ex.example_uid or f"_ex{i}")
                    for i, ex in enumerate(examples)
                ],
                "query_uids": [ex.query_uid for ex in examples],
                "positive_ids": [list(ex.positive_ids) for ex in examples],
                "doc_uids": [r.uid for r in d_records],
            }
        scores = [ex.extra.get("teacher_scores") for ex in examples]
        if all(s is not None and len(s) >= group_size for s in scores):
            meta["teacher_scores"] = torch.tensor(
                [s[:group_size] for s in scores], dtype=torch.float
            )
        batch["meta"] = meta
        return batch
