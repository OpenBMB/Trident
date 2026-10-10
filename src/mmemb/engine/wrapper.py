"""Wrap the encoder and the loss into a single nn.Module.

Why? DDP expects "one forward per backward". Calling model(query) and
model(doc) separately in the trainer would break gradient synchronization, so
query/doc encoding and the loss computation all happen in **one forward**.

Side benefit: learnable loss parameters (e.g. a learnable temperature) are
automatically synchronized by DDP and handled by the optimizer.

Multi-view (several query views / positive views per sample):
    query_inputs holds B*Nq items and doc_inputs holds B*(Np+R) items. Shapes
    remain regular, so cross-GPU all_gather works unchanged. Nq / Np are the `k`
    of the config; 2, 3, 4, ... make no difference here.

`example_uids`: when a row has more views than k it is split into several view
groups, and sibling groups can appear in the same batch (or on different GPUs).
Their uids are also all-gathered so the loss can remove sibling groups from the
in-batch negatives (otherwise they would be false negatives).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..losses.base import BaseLoss, LossInput
from ..models.base import BaseEmbedder
from ..utils.dist import (
    all_gather_object,
    gather_detached,
    gather_mask,
    gather_with_grad,
    get_rank,
    get_world_size,
)
from ..utils.misc import get_logger

logger = get_logger(__name__)


def _as_int(v: Any, default: int = 1) -> int:
    """DataParallel turns scalars into tensors; convert them back to Python ints."""
    if v is None:
        return default
    if torch.is_tensor(v):
        return int(v.reshape(-1)[0].item())
    return int(v)


class ContrastiveWrapper(nn.Module):
    def __init__(
        self,
        encoder: BaseEmbedder,
        loss_fn: BaseLoss,
        cross_device: str = "grad",  # grad | detach | off
        collect_uids: bool = False,
        collect_example_uids: bool = True,
        collect_identity: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.loss_fn = loss_fn
        self.cross_device = cross_device
        self.collect_uids = collect_uids
        self.collect_example_uids = collect_example_uids
        self.collect_identity = collect_identity
        # Extension point: for non-shared dual towers, build a separate doc encoder here
        # and route doc_inputs through self.doc_encoder in forward.
        self.doc_encoder: Optional[BaseEmbedder] = None
        # Cache the metrics of the last forward (read by the Trainer; avoids DataParallel gather issues)
        self._last_metrics: Dict[str, float] = {}

    # ------------------------------------------------------------------
    def _encode(self, inputs: Dict[str, Any], role: str) -> torch.Tensor:
        enc = self.doc_encoder if (role == "doc" and self.doc_encoder is not None) else self.encoder
        return enc.encode_features(**inputs)

    def _gather(self, t: torch.Tensor) -> torch.Tensor:
        if get_world_size() == 1 or self.cross_device == "off":
            return t
        if self.cross_device == "detach":
            return gather_detached(t)
        return gather_with_grad(t)

    def _gather_mask(self, t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if t is None:
            return None
        if get_world_size() == 1 or self.cross_device == "off":
            return t
        return gather_mask(t)

    def forward(
        self,
        query_inputs: Dict[str, Any],
        doc_inputs: Dict[str, Any],
        group_size: int = 2,
        meta: Optional[Dict[str, Any]] = None,
        num_query_views: int = 1,
        num_positives: int = 1,
        query_valid: Optional[torch.Tensor] = None,
        doc_valid: Optional[torch.Tensor] = None,
        neg_valid: Optional[torch.Tensor] = None,
        query_index: Optional[torch.Tensor] = None,
        doc_index: Optional[torch.Tensor] = None,
        **_unused,
    ) -> Dict[str, Any]:
        meta = dict(meta or {})
        group_size = _as_int(group_size, 2)
        n_qview = _as_int(num_query_views, 1)
        n_pos = _as_int(num_positives, 1)

        q = self._encode(query_inputs, "query")  # [B*Nq, D] ([U, D] when deduplicated)
        d_flat = self._encode(doc_inputs, "doc")  # [B*G, D] ([U', D] when deduplicated)

        # The collator deduplicated records: restore the full layout from unique embeddings
        if query_index is not None:
            q = q.index_select(0, query_index.reshape(-1).to(q.device))
        if doc_index is not None:
            d_flat = d_flat.index_select(0, doc_index.reshape(-1).to(d_flat.device))

        D = q.size(-1)
        if q.size(0) % n_qview != 0:
            raise RuntimeError(
                f"number of queries ({q.size(0)}) is not a multiple of num_query_views ({n_qview})"
            )
        bsz = q.size(0) // n_qview
        if d_flat.size(0) != bsz * group_size:
            raise RuntimeError(
                f"number of docs ({d_flat.size(0)}) != B*G ({bsz}*{group_size}); "
                "check collator / num_negatives / num_positives"
            )
        if n_pos > group_size:
            raise RuntimeError(f"num_positives ({n_pos}) must not exceed group_size ({group_size})")
        d = d_flat.view(bsz, group_size, D)

        q_all = self._gather(q)
        d_all = self._gather(d)
        world = get_world_size() if self.cross_device != "off" else 1
        rank = get_rank() if world > 1 else 0
        q_offset = rank * bsz * n_qview
        d_offset = rank * bsz

        q_valid = query_valid.bool() if query_valid is not None else None
        d_valid = doc_valid.bool() if doc_valid is not None else None
        if q_valid is not None:
            q_valid = q_valid.view(bsz, n_qview).to(q.device)
        if d_valid is not None:
            d_valid = d_valid.view(bsz, n_pos).to(q.device)
        # View validity of hard negatives: [B, R, Np], R = (G - Np) // Np
        n_valid = neg_valid.bool() if neg_valid is not None else None
        if n_valid is not None:
            n_hard = (group_size - n_pos) // max(n_pos, 1)
            if n_pos > 0 and (group_size - n_pos) % n_pos == 0 and n_hard > 0:
                n_valid = n_valid.reshape(bsz, n_hard, n_pos).to(q.device)
            else:  # shape mismatch: treat as unavailable; the loss assumes all views are valid (safe degradation)
                n_valid = None

        if self.collect_uids and meta.get("doc_uids") is not None:
            if world > 1:
                gathered = all_gather_object(meta["doc_uids"])
                meta["doc_uids_all"] = [u for part in gathered for u in part]
            else:
                meta["doc_uids_all"] = list(meta["doc_uids"])

        # One uid per sample (length B); after gathering, aligned with dim 0 of d_all
        if self.collect_example_uids and meta.get("example_uids") is not None:
            if world > 1:
                gathered = all_gather_object(meta["example_uids"])
                meta["example_uids_all"] = [u for part in gathered for u in part]
            else:
                meta["example_uids_all"] = list(meta["example_uids"])

        # Gather identity info across GPUs: candidates come from all GPUs, so masking rules must see all ids.
        # One all_gather_object of a few string lists is negligible compared with a forward pass.
        if self.collect_identity and isinstance(meta.get("identity"), dict):
            local = meta["identity"]
            if world > 1:
                parts = all_gather_object(local)
                meta["identity_all"] = {
                    key: [v for part in parts for v in part[key]] for key in local
                }
            else:
                meta["identity_all"] = {k: list(v) for k, v in local.items()}

        out = self.loss_fn(
            LossInput(
                q=q,
                d=d,
                q_all=q_all,
                d_all=d_all,
                offset=q_offset,
                world_size=world,
                meta=meta,
                num_query_views=n_qview,
                num_positives=n_pos,
                doc_offset=d_offset,
                q_valid=q_valid,
                d_valid=d_valid,
                q_valid_all=self._gather_mask(q_valid),
                d_valid_all=self._gather_mask(d_valid),
                neg_valid=n_valid,
                neg_valid_all=self._gather_mask(n_valid),
            )
        )
        # Only return tensors that DataParallel can gather; metrics are stored on the instance
        # (DataParallel would otherwise try to gather a dict of floats and fail).
        self._last_metrics = dict(out.metrics)
        self._last_metrics["global_batch"] = float(d_all.size(0))
        return {"loss": out.loss, "q": q, "d": d}  # only these three fields; no metrics

    # ------------------------------------------------------------------
    def gradient_checkpointing_enable(self, **kwargs):  # called by the HF Trainer
        backbone = getattr(self.encoder, "backbone", None)
        if backbone is not None and hasattr(backbone, "gradient_checkpointing_enable"):
            backbone.gradient_checkpointing_enable(**kwargs)

    def save_pretrained(self, save_dir: str) -> None:
        self.encoder.save_pretrained(save_dir)
        extra = {k: v for k, v in self.loss_fn.state_dict().items()}
        if extra:
            torch.save(extra, f"{save_dir}/loss_state.bin")
