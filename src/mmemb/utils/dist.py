"""Distributed helpers, centred on a **differentiable all_gather** used for
cross-GPU negatives in contrastive learning.

Gradient derivation (why backward does all_reduce and then takes the local slice):
    The overall objective is L = mean_r L_r, where L_r is the loss on rank r
    computed from local queries x global documents.
    dL_r/dθ = Σ_j (dL_r/de_j)(de_j/dθ), and only rank j can compute de_j/dθ.
    In backward we all_reduce(SUM) the gradient so rank j receives Σ_r dL_r/de_j,
    and local backprop yields Σ_r dL_r/dθ; DDP then averages (/W), giving exactly
    d(mean_r L_r)/dθ. Hence **no** extra world_size scaling is needed.
"""
from __future__ import annotations

from typing import Any, List

import torch
import torch.distributed as dist


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_dist():
        dist.barrier()


def all_gather_object(obj: Any) -> List[Any]:
    if not is_dist():
        return [obj]
    buf: List[Any] = [None] * get_world_size()
    dist.all_gather_object(buf, obj)
    return buf


class _AllGatherWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):  # type: ignore[override]
        ctx.world_size = get_world_size()
        ctx.rank = get_rank()
        buf = [torch.zeros_like(x) for _ in range(ctx.world_size)]
        dist.all_gather(buf, x.contiguous())
        buf[ctx.rank] = x  # keep the autograd graph for the local shard
        return torch.cat(buf, dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        grad = grad_output.contiguous()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        return grad.chunk(ctx.world_size, dim=0)[ctx.rank]


def gather_with_grad(x: torch.Tensor) -> torch.Tensor:
    """Gather tensors across ranks while keeping gradients. No-op on a single GPU."""
    if not is_dist() or get_world_size() == 1:
        return x
    return _AllGatherWithGrad.apply(x)


def gather_detached(x: torch.Tensor) -> torch.Tensor:
    """Gather across ranks; shards from other ranks are detached (cheaper, deadlock-free fallback)."""
    if not is_dist() or get_world_size() == 1:
        return x
    buf = [torch.zeros_like(x) for _ in range(get_world_size())]
    dist.all_gather(buf, x.detach().contiguous())
    buf[get_rank()] = x
    return torch.cat(buf, dim=0)


def gather_mask(x: torch.Tensor) -> torch.Tensor:
    """Gather a boolean mask across ranks (no gradients).

    Some backends handle all_gather on bool tensors unreliably, so the mask is
    converted to float first. Shape [B, K] -> [B*W, K].
    """
    if not is_dist() or get_world_size() == 1:
        return x
    f = x.float().contiguous()
    buf = [torch.zeros_like(f) for _ in range(get_world_size())]
    dist.all_gather(buf, f)
    return torch.cat(buf, dim=0) > 0.5
