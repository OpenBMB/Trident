from __future__ import annotations

from typing import Any, Dict

from ..config import to_plain
from ..registry import LOSSES
from ..utils.misc import get_logger
from .base import BaseLoss, LossInput, LossOutput
from .infonce import InfoNCELoss  # noqa: F401
from .modality_balance import ModalityBalanceLoss  # noqa: F401
from .spec import canonicalize, describe, identity_requirements  # noqa: F401

logger = get_logger(__name__)

__all__ = [
    "BaseLoss", "LossInput", "LossOutput", "InfoNCELoss", "ModalityBalanceLoss",
    "build_loss", "canonicalize", "describe", "identity_requirements", "LOSSES",
]


def build_loss(cfg: Dict[str, Any]) -> BaseLoss:
    cfg = dict(to_plain(cfg))
    loss_type = cfg.pop("type", None)
    if not loss_type:
        raise ValueError("loss config is missing the `type` field")
    logger.info("Building loss: type=%s (available: %s)", loss_type, LOSSES.keys())
    return LOSSES.build(loss_type, **cfg)
