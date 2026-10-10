"""Model factory. `build_encoder(cfg)` / `load_encoder(dir)` are the only public entry points."""
from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict

from ..config import to_plain
from ..registry import MODELS
from ..utils.env import PROFILES, MODEL_REQUIREMENTS, active_profile, check_model_env
from ..utils.misc import get_logger
from . import pooling  # noqa: F401  triggers POOLERS registration
from .base import BaseEmbedder

logger = get_logger(__name__)

# ---------------------------------------------------------------- soft registration
# A missing dependency of one backbone (e.g. timm/einops for jina-clip remote code)
# must **not** make other backbones unavailable, so every backbone is registered
# if importable and only logs a warning otherwise. Requirements are listed in
# MODEL_REQUIREMENTS in mmemb/utils/env.py.
_IMPORT_ERRORS: Dict[str, str] = {}


def _try_register(module: str, name: str, types: Any) -> None:
    try:
        __import__(f"{__name__}.{module}", fromlist=[name])
    except Exception as e:  # noqa: BLE001
        for t in ([types] if isinstance(types, str) else types):
            _IMPORT_ERRORS[t] = f"{type(e).__name__}: {e}"
        logger.warning("%s not registered (%s). %s", types, e, _env_hint(
            types if isinstance(types, str) else types[0]))


def _env_hint(model_type: str) -> str:
    req = MODEL_REQUIREMENTS.get(model_type)
    if req is None:
        return ""
    return f"This backbone requires: {PROFILES.get(req.profile, req.profile)}"


_try_register("qwen3_vl_embedding", "Qwen3VLEmbeddingEncoder", "qwen3_vl_embedding")
_try_register("jina_clip", "JinaCLIPEncoder", ["jina_clip", "jina_clip_v2", "jina_clip_v1"])

logger.info(
    "Environment profile=%s, available encoders: %s", active_profile(), MODELS.keys()
)

__all__ = ["BaseEmbedder", "build_encoder", "load_encoder", "MODELS"]


def build_encoder(cfg: Dict[str, Any]) -> BaseEmbedder:
    cfg = to_plain(cfg)
    model_type = cfg.get("type")
    if not model_type:
        raise ValueError("model config is missing the `type` field")
    if model_type not in MODELS:
        ok, reasons = check_model_env(model_type)
        detail = "; ".join(reasons) or _IMPORT_ERRORS.get(model_type, "")
        raise KeyError(
            f"model.type='{model_type}' is not available in the current environment.\n"
            f"  reason: {detail or 'not registered (typo?)'}\n"
            f"  {_env_hint(model_type)}\n"
            f"  available: {MODELS.keys()}"
        )
    logger.info("Building encoder: type=%s (available: %s)", model_type, MODELS.keys())
    encoder: BaseEmbedder = MODELS.build(model_type, cfg)
    encoder.apply_freeze(cfg.get("freeze_patterns") or [], cfg.get("unfreeze_patterns") or [])
    return encoder


def load_encoder(save_dir: str, **override: Any) -> BaseEmbedder:
    """Load an encoder saved by save_pretrained (full weights or LoRA adapter are detected automatically)."""
    cfg_path = os.path.join(save_dir, BaseEmbedder.CONFIG_NAME)
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"{cfg_path} does not exist; is this a directory saved by mmemb?")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg = {**cfg, **override}

    is_adapter = os.path.isfile(os.path.join(save_dir, "adapter_config.json"))
    if is_adapter:
        cfg = copy.deepcopy(cfg)
        cfg["lora"] = {**(cfg.get("lora") or {}), "enable": False}
        cfg["processor_name_or_path"] = save_dir
        encoder = build_encoder(cfg)
        from peft import PeftModel

        encoder.backbone = PeftModel.from_pretrained(encoder.backbone, save_dir)
        logger.info("LoRA adapter loaded: %s", save_dir)
    else:
        cfg = copy.deepcopy(cfg)
        cfg["pretrained_model_name_or_path"] = save_dir
        cfg["processor_name_or_path"] = save_dir
        encoder = build_encoder(cfg)

    encoder.load_extra(save_dir)
    return encoder