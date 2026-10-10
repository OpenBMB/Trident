from __future__ import annotations

import inspect
import logging
import os
import random
import sys
from typing import Any, Callable, Dict

import numpy as np
import torch

_LOG_READY = False


def get_logger(name: str = "mmemb") -> logging.Logger:
    global _LOG_READY
    logger = logging.getLogger(name)
    if not _LOG_READY:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s", "%H:%M:%S")
        )
        root = logging.getLogger("mmemb")
        root.addHandler(handler)
        root.setLevel(os.environ.get("MMEMB_LOG_LEVEL", "INFO"))
        root.propagate = False
        _LOG_READY = True
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "auto": "auto",
}


def resolve_dtype(name: str | None):
    if name is None:
        return "auto"
    if isinstance(name, torch.dtype):
        return name
    key = str(name).lower()
    if key not in DTYPES:
        raise ValueError(f"Unknown dtype: {name}; choices: {sorted(DTYPES)}")
    return DTYPES[key]


def call_with_supported_kwargs(fn: Callable, /, **kwargs) -> Any:
    """Call `fn` with only the kwargs present in its signature (for cross-version compatibility)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(**kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**kwargs)
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**allowed)


def load_hf_pretrained(loader, path: str, dtype: Any = None, **kwargs):
    """Load a HF model with the dtype kwarg name expected by the installed transformers.

        transformers >= 4.56 : dtype=
        transformers <  4.56 : torch_dtype=

    Note: this must be decided **by version**, not by "try `dtype`, fall back to
    `torch_dtype` on error". `from_pretrained` accepts `**kwargs`, so older
    versions silently swallow `dtype=` as a config field and load the model in
    fp32, doubling memory usage without raising.
    """
    if dtype is None:
        return loader(path, **kwargs)

    from .env import dtype_kwarg_name  # lazy import to avoid a circular dependency

    primary = dtype_kwarg_name()
    fallback = "torch_dtype" if primary == "dtype" else "dtype"

    try:
        return loader(path, **{primary: dtype}, **kwargs)
    except TypeError as e:  # rare custom loaders with explicit parameter lists
        get_logger(__name__).warning("`%s=` is not supported (%s); retrying with `%s=`", primary, e, fallback)
        return loader(path, **{fallback: dtype}, **kwargs)


def count_params(module: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def human(n: int) -> str:
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.2f}{unit}" if unit else str(n)
        n /= 1000.0
    return f"{n:.2f}T"
