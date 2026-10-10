"""Runtime environment checks, so both backbones run in a **single
transformers 4.57 environment**.

Background
----------
The released code supports two training backbones:

    backbone            transformers      notes
    ------------------  ----------------  ----------------------------------------
    qwen3_vl_embedding  >= 4.57           needs transformers.models.qwen3_vl
    jina_clip (v2/v1)   >= 4.49, < 5      Hub remote code; also needs timm / einops

Their intersection is transformers 4.57.x, which is the single supported
environment (see `requirements.txt`).

This module does three things:
  1. version parsing that does not depend on `packaging`;
  2. a `MODEL_REQUIREMENTS` table describing what each `model.type` needs;
  3. `check_model_env()`, which returns actionable error messages instead of a
     raw ImportError traceback.

Convention: this file **only imports the standard library**, so it can be
imported even when torch / transformers are missing or broken.
"""
from __future__ import annotations

import importlib
import importlib.util
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "parse_version",
    "package_version",
    "transformers_version",
    "torch_version",
    "has_module",
    "dtype_kwarg_name",
    "MODEL_REQUIREMENTS",
    "check_model_env",
    "require_model_env",
    "describe_env",
    "active_profile",
]


# ------------------------------------------------------------------ version helpers
def parse_version(text: Optional[str]) -> Tuple[int, ...]:
    """"4.57.6.dev0" -> (4, 57, 6). Returns () ("unknown") if it cannot be parsed."""
    if not text:
        return ()
    m = re.match(r"\s*v?(\d+(?:\.\d+)*)", str(text))
    if not m:
        return ()
    return tuple(int(x) for x in m.group(1).split("."))


def _pad(v: Sequence[int], n: int) -> Tuple[int, ...]:
    return tuple(list(v) + [0] * (n - len(v)))[:n]


def version_ge(v: Sequence[int], target: Sequence[int]) -> bool:
    """v >= target. An empty (unknown) version always returns True."""
    if not v:
        return True
    n = max(len(v), len(target))
    return _pad(v, n) >= _pad(target, n)


def version_lt(v: Sequence[int], target: Sequence[int]) -> bool:
    if not v:
        return True
    n = max(len(v), len(target))
    return _pad(v, n) < _pad(target, n)


def package_version(name: str) -> Optional[str]:
    """Return the installed package version, or None if it is not installed."""
    try:
        mod = importlib.import_module(name)
    except Exception:  # noqa: BLE001
        return None
    return getattr(mod, "__version__", None) or "unknown"


def transformers_version() -> Tuple[int, ...]:
    return parse_version(package_version("transformers"))


def torch_version() -> Tuple[int, ...]:
    return parse_version(package_version("torch"))


def has_module(dotted: str) -> bool:
    """Whether a module exists (without importing it, to avoid heavy side effects)."""
    try:
        return importlib.util.find_spec(dotted) is not None
    except (ImportError, AttributeError, ValueError):
        return False


# ------------------------------------------------------------------ API differences
def dtype_kwarg_name() -> str:
    """Name of the dtype keyword argument of `from_pretrained`.

    transformers >= 4.56 uses `dtype` (`torch_dtype` still works but is
    deprecated); older versions only accept `torch_dtype`.

    Note: this must be decided by version, not by "try `dtype`, fall back on
    TypeError". `from_pretrained` accepts `**kwargs`, so older versions silently
    swallow `dtype=` as a config field and load the model in fp32, doubling
    memory usage without any error.
    """
    return "dtype" if version_ge(transformers_version(), (4, 56)) else "torch_dtype"


# Some TrainingArguments fields were renamed across transformers versions.
# These aliases let train.py pick whichever name the installed version supports.
TRAINING_ARG_ALIASES: Dict[str, List[str]] = {
    "eval_strategy": ["evaluation_strategy"],          # named evaluation_strategy before 4.41
    "evaluation_strategy": ["eval_strategy"],
    "dispatch_batches": ["accelerator_config"],        # moved into accelerator_config in 4.41+
    "tokenizer": ["processing_class"],                 # Trainer argument renamed in 4.46+
    "processing_class": ["tokenizer"],
}


# ------------------------------------------------------------------ requirement table
class ModelRequirement:
    """Environment requirements of one `model.type`."""

    def __init__(
        self,
        transformers_min: Tuple[int, ...] = (4, 30),
        transformers_max: Optional[Tuple[int, ...]] = None,   # exclusive upper bound (< max)
        modules: Sequence[str] = (),
        optional_modules: Sequence[str] = (),
        profile: str = "any",
        note: str = "",
    ) -> None:
        self.transformers_min = transformers_min
        self.transformers_max = transformers_max
        self.modules = list(modules)
        self.optional_modules = list(optional_modules)
        self.profile = profile
        self.note = note


MODEL_REQUIREMENTS: Dict[str, ModelRequirement] = {
    "qwen3_vl_embedding": ModelRequirement(
        transformers_min=(4, 57),
        transformers_max=(6, 0),      # 5.x is also supported
        modules=["transformers.models.qwen3_vl"],
        optional_modules=["qwen_vl_utils"],
        profile="tf457",
        note="Qwen3-VL was added to transformers in 4.57; 5.x is also supported",
    ),
    "jina_clip": ModelRequirement(
        transformers_min=(4, 49),
        transformers_max=(5, 0),      # upstream remote code is not yet compatible with 5.x
        # The model is Hub remote code (not part of transformers); only its dependencies are checked.
        modules=["timm", "einops"],
        optional_modules=["flash_attn", "xformers"],
        profile="tf457",
        note=(
            "jina-clip is implemented as Hub remote code (jinaai/jina-clip-implementation) "
            "and requires trust_remote_code=true; the vision tower needs timm and the text tower needs einops. "
            "flash-attn / xformers are used automatically when installed (optional)."
        ),
    ),
    "jina_clip_v2": ModelRequirement(
        transformers_min=(4, 49),
        transformers_max=(5, 0),
        modules=["timm", "einops"],
        optional_modules=["flash_attn", "xformers"],
        profile="tf457",
        note="Alias of jina_clip (same implementation)",
    ),
    "jina_clip_v1": ModelRequirement(
        transformers_min=(4, 49),
        transformers_max=(5, 0),
        modules=["timm", "einops"],
        optional_modules=["flash_attn", "xformers"],
        profile="tf457",
        note="jina-clip-v1 (English, 224px, no MRL); shares the encoder implementation with v2",
    ),
}

# Recommended environment profiles
PROFILES = {
    "tf457": "transformers 4.57.x / torch 2.6+ / timm + einops",
    "any": "any environment",
}


# ------------------------------------------------------------------ checks
def check_model_env(model_type: str) -> Tuple[bool, List[str]]:
    """Return (ok, reasons). Unknown model types are always accepted."""
    req = MODEL_REQUIREMENTS.get(model_type)
    if req is None:
        return True, []

    reasons: List[str] = []
    raw = package_version("transformers")
    tv = parse_version(raw)
    tv_str = raw or "not installed"

    if raw is None:
        # Not installed: fail explicitly instead of treating it as "unknown version".
        return False, ["transformers is not installed"]

    if not version_ge(tv, req.transformers_min):
        reasons.append(
            f"requires transformers>={'.'.join(map(str, req.transformers_min))}, found {tv_str}"
        )
    if req.transformers_max is not None and not version_lt(tv, req.transformers_max):
        reasons.append(
            f"requires transformers<{'.'.join(map(str, req.transformers_max))}, found {tv_str}"
        )
    for mod in req.modules:
        if not has_module(mod):
            reasons.append(f"missing module {mod}")

    return (not reasons), reasons


def require_model_env(model_type: str) -> None:
    """Raise RuntimeError with an actionable message if the environment is unsuitable."""
    ok, reasons = check_model_env(model_type)
    if ok:
        return
    req = MODEL_REQUIREMENTS[model_type]
    hint = PROFILES.get(req.profile, req.profile)
    raise RuntimeError(
        f"The environment does not satisfy model.type='{model_type}':\n"
        + "\n".join(f"  - {r}" for r in reasons)
        + (f"\n  note: {req.note}" if req.note else "")
        + f"\n  recommended environment: {hint}"
        + "\n  See requirements.txt and the 'Requirements' section of README.md."
    )


def active_profile() -> str:
    """Which profile the current environment resembles (by transformers version; for logging)."""
    tv = transformers_version()
    if not tv:
        return "unknown"
    if not version_lt(tv, (5, 0)):
        return "tf5x"          # only qwen3vl can run
    if version_ge(tv, (4, 57)):
        return "tf457"         # recommended
    if version_ge(tv, (4, 49)):
        return "tf4x"          # jina_clip runs, qwen3vl does not
    return "legacy"


def available_models() -> Dict[str, Tuple[bool, List[str]]]:
    return {name: check_model_env(name) for name in sorted(MODEL_REQUIREMENTS)}


def describe_env() -> str:
    lines = [
        f"python           {sys.version.split()[0]}",
        f"torch            {package_version('torch') or 'not installed'}",
        f"transformers     {package_version('transformers') or 'not installed'}",
        f"peft             {package_version('peft') or 'not installed'}",
        f"accelerate       {package_version('accelerate') or 'not installed'}",
        f"qwen_vl_utils    {package_version('qwen_vl_utils') or 'not installed'}",
        f"timm             {package_version('timm') or 'not installed'}      (required by the jina_clip vision tower)",
        f"einops           {package_version('einops') or 'not installed'}      (required by the jina_clip text tower)",
        f"flash_attn       {package_version('flash_attn') or 'not installed'}      (optional, used if installed)",
        f"xformers         {package_version('xformers') or 'not installed'}      (optional, jina_clip vision x-attn)",
        f"dtype kwarg      {dtype_kwarg_name()}",
        f"env profile      {active_profile()}",
    ]
    return "\n".join(lines)
