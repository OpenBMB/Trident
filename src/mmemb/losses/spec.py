"""**Single source of truth** for the loss config: optional modules are registered here.

## Modules

    false_negative   -- false-negative masking (other rows' positives that are also
                        my positives are removed from the denominator)
    matryoshka       -- Matryoshka (multi-dimension) joint training
    modality_balance -- explicit modality-balance regularizer (flattens the
                        dispersion inside k x k similarity blocks; Appendix G)

## Layout

    loss:
      type: infonce
      temperature: 0.02          # core loss parameters stay flat
      symmetric: true
      modules:
        false_negative:   {enable: true, mask_sibling_positives: false}
        matryoshka:       {enable: true, dims: [256, 512, 1024]}
        modality_balance: {enable: true, weight: 0, scope: all}

Three rules:

  1. Every module has `enable`; `enable: false` = not constructed at all, zero overhead.
  2. All other fields are that module's own parameters.
  3. Modules with a `weight` (kind=aux) add a term to the loss. With `weight: 0`
     the term is **still computed and logged to TensorBoard but does not take part
     in backprop**, which is convenient as an ablation control.

Legacy flat keys (e.g. `matryoshka_dims: [...]`) are translated automatically by
`canonicalize` with a WARNING.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# modifier = changes the behavior of the core loss (no weight)
KIND_MODIFIER = "modifier"
# aux = adds a term to the total loss (has a weight; weight=0 means log-only)
KIND_AUX = "aux"


@dataclass
class ModuleSpec:
    name: str
    kind: str
    summary: str
    defaults: Dict[str, Any] = field(default_factory=dict)
    # legacy flat key -> sub-key of this module
    legacy_flat: Dict[str, str] = field(default_factory=dict)
    # legacy top-level block name (content moved verbatim)
    legacy_block: Optional[str] = None
    # only meaningful with multi-view data
    requires_multiview: bool = False

    @property
    def is_aux(self) -> bool:
        return self.kind == KIND_AUX


# ---------------------------------------------------------------------------
# Module registry. Adding a module = adding an entry here (no YAML parsing / train.py changes).
# ---------------------------------------------------------------------------
MODULE_SPECS: List[ModuleSpec] = [
    ModuleSpec(
        name="false_negative",
        kind=KIND_MODIFIER,
        summary="false-negative masking: remove candidates that are actually my positives from the denominator",
        defaults={
            "enable": True,
            "mask_positive_ids": True,   # candidate doc_id ∈ my positive_ids (main rule)
            "mask_same_group": True,     # candidate shares my group_id (exclusive cluster)
            "mask_by_query_id": True,    # symmetric direction: candidate query has my query_id
            "mask_same_example": True,   # sibling view groups split from the same row
            # True: remove the other positives of the same sample from each positive's
            # denominator (independent per-positive normalization).
            # False: all positives share one softmax denominator (Multi-Positive View
            # InfoNCE, paper Eq. 4). The Trident configs set this to False explicitly.
            "mask_sibling_positives": True,
        },
        legacy_flat={
            "mask_false_negatives": "mask_positive_ids",
            "mask_same_group": "mask_same_group",
            "mask_by_query_id": "mask_by_query_id",
            "mask_same_example": "mask_same_example",
            "mask_sibling_positives": "mask_sibling_positives",
        },
    ),
    ModuleSpec(
        name="matryoshka",
        kind=KIND_MODIFIER,
        summary="Matryoshka joint training over several embedding dimensions",
        defaults={"enable": False, "dims": None, "weights": None},
        legacy_flat={"matryoshka_dims": "dims", "matryoshka_weights": "weights"},
    ),
    ModuleSpec(
        name="modality_balance",
        kind=KIND_AUX,
        summary=(
            "modality-balance regularizer: flattens dispersion inside k x k similarity blocks "
            "(weight=0: log only, no gradient)"
        ),
        defaults={
            # Default enable=true + weight=0: training is unchanged; only the balance/*
            # diagnostic curves are logged. Set a positive weight to enable the regularizer.
            "enable": True,
            "weight": 0.0,
            "scope": "positives",        # positives | all
            "pos_center": "mean",        # mean | max -- positive blocks
            "neg_center": "mean",        # mean | min -- negative blocks (scope=all only)
            "metric": "variance",        # variance | std | pairwise
            "neg_weight": 1.0,           # weight of negative blocks relative to positive blocks (scope=all)
            "include_hard_negatives": True,
            "detach_center": False,
            # with weight=0 (log only): compute every N steps; with weight>0: every step
            "log_every": 1,
        },
        legacy_flat={
            "balance_weight": "weight",
            "balance_scope": "scope",
            "balance_pos_center": "pos_center",
            "balance_neg_center": "neg_center",
            "balance_metric": "metric",
            "balance_neg_weight": "neg_weight",
            "balance_include_hard_negatives": "include_hard_negatives",
            "balance_detach_center": "detach_center",
        },
        legacy_block="modality_balance",
        requires_multiview=True,
    ),
]

SPEC_BY_NAME: Dict[str, ModuleSpec] = {s.name: s for s in MODULE_SPECS}

# Core loss parameters (flat; not part of any module)
CORE_KEYS = {
    "type",
    "temperature",
    "learnable_temperature",
    "min_temperature",
    "use_inbatch_negatives",
    "inbatch_scope",
    "symmetric",
    "symmetric_weight",
    "label_smoothing",
    "multi_positive_reduction",
    # injected from the multi-view config of the data section (view_config), not written by users
    "num_query_views",
    "num_positives",
    "view_strategy",
}

_FLAT_TO_MODULE: Dict[str, Tuple[str, str]] = {
    old: (spec.name, new)
    for spec in MODULE_SPECS
    for old, new in spec.legacy_flat.items()
}
_BLOCK_TO_MODULE: Dict[str, str] = {
    spec.legacy_block: spec.name for spec in MODULE_SPECS if spec.legacy_block
}


# ---------------------------------------------------------------------------
def canonicalize(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Normalize any layout (legacy flat / legacy nested / `modules`) into canonical form.

    Returns (canonical config, list of warnings). In the canonical config `modules`
    is always complete: every module has an entry, missing fields use defaults.
    """
    cfg = copy.deepcopy(dict(cfg or {}))
    warnings: List[str] = []
    modules: Dict[str, Dict[str, Any]] = {s.name: {} for s in MODULE_SPECS}

    # 1) `loss.modules` takes precedence
    for name, sub in dict(cfg.pop("modules", None) or {}).items():
        if name not in SPEC_BY_NAME:
            warnings.append(
                f"loss.modules.{name} is not a known module and is ignored. "
                f"Available modules: {', '.join(sorted(SPEC_BY_NAME))}"
            )
            continue
        if sub is None or sub is False:
            modules[name] = {"enable": False}
        elif sub is True:
            modules[name] = {"enable": True}
        elif isinstance(sub, dict):
            modules[name] = dict(sub)
        else:
            modules[name] = {"enable": True}

    # 2) legacy top-level nested blocks
    for block, name in _BLOCK_TO_MODULE.items():
        if block not in cfg:
            continue
        raw = cfg.pop(block)
        if modules[name]:
            warnings.append(
                f"Both loss.{block} and loss.modules.{name} are set; using modules (legacy block ignored)"
            )
            continue
        if raw is None or raw is False:
            modules[name] = {"enable": False}
        elif raw is True:
            modules[name] = {"enable": True}
        elif isinstance(raw, dict):
            modules[name] = dict(raw)
        warnings.append(f"loss.{block} is a legacy layout; please use loss.modules.{name}")

    # 3) legacy flat keys: matryoshka_* / mask_*
    moved: Dict[str, List[str]] = {}
    for old in list(cfg.keys()):
        if old in CORE_KEYS or old not in _FLAT_TO_MODULE:
            continue
        name, new = _FLAT_TO_MODULE[old]
        value = cfg.pop(old)
        if new in modules[name]:
            warnings.append(f"loss.{old} conflicts with loss.modules.{name}.{new}; using modules")
            continue
        modules[name][new] = value
        moved.setdefault(name, []).append(f"{old} -> modules.{name}.{new}")
    for name, items in moved.items():
        warnings.append(f"Migrated {len(items)} legacy flat loss key(s): " + "; ".join(items))

    # 4) fill defaults + infer `enable`
    for spec in MODULE_SPECS:
        written = modules[spec.name]
        sub = {**spec.defaults, **written}
        if "enable" not in written:
            if spec.name == "matryoshka":
                sub["enable"] = bool(written.get("dims"))
            else:
                sub["enable"] = bool(spec.defaults.get("enable", False))
        sub["enable"] = bool(sub.get("enable", False))
        modules[spec.name] = sub

    # 5) remaining unknown keys
    for key in cfg:
        if key not in CORE_KEYS:
            warnings.append(
                f"loss.{key} is not a known parameter and is ignored (possibly a typo)"
            )

    cfg["modules"] = modules
    return cfg, warnings


# ---------------------------------------------------------------------------
def identity_requirements(canonical: Dict[str, Any]) -> Dict[str, bool]:
    """Whether the data pipeline must collect identity info; decided by the false_negative module."""
    fn = canonical["modules"]["false_negative"]
    if not fn["enable"]:
        return {"collect_uids": False, "collect_identity": False}
    any_rule = any(
        bool(fn[k])
        for k in ("mask_positive_ids", "mask_same_group", "mask_by_query_id", "mask_same_example")
    )
    return {
        "collect_uids": bool(fn["mask_positive_ids"]),
        "collect_identity": any_rule,
    }


def enabled_modules(canonical: Dict[str, Any]) -> List[str]:
    return [s.name for s in MODULE_SPECS if canonical["modules"][s.name]["enable"]]


def describe(canonical: Dict[str, Any]) -> str:
    """Print a summary table at startup showing which modules are enabled."""
    mods = canonical["modules"]
    lines = [f"loss.type = {canonical.get('type', 'infonce')}", "loss modules:"]
    for spec in MODULE_SPECS:
        sub = mods[spec.name]
        if not sub["enable"]:
            lines.append(f"  [ ] {spec.name:<16} off")
            continue
        detail = [f"{k}={v}" for k, v in sub.items() if k != "enable" and v != spec.defaults.get(k)]
        if spec.is_aux:
            # The weight of an aux module decides whether it actually affects training; always print it
            w = float(sub.get("weight", 0.0) or 0.0)
            tag = f"weight={w}" + (" (log only, no gradient)" if w == 0 else "")
            detail = [tag] + [d for d in detail if not d.startswith("weight=")]
        lines.append(f"  [x] {spec.name:<16} " + (" ".join(detail) if detail else "on (default parameters)"))
    return "\n".join(lines)


def check_consistency(canonical: Dict[str, Any], views: Dict[str, Any]) -> List[str]:
    """Cross-section consistency check: loss module prerequisites vs. the data config."""
    warns: List[str] = []
    mods = canonical["modules"]
    is_mv = bool(views.get("is_multiview"))
    for spec in MODULE_SPECS:
        sub = mods[spec.name]
        if not sub["enable"]:
            continue
        # aux modules with weight=0 are log-only and silently produce no metrics for
        # single-view data; no need to warn on every start
        if spec.is_aux and not float(sub.get("weight", 0.0) or 0.0):
            continue
        if spec.requires_multiview and not is_mv:
            warns.append(
                f"loss.modules.{spec.name} requires multi-view data (data.num_query_views / "
                f"num_positives > 1); the data is single-view, so it has no effect"
            )
    return warns
