"""Configuration system.

Design principle: **component configs are free-form dicts**; only the `type`
field is a framework convention. A new loss / model can read any extra
hyper-parameters directly from YAML without touching a dataclass.

Supports:
  - a main YAML config with `_base_` inheritance
  - command-line overrides: --set model.pooling=mean loss.temperature=0.05
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, List

import yaml


class AttrDict(dict):
    """A dict that also supports attribute access, e.g. `cfg.model.pooling`."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def __setattr__(self, key, value):
        self[key] = _wrap(value)

    def get(self, key, default=None):  # noqa: A003
        return super().get(key, default)


def _wrap(obj):
    if isinstance(obj, dict):
        return AttrDict({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def to_plain(obj):
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_plain(v) for v in obj]
    return obj


def deep_merge(base: Dict, override: Dict) -> Dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            # If the component type changes (e.g. loss.type infonce -> weighted_sum),
            # replace the whole section instead of merging stale parameters.
            old_type, new_type = out[k].get("type"), v.get("type")
            if old_type is not None and new_type is not None and old_type != new_type:
                out[k] = copy.deepcopy(v)
            else:
                out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_raw(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    bases = cfg.pop("_base_", None)
    if not bases:
        return cfg
    if isinstance(bases, str):
        bases = [bases]
    merged: Dict = {}
    for b in bases:
        b_path = b if os.path.isabs(b) else os.path.join(os.path.dirname(path), b)
        merged = deep_merge(merged, _load_raw(b_path))
    return deep_merge(merged, cfg)


def apply_overrides(cfg: Dict, overrides: List[str] | None) -> Dict:
    """`overrides` look like ["model.pooling=mean", "train.learning_rate=1e-5"]."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got: {item}")
        key, raw = item.split("=", 1)
        try:
            value = yaml.safe_load(raw)
        except Exception:
            value = raw
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"Override path conflict: {key}")
        node[parts[-1]] = value
    return cfg


def load_config(path: str, overrides: List[str] | None = None) -> AttrDict:
    cfg = _load_raw(path)
    cfg = apply_overrides(cfg, overrides)
    cfg.setdefault("model", {})
    cfg.setdefault("loss", {})
    cfg.setdefault("data", {})
    cfg.setdefault("train", {})
    return _wrap(cfg)


def dump_config(cfg: AttrDict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_plain(cfg), f, allow_unicode=True, sort_keys=False)
