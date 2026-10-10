"""Instruction resolution, **shared by training and evaluation**.

Why a dedicated module? Inconsistent instructions are a subtle class of bugs:
the model's `_compose_text` has three fallback levels (record.instruction ->
task_instructions -> global prompt). If an evaluation script forgets to inject
instructions it silently falls back to the third level; nothing fails, but the
embeddings no longer match the training distribution.

This module therefore provides a single entry point:
    resolve_instruction(table, role, record) -> Optional[str]

Lookup order (identical to the order used when the dataset injects instructions):
    1. f"{role}_{modality}"   e.g. query_image / doc_text (per-modality)
    2. role                   e.g. query / doc (fallback)
    3. None                   defer to the model's global prompt config

`modality` comes from Record.modality: text / image / mixed.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

# Built-in presets; keep in sync with `task_instructions` in configs/*.yaml.
BUILTIN_TASK_INSTRUCTIONS: Dict[str, Dict[str, str]] = {
    "default": {
        "query": "Represent the user's query for retrieving relevant content.",
        "doc": "Represent the candidate content for retrieval.",
    },
    "multiview": {
        "query_text": "Given a text query, retrieve the matching content.",
        "query_image": "Given an image query, retrieve the matching content.",
        "doc_text": "Represent this text document for retrieval.",
        "doc_image": "Represent this image document for retrieval.",
    },
}


def record_modality(record: Any) -> str:
    """text / image / mixed. Also works for records without a `modality` attribute."""
    modality = getattr(record, "modality", None)
    if modality:
        return str(modality)
    has_img = bool(getattr(record, "images", None))
    has_txt = bool(getattr(record, "text", None))
    if has_img and has_txt:
        return "mixed"
    return "image" if has_img else "text"


def instruction_key(
    table: Dict[str, str], role: str, modality: str
) -> Optional[str]:
    """Return the matching key (per-modality key first), or None."""
    for key in (f"{role}_{modality}", role):
        if table.get(key):
            return key
    return None


def resolve_instruction(
    table: Optional[Dict[str, str]], role: str, record: Any
) -> Optional[str]:
    if not table:
        return None
    key = instruction_key(table, role, record_modality(record))
    return table.get(key) if key else None


def resolve_instruction_verbose(
    table: Optional[Dict[str, str]], role: str, record: Any
) -> Tuple[Optional[str], Optional[str]]:
    """Return (instruction, matched key); used by evaluation scripts for auditing."""
    if not table:
        return None, None
    key = instruction_key(table, role, record_modality(record))
    return (table.get(key) if key else None), key


# ----------------------------------------------------------------- loading
def _extract_table(obj: Any) -> Dict[str, Dict[str, str]]:
    """Extract the task -> {role_key: instruction} table from any supported source."""
    if not isinstance(obj, dict):
        raise TypeError("task_instructions must be a dict")
    # A full training config: configs/xxx.yaml
    data = obj.get("data")
    if isinstance(data, dict) and isinstance(data.get("task_instructions"), dict):
        return dict(data["task_instructions"])
    # Only the `data` section
    if isinstance(obj.get("task_instructions"), dict):
        return dict(obj["task_instructions"])
    # Already a task -> table mapping
    return dict(obj)


def load_task_instructions(source: Optional[str]) -> Dict[str, Dict[str, str]]:
    """`source` can be:

      * None / "builtin"  -> built-in presets
      * a YAML / JSON path (**recommended: pass the training config itself**;
        `data.task_instructions` is extracted from it so evaluation matches training)
      * an inline JSON string, e.g. '{"multiview": {"query_text": "..."}}'
    """
    if not source or source == "builtin":
        return {k: dict(v) for k, v in BUILTIN_TASK_INSTRUCTIONS.items()}

    text = source.strip()
    if text.startswith("{"):
        return _extract_table(json.loads(text))

    if not os.path.isfile(text):
        raise FileNotFoundError(
            f"Instruction source {text} is neither a built-in preset nor a valid path / JSON string"
        )

    if text.endswith((".yaml", ".yml")):
        # Training configs may use `_base_` inheritance; use the framework loader to resolve it.
        try:
            from ..config import load_config, to_plain

            return _extract_table(to_plain(load_config(text)))
        except Exception:
            import yaml

            with open(text, "r", encoding="utf-8") as f:
                return _extract_table(yaml.safe_load(f) or {})

    with open(text, "r", encoding="utf-8") as f:
        return _extract_table(json.load(f))
