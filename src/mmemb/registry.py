"""Minimal registry: every swappable component (model / loss / pooler / dataset)
is registered here.

To add a component:
    @MODELS.register("my_model")
    class MyModel(BaseEmbedder): ...
then set `model.type: my_model` in the YAML config.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List


class Registry:
    def __init__(self, name: str) -> None:
        self.name = name
        self._table: Dict[str, Any] = {}

    def register(self, key: str | None = None) -> Callable:
        def deco(obj):
            k = key or getattr(obj, "alias", None) or obj.__name__.lower()
            if k in self._table and self._table[k] is not obj:
                raise KeyError(f"[{self.name}] duplicate registration: {k}")
            self._table[k] = obj
            return obj

        return deco

    def get(self, key: str):
        if key not in self._table:
            raise KeyError(
                f"[{self.name}] unknown component '{key}'; available: {sorted(self._table)}"
            )
        return self._table[key]

    def build(self, key: str, *args, **kwargs):
        return self.get(key)(*args, **kwargs)

    def keys(self) -> List[str]:
        return sorted(self._table)

    def __contains__(self, key: str) -> bool:
        return key in self._table


MODELS = Registry("models")
LOSSES = Registry("losses")
POOLERS = Registry("poolers")
DATASETS = Registry("datasets")
