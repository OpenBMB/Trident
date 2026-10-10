"""Share training progress from the main process with dataloader workers.

## Why this is needed

The data-side curriculum samples negatives of different difficulty depending
on training progress, but `Dataset.__getitem__` runs inside **dataloader worker
subprocesses**. A plain Python attribute updated by a callback in the main
process is invisible to the workers' copies, so the curriculum would never move.

`multiprocessing.Value` lives in shared memory: forked workers and the main
process read/write the same block, so a value written once per optimizer step
is immediately visible to the workers.

## Limitations

1. **Only works with the `fork` start method** (Linux default). With `spawn`
   (Windows / macOS, or an explicit `multiprocessing_context`), workers only
   see the initial value and the curriculum stays at its starting difficulty.
   `SharedProgress.check_start_method()` emits a warning in that case.
2. **Workers may lag slightly behind**: the dataloader prefetches, so a batch is
   often built a few steps earlier. The curriculum changes slowly, so this is fine.
3. `persistent_workers=True` does not affect correctness (shared memory stays
   valid); with `num_workers=0` the main process reads/writes directly.
"""
from __future__ import annotations

import multiprocessing as mp
from typing import Optional

from ..utils.misc import get_logger

logger = get_logger(__name__)


class SharedProgress:
    """Training progress p ∈ [0, 1] shared across processes."""

    def __init__(self, initial: float = 0.0) -> None:
        self._value = mp.Value("d", float(initial))

    def get(self) -> float:
        return float(self._value.value)

    def set(self, p: float) -> None:
        self._value.value = min(1.0, max(0.0, float(p)))

    @staticmethod
    def check_start_method() -> Optional[str]:
        """Warn early under `spawn`, where shared-memory updates are not visible."""
        try:
            method = mp.get_start_method(allow_none=True)
        except Exception:
            return None
        if method and method != "fork":
            msg = (
                f"multiprocessing start method is {method} (not fork); dataloader workers "
                "cannot see the progress written by the main process, so the data-side "
                "curriculum will stay at its initial difficulty. "
                "Set train.dataloader_num_workers to 0 or use fork."
            )
            logger.warning("%s", msg)
            return msg
        return None
