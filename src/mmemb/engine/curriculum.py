"""Feed training progress (global_step / max_steps) to the data-side hard-negative curriculum.

The data-side curriculum (data.hard_negatives, which moves through lexical
similarity ranks over time) needs to know "current step / total steps", but it
runs inside dataloader workers and cannot see the Trainer state. A
TrainerCallback therefore writes progress into shared memory in the main
process, and workers read it whenever they fetch a sample.
"""
from __future__ import annotations

from typing import Any, Optional

from transformers import TrainerCallback

from ..utils.misc import get_logger

logger = get_logger(__name__)


class DataCurriculumCallback(TrainerCallback):
    """Write training progress into shared memory for the data-side curriculum in dataloader workers.

    Progress is defined as global_step / max_steps, so the full curriculum is
    traversed even with `num_train_epochs: 1` (an epoch-based curriculum would do
    nothing in single-epoch training).
    """

    def __init__(self, progress: Any, total_steps: Optional[int] = None) -> None:
        self.progress = progress
        self.total_steps = None if total_steps in (None, "auto", -1, 0) else int(total_steps)

    @property
    def enabled(self) -> bool:
        return self.progress is not None

    def _update(self, state) -> None:
        total = int(self.total_steps or state.max_steps or 0)
        if total > 0:
            self.progress.set(int(state.global_step or 0) / total)

    def on_train_begin(self, args, state, control, **kwargs: Any):
        self._update(state)
        if args.dataloader_num_workers > 0:
            logger.info(
                "Data-side curriculum enabled (dataloader_num_workers=%d): progress is passed to workers "
                "via shared memory; prefetching causes a lag of a few steps, which is harmless.",
                args.dataloader_num_workers,
            )

    def on_step_end(self, args, state, control, **kwargs: Any):
        self._update(state)
