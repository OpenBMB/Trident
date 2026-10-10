"""Training engine built on the HF Trainer.

Reusing the Trainer gives DDP / DeepSpeed / bf16 / gradient accumulation /
warmup + schedulers / checkpointing and resuming / wandb & tensorboard for free.

Only four things are customized:
  1. compute_loss  -- a single forward through ContrastiveWrapper
  2. log           -- also log loss-internal metrics such as acc / temperature
  3. evaluate      -- retrieval metrics (Recall@K / MRR)
  4. _save         -- save in mmemb's encoder format (LoRA saves only the adapter)
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback

from ..eval.retrieval import collect_examples, evaluate_retrieval
from ..utils.misc import get_logger


logger = get_logger(__name__)


class EpochSeedCallback(TrainerCallback):
    """Notify the dataset at every epoch so it resamples hard negatives."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def on_epoch_begin(
        self,
        args,
        state,
        control,
        **kwargs: Any,
    ):
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(int(state.epoch or 0))


class EmbeddingTrainer(Trainer):
    def __init__(
        self,
        *args,
        eval_k_list: Sequence[int] = (1, 5, 10),
        eval_dim: Optional[int] = None,
        eval_balance: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.eval_k_list = list(eval_k_list)
        self.eval_dim = eval_dim
        # Modality-balance diagnostics on the eval set (eval_balance/* metrics). Embeddings are
        # already computed, so this only costs a few einsums; enabled by default.
        self.eval_balance = bool(eval_balance)

        # Sum of custom metrics over the training steps of the current logging period.
        #
        # Counts are kept per key: some metrics are not produced every step (e.g. modality_balance
        # with weight=0 may be computed every few steps); dividing by all steps would shrink them.
        self._metric_sums: Dict[str, float] = defaultdict(float)
        self._metric_counts: Dict[str, int] = defaultdict(int)
        self._metric_count: int = 0

    # ------------------------------------------------------------------ loss
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        **kwargs: Any,
    ):
        """Run one ContrastiveWrapper forward and collect auxiliary training metrics."""

        outputs = model(**inputs)
        loss = outputs["loss"]

        # Read metrics from the model's instance attribute.
        #
        # Python floats are not put into `outputs` to prevent DataParallel/DDP from
        # trying to gather these scalars.
        metrics = getattr(model, "_last_metrics", {})

        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().float().item()

            self._metric_sums[key] += float(value)
            self._metric_counts[key] += 1

        self._metric_count += 1

        if return_outputs:
            return loss, outputs

        return loss

    # ------------------------------------------------------------------- log
    def log(
        self,
        logs: Dict[str, float],
        *args,
        **kwargs: Any,
    ):
        """Add model metrics such as acc and temperature to the training logs."""

        # Only attach these metrics when the Trainer logs the training loss.
        # Evaluation logs usually do not contain "loss", so training metrics are not cleared by mistake.
        if self._metric_count > 0 and "loss" in logs:
            for key, value_sum in self._metric_sums.items():
                logs[key] = round(
                    value_sum / max(self._metric_counts.get(key, 1), 1),
                    6,
                )

            self._metric_sums.clear()
            self._metric_counts.clear()
            self._metric_count = 0

        return super().log(logs, *args, **kwargs)

    # ------------------------------------------------------------------ eval
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **_unused: Any,
    ) -> Dict[str, float]:
        """Run retrieval evaluation on rank 0 only and broadcast the results to the other ranks.

        Distributed flow:

        1. all ranks synchronize before evaluation;
        2. rank 0 runs the full retrieval evaluation;
        3. rank 0 broadcasts the metrics to all ranks;
        4. all ranks trigger on_evaluate with identical metrics;
        5. all ranks synchronize again before returning to the training loop.

        This avoids computing Recall@K / MRR on every GPU while ensuring that Trainer
        logic such as load_best_model_at_end and metric_for_best_model sees consistent
        results on all processes.
        """

        del ignore_keys  # not used by the custom retrieval evaluation.

        dataset = (
            eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )

        if dataset is None:
            return {}

        # ---------------------------------------------------------- pre-eval barrier
        # Make sure all ranks have finished the current training step before evaluating.
        self.accelerator.wait_for_everyone()

        metrics: Dict[str, float] = {}

        # ------------------------------------------------------------ rank 0 eval
        if self.is_world_process_zero():
            model = self.accelerator.unwrap_model(self.model)
            was_training = model.training

            model.eval()

            try:
                raw_metrics = evaluate_retrieval(
                    model.encoder,
                    collect_examples(dataset),
                    batch_size=self.args.per_device_eval_batch_size,
                    k_list=self.eval_k_list,
                    dim=self.eval_dim,
                    balance=self.eval_balance,
                )

                metrics = {
                    f"{metric_key_prefix}_{key}": round(float(value), 6)
                    for key, value in raw_metrics.items()
                }

            finally:
                # Restore the original train/eval mode even if evaluation raises.
                if was_training:
                    model.train()

        # ------------------------------------------------------- metrics broadcast
        # Every process must enter this collective.
        #
        # Without distributed training, keep the metrics produced by rank 0.
        if dist.is_available() and dist.is_initialized():
            object_list: List[Dict[str, float]] = [metrics]

            # For NCCL, pass the CUDA device of the current process explicitly;
            # for CPU backends such as Gloo, do not pass a device.
            backend = dist.get_backend()

            if backend == "nccl":
                dist.broadcast_object_list(
                    object_list,
                    src=0,
                    device=self.accelerator.device,
                )
            else:
                dist.broadcast_object_list(
                    object_list,
                    src=0,
                )

            metrics = object_list[0]

        # --------------------------------------------------------------- logging
        # Only the global main process writes to wandb / tensorboard / stdout to avoid duplicate logs.
        if self.is_world_process_zero():
            self.log(metrics)

        # All ranks trigger callbacks and see exactly the same metrics.
        #
        # This is especially important for configurations such as:
        #
        #   load_best_model_at_end=True
        #   metric_for_best_model="mrr"
        #
        # otherwise the TrainerControl state could differ across ranks.
        self.control = self.callback_handler.on_evaluate(
            self.args,
            self.state,
            self.control,
            metrics,
        )

        # ---------------------------------------------------------- post-eval barrier
        # Prevent non-main processes from starting the next DDP forward while rank 0 is still evaluating.
        self.accelerator.wait_for_everyone()

        return metrics

    # ------------------------------------------------------------------ save
    def _save(
        self,
        output_dir: Optional[str] = None,
        state_dict=None,
    ) -> None:
        """Save the model in mmemb's custom format.

        The HF Trainer normally only calls _save on processes with should_save=True.
        This adds an extra guard so that external code or custom callbacks calling
        _save directly do not make several processes write to the same directory.
        """

        del state_dict  # saving is handled by the model's own save_pretrained.

        if not self.args.should_save:
            return

        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        model = self.accelerator.unwrap_model(self.model)

        if hasattr(model, "save_pretrained"):
            model.save_pretrained(output_dir)
        else:  # pragma: no cover
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, "pytorch_model.bin"),
            )

        torch.save(
            self.args,
            os.path.join(output_dir, "training_args.bin"),
        )

        logger.info("Checkpoint written to %s", output_dir)