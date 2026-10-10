"""Training monitoring: report loss-internal metrics and collapse probes to wandb / tensorboard.

Why this file exists
--------------------
The HF Trainer only logs `loss / grad_norm / learning_rate / epoch` by default.
Metrics computed inside InfoNCE (`acc / pos_sim / n_cand / temperature`) live in
`LossOutput.metrics` and would otherwise be dropped.

Approach: a forward hook on the loss module captures `LossOutput.metrics`, and a
TrainerCallback placed first in the callback list injects them into `logs`, so
wandb / tensorboard / console all receive them without modifying trainer.py.

It also provides contrastive-specific collapse monitoring (metrics prefixed with
`contrast/`), a failure mode that is hard to spot from the loss value alone.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional

import torch
from transformers import TrainerCallback

from ..losses.base import BaseLoss
from ..utils.misc import get_logger

logger = get_logger(__name__)


# ====================================================================== metric buffer
class MetricAccumulator:
    """Accumulate loss metrics between two log events and report their mean.

    With logging_steps=10, the acc of all 10 steps is averaged instead of
    reporting only the last step's value.

    Counts are kept **per key**: some metrics are not produced every step
    (e.g. modality_balance with weight=0 and log_every>1), and dividing by the
    total number of steps would systematically shrink them.
    """

    def __init__(self) -> None:
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}
        self._count = 0

    def update(self, metrics: Dict[str, Any]) -> None:
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if math.isfinite(float(v)):
                    self._sums[k] = self._sums.get(k, 0.0) + float(v)
                    self._counts[k] = self._counts.get(k, 0) + 1
        self._count += 1

    def drain(self) -> Dict[str, float]:
        if self._count == 0:
            return {}
        out = {k: v / max(self._counts.get(k, 1), 1) for k, v in self._sums.items()}
        self._sums.clear()
        self._counts.clear()
        self._count = 0
        return out

    @property
    def empty(self) -> bool:
        return self._count == 0


# ====================================================================== collapse probe
@torch.no_grad()
def _collapse_probe(loss_input) -> Dict[str, float]:
    """Detect embedding collapse, the most insidious failure mode of contrastive learning.

    Under collapse the loss stays at ln(#candidates), which is easy to miss
    (e.g. "4.159 is ln(64)"). The criteria are logged directly as curves:

      contrast/q_cos_mean  mean cosine between queries. Healthy: 0.1-0.5;
                           close to 1.0 = all queries are encoded into the same vector
      contrast/d_cos_mean  mean cosine between docs, same as above
      contrast/q_std       per-dimension std of query embeddings; close to 0 = collapsed
    """
    out: Dict[str, float] = {}
    for name, tensor in (("q", getattr(loss_input, "q", None)),
                         ("d", getattr(loss_input, "d", None))):
        if tensor is None:
            continue
        x = tensor.detach().float()
        if x.ndim == 3:          # docs are [B, G, D]; only look at the positive of each group
            x = x[:, 0]
        if x.size(0) < 2:
            continue
        x = torch.nn.functional.normalize(x, dim=-1)
        sim = x @ x.t()
        off = sim[~torch.eye(x.size(0), dtype=torch.bool, device=x.device)]
        out[f"contrast/{name}_cos_mean"] = off.mean().item()
        out[f"contrast/{name}_cos_max"] = off.max().item()
        out[f"contrast/{name}_std"] = x.std(dim=0).mean().item()
    return out


# ====================================================================== Callback
class TrainingMonitorCallback(TrainerCallback):
    """Inject loss-internal metrics and collapse probes into the Trainer logs.

    Must be placed first in the callback list (handled by `setup_monitoring`) so that
    WandbCallback / TensorBoardCallback receive the completed logs.

    Parameters
    ----
    probe_every : run the collapse probe every N steps. The probe costs one B x B
                  matmul (cheap) but is not needed every step. Set 0 to disable.
    """

    def __init__(self, probe_every: int = 25) -> None:
        self.acc = MetricAccumulator()
        self.probe_every = int(probe_every)
        self._probe: Dict[str, float] = {}
        self._handle = None
        self._step = 0

    # ---------------- hook: capture metrics from the loss module ----------------
    def _attach(self, model) -> None:
        target = None
        for module in model.modules():
            if isinstance(module, BaseLoss):
                target = module
                break
        if target is None:
            logger.warning("No BaseLoss submodule found in the model; loss-internal metrics cannot be reported")
            return

        def hook(module, args, kwargs, output):
            metrics = getattr(output, "metrics", None)
            if metrics:
                self.acc.update(metrics)
            self._step += 1
            if self.probe_every and self._step % self.probe_every == 0:
                loss_input = args[0] if args else kwargs.get("x")
                if loss_input is not None:
                    try:
                        self._probe = _collapse_probe(loss_input)
                    except Exception as e:  # a failing probe must never affect training
                        logger.debug("Collapse probe failed: %s", e)

        self._handle = target.register_forward_hook(hook, with_kwargs=True)
        logger.info("Metric hook attached to %s", type(target).__name__)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self._attach(model)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return control

    # ---------------- inject into logs ----------------
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not state.is_world_process_zero:
            return control
        if any(k.startswith("eval_") for k in logs):
            return control  # do not mix training metrics into evaluation logs

        logs.update(self.acc.drain())
        logs.update(self._probe)

        # Collapse reference: a loss stuck at ln(n_cand) means all embeddings are identical
        n_cand = logs.get("n_cand")
        loss = logs.get("loss")
        if n_cand and n_cand > 1:
            ceiling = math.log(float(n_cand))
            logs["contrast/loss_ceiling"] = ceiling
            if isinstance(loss, (int, float)):
                # >0 means the model has learned something; ≈0 indicates collapse
                logs["contrast/loss_margin"] = ceiling - float(loss)
        return control


# ====================================================================== setup
def setup_monitoring(trainer, cfg) -> None:
    """Call once before trainer.train().

    `cfg` can be a dict, a dataclass or an OmegaConf DictConfig.

    Settings are read from the optional `monitor` section of the YAML config:

        monitor:
          wandb:
            enable: true
            project: mmemb
            entity: null            # team name; leave empty for personal accounts
            run_name: null          # defaults to train.run_name / output dir name
            mode: online            # online / offline / disabled
            tags: [qwen3vl, lora]
            notes: "Trident training run"
            watch_model: false      # upload parameter/gradient histograms (bandwidth heavy)
          tensorboard: true
          probe_every: 25
    """
    # ---- support dict / dataclass / OmegaConf configs
    def safe_get(obj, key, default=None):
        """Safely read a value from a dict or a dataclass."""
        if hasattr(obj, "get") and callable(obj.get):
            return obj.get(key, default)
        elif hasattr(obj, key):
            return getattr(obj, key, default)
        else:
            return default

    monitor_cfg = safe_get(cfg, "monitor") or {}
    wandb_cfg = safe_get(monitor_cfg, "wandb") or {}

    report_to: List[str] = []
    if safe_get(wandb_cfg, "enable"):
        if _init_wandb_env(wandb_cfg, trainer):
            report_to.append("wandb")
    if safe_get(monitor_cfg, "tensorboard", True):
        report_to.append("tensorboard")

    if report_to:
        trainer.args.report_to = report_to
        logger.info("Monitoring backends: %s", ", ".join(report_to))
    else:
        trainer.args.report_to = []

    # The metrics callback must come first, otherwise wandb receives incomplete logs
    cb = TrainingMonitorCallback(probe_every=int(safe_get(monitor_cfg, "probe_every", 25)))
    trainer.add_callback(cb)
    handler = trainer.callback_handler
    handler.callbacks.remove(cb)
    handler.callbacks.insert(0, cb)


def _init_wandb_env(wandb_cfg, trainer) -> bool:
    """Set wandb environment variables, which HF's WandbCallback reads at init."""
    try:
        import wandb  # noqa: F401
    except ImportError:
        logger.warning("wandb is not installed (pip install wandb); skipping and using tensorboard only")
        return False

    # ---- support dict / dataclass
    def safe_get(obj, key, default=None):
        if hasattr(obj, "get") and callable(obj.get):
            return obj.get(key, default)
        elif hasattr(obj, key):
            return getattr(obj, key, default)
        else:
            return default

    mode = str(safe_get(wandb_cfg, "mode", "online"))
    os.environ["WANDB_MODE"] = mode
    if mode == "offline":
        logger.info("wandb offline mode. Upload after training with `wandb sync %s`",
                    os.path.join(trainer.args.output_dir, "wandb", "offline-run-*"))

    os.environ.setdefault("WANDB_PROJECT", str(safe_get(wandb_cfg, "project", "mmemb")))
    if safe_get(wandb_cfg, "entity"):
        os.environ.setdefault("WANDB_ENTITY", str(safe_get(wandb_cfg, "entity")))
    if safe_get(wandb_cfg, "notes"):
        os.environ.setdefault("WANDB_NOTES", str(safe_get(wandb_cfg, "notes")))
    if safe_get(wandb_cfg, "tags"):
        os.environ.setdefault("WANDB_TAGS", ",".join(str(t) for t in safe_get(wandb_cfg, "tags")))
    if not safe_get(wandb_cfg, "watch_model", False):
        os.environ.setdefault("WANDB_WATCH", "false")
    # Do not upload checkpoints by default (even LoRA adapters are tens of MB)
    os.environ.setdefault("WANDB_LOG_MODEL", "false")

    run_name = safe_get(wandb_cfg, "run_name") or getattr(trainer.args, "run_name", None)
    if run_name:
        trainer.args.run_name = str(run_name)
    return True
