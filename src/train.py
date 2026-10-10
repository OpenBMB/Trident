#!/usr/bin/env python
"""Training entry point for Trident.

Single GPU:
    python src/train.py \
        --config configs/trident_qwen3vl.yaml

Multi-GPU:
    torchrun --nproc_per_node 8 \
        src/train.py \
        --config configs/trident_qwen3vl.yaml \
        --set train.per_device_train_batch_size=8 loss.temperature=0.02
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from typing import Dict  # noqa: F401

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    ),
)

from transformers import TrainingArguments  # noqa: E402

from mmemb.config import dump_config, load_config, to_plain  # noqa: E402
from mmemb.data import SharedProgress, build_collator, build_dataset, view_config  # noqa: E402
from mmemb.engine import (  # noqa: E402
    ContrastiveWrapper,
    DataCurriculumCallback,
    EmbeddingTrainer,
    EpochSeedCallback,
)
from mmemb.losses import build_loss  # noqa: E402
from mmemb.losses.spec import (  # noqa: E402
    canonicalize,
    check_consistency,
    describe,
    identity_requirements,
)
from mmemb.models import build_encoder  # noqa: E402
from mmemb.utils.dist import is_main_process  # noqa: E402
from mmemb.utils.env import TRAINING_ARG_ALIASES, describe_env  # noqa: E402
from mmemb.utils.misc import get_logger, set_seed  # noqa: E402
from mmemb.engine.monitor import setup_monitoring


logger = get_logger("train")


# Arguments that are always enforced by the framework.
#
# Any value set for these keys in the YAML config is overridden here.
FORCED_TRAINING_ARGS = {
    # Batches are custom nested structures; the Trainer must not drop fields.
    "remove_unused_columns": False,

    # Cross-GPU negative gathering requires identical batch sizes on every rank.
    "dataloader_drop_last": True,

    # The Trainer's default `labels` field is not used.
    "label_names": [],

    # Disable the Trainer's built-in per-epoch / per-step evaluation.
    #
    # The final evaluation is triggered once by `trainer.evaluate()` after
    # `train()` finishes, so retrieval evaluation runs exactly once per run.
    "eval_strategy": "no",
}


def build_training_args(train_cfg: dict) -> TrainingArguments:
    """Build Hugging Face `TrainingArguments` from the `train` config section."""

    cfg = dict(to_plain(train_cfg))

    # Forced arguments override user-provided values.
    cfg.update(FORCED_TRAINING_ARGS)

    valid_fields = {
        field.name
        for field in dataclasses.fields(TrainingArguments)
    }

    # Some arguments were renamed across transformers versions (e.g.
    # `evaluation_strategy` -> `eval_strategy` in 4.41). Map known aliases
    # first; only keys that cannot be mapped are reported as unsupported.
    filtered_cfg = {}
    renamed = []
    unknown = []

    for key, value in cfg.items():
        if key in valid_fields:
            filtered_cfg[key] = value
            continue

        alias = next(
            (a for a in TRAINING_ARG_ALIASES.get(key, []) if a in valid_fields),
            None,
        )
        if alias is not None:
            filtered_cfg[alias] = value
            renamed.append(f"{key}->{alias}")
        else:
            unknown.append(key)

    if renamed:
        logger.info("Renamed train arguments for the installed transformers version: %s", ", ".join(renamed))

    if unknown:
        logger.warning(
            "Ignoring train arguments not supported by the installed transformers version: %s",
            sorted(unknown),
        )

    return TrainingArguments(**filtered_cfg)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        help="Path to the training config (YAML).",
    )

    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        help=(
            "Config overrides, e.g. "
            "loss.temperature=0.02 "
            "train.per_device_train_batch_size=8"
        ),
    )

    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint directory to resume from, or `true` / `auto` to pick the latest checkpoint.",
    )

    cli_args = parser.parse_args()

    # ------------------------------------------------------------ config
    cfg = load_config(
        cli_args.config,
        cli_args.set,
    )

    training_args = build_training_args(cfg.train)

    set_seed(
        int(cfg.train.get("seed", 42))
    )

    if is_main_process():
        # Log the runtime environment first so mismatched setups are easy to spot.

        logger.info("Environment:\n%s", describe_env())
        logger.info("Config:\n%s", cfg)

    # Multi-view configuration, shared by the dataset, collator and loss.
    views = view_config(cfg.data)

    # Normalize the loss config: legacy keys are translated into
    # `loss.modules.*`, so every consumer reads a single canonical layout.
    loss_cfg, loss_warns = canonicalize(cfg.loss)
    for w in loss_warns + check_consistency(loss_cfg, views):
        logger.warning("%s", w)

    # ------------------------------------------------------ model & loss
    encoder = build_encoder(cfg.model)
    loss_fn = build_loss(loss_cfg)

    engine_cfg = cfg.get("engine") or {}

    # Whether the data pipeline must collect identity information is decided
    # solely by the loss's `false_negative` module.
    need = identity_requirements(loss_cfg)
    need_uids, need_identity = need["collect_uids"], need["collect_identity"]

    if is_main_process() and views["is_multiview"]:
        logger.info(
            "Multi-view training: block size %dx%d (num_query_views x num_positives), "
            "view_strategy=%s; all views of a sample will %sbe used",
            views["num_query_views"],
            views["num_positives"],
            views["view_strategy"],
            "" if views["expands_views"] else "(possibly not) ",
        )

    if is_main_process():
        logger.info("%s", describe(loss_cfg))

    model = ContrastiveWrapper(
        encoder=encoder,
        loss_fn=loss_fn,
        cross_device=engine_cfg.get(
            "cross_device",
            "grad",
        ),
        collect_uids=need_uids,
        # When one row is split into several view groups, sibling groups must
        # be identifiable inside the loss.
        collect_example_uids=views["expands_views"],
        collect_identity=need_identity,
    )

    # ------------------------------------------------------------ data
    # Progress carrier for the data-side curriculum (shared memory: written by
    # the main process every step, read by dataloader workers).
    hn_cfg = cfg.data.get("hard_negatives") or {}
    progress = SharedProgress() if hn_cfg.get("enable") else None

    train_ds = build_dataset(
        cfg.data,
        "train",
        progress=progress,
    )

    eval_ds = build_dataset(
        cfg.data,
        "eval",
    )

    if train_ds is None:
        raise ValueError("data.train_path is not set")

    # Multi-view / multi-positive block size comes from `data.multiview_k`
    # (or `num_query_views` / `num_positives`); default is 1 (single view).
    cfg.data["collect_identity"] = need_identity
    collator = build_collator(
        cfg.data,
        encoder=encoder,
        collect_uids=need_uids,
    )

    # ---------------------------------------------------------- trainer
    trainer = EmbeddingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        eval_k_list=engine_cfg.get(
            "eval_k_list",
            [1, 5, 10],
        ),
        eval_dim=engine_cfg.get("eval_dim"),
        callbacks=[
            EpochSeedCallback(train_ds),
        ],
    )

    # Optional data-side curriculum (progressively harder lexical negatives).
    data_cb = DataCurriculumCallback(progress, total_steps=hn_cfg.get("total_steps"))
    if data_cb.enabled:
        trainer.add_callback(data_cb)

    # Only rank 0 writes the resolved config.
    if is_main_process():
        os.makedirs(
            training_args.output_dir,
            exist_ok=True,
        )

        cfg["loss"] = loss_cfg   # store the canonical loss config for reproducibility
        dump_config(
            cfg,
            os.path.join(
                training_args.output_dir,
                "mmemb_run_config.yaml",
            ),
        )

    # ----------------------------------------------------------- resume
    resume = cli_args.resume

    if isinstance(resume, str):
        if resume.lower() in ("true", "auto"):
            resume = True

    # ------------------------------------------------------------ train
    setup_monitoring(trainer, cfg)
    train_result = trainer.train(
        resume_from_checkpoint=resume,
    )

    # The Trainer ensures only the main process writes logs and files.
    trainer.log_metrics(
        "train",
        train_result.metrics,
    )

    trainer.save_metrics(
        "train",
        train_result.metrics,
    )

    trainer.save_state()

    # ------------------------------------------------------- final eval
    if eval_ds is not None:
        # `evaluate` must be called on every rank.
        final_metrics = trainer.evaluate()
        # log_metrics / save_metrics only write on the main process.
        trainer.log_metrics("eval", final_metrics)
        trainer.save_metrics("eval", final_metrics)
        if trainer.is_world_process_zero():
            logger.info("Final eval metrics: %s", final_metrics)

    # ------------------------------------------------------- final save
    final_dir = os.path.join(
        training_args.output_dir,
        "final",
    )

    # `save_model` respects `should_save`; normally only rank 0 writes files.
    trainer.save_model(final_dir)

    if trainer.is_world_process_zero():
        logger.info(
            "Final model saved to %s",
            final_dir,
        )


if __name__ == "__main__":
    main()
