from ..registry import DATASETS
from .collator import ContrastiveCollator
from .dataset import (
    ContrastiveJsonlDataset,
    RetrievalEvalDataset,
    resolve_view_counts,
)
from .lexical_negatives import DifficultySchedule, LexicalNegativeIndex
from .progress import SharedProgress
from .instructions import (
    BUILTIN_TASK_INSTRUCTIONS,
    load_task_instructions,
    resolve_instruction,
    resolve_instruction_verbose,
)
from .schema import Example, Record, parse_example, parse_record

__all__ = [
    "DATASETS",
    "ContrastiveCollator",
    "ContrastiveJsonlDataset",
    "RetrievalEvalDataset",
    "Example",
    "Record",
    "parse_example",
    "parse_record",
    "build_dataset",
    "build_collator",
    "view_config",
    "SharedProgress",
    "LexicalNegativeIndex",
    "DifficultySchedule",
    "resolve_view_counts",
    "BUILTIN_TASK_INSTRUCTIONS",
    "load_task_instructions",
    "resolve_instruction",
    "resolve_instruction_verbose",
]


def view_config(cfg) -> dict:
    """Resolve the multi-view / k settings of the `data` section into a canonical dict.

    The dataset, collator and inspection tools all read view settings through
    this single function, so they cannot silently diverge (e.g. the dataset
    using k=3 while the collator pads to 2).

    Returns:
        {"num_query_views": Nq, "num_positives": Np,
         "view_strategy": ..., "view_pad": ..., "max_view_groups": ...,
         "is_multiview": bool, "expands_views": bool}
    """
    from ..config import to_plain

    cfg = to_plain(cfg) or {}
    nq, npos = resolve_view_counts(
        cfg.get("multiview_k"),
        cfg.get("num_query_views"),
        cfg.get("num_positives"),
    )
    strategy = cfg.get("view_strategy", "first")
    return {
        "num_query_views": nq,
        "num_positives": npos,
        "view_strategy": strategy,
        "view_pad": cfg.get("view_pad", "mask"),
        "max_view_groups": int(cfg.get("max_view_groups", -1)),
        "is_multiview": nq != 1 or npos != 1,
        "expands_views": strategy in ("chunk", "shuffle_chunk"),
    }


def build_dataset(cfg, split: str = "train", progress=None):
    """`cfg` is the `data` section; `split` selects `train_path` or `eval_path`.

    `progress` is a SharedProgress used only for the train split (data-side curriculum).
    """
    from ..config import to_plain

    cfg = to_plain(cfg)
    path = cfg.get(f"{split}_path")
    if not path:
        return None
    ds_type = cfg.get("type", "jsonl") if split == "train" else cfg.get("eval_type", "jsonl_eval")
    views = view_config(cfg)
    kwargs = dict(
        num_negatives=int(cfg.get("num_negatives", 1)),
        num_query_views=views["num_query_views"],
        num_positives=views["num_positives"],
        view_strategy=views["view_strategy"],
        view_pad=views["view_pad"],
        max_view_groups=views["max_view_groups"],
        image_root=cfg.get("image_root", ""),
        task_instructions=cfg.get("task_instructions") or {},
        negative_strategy=cfg.get("negative_strategy", "shuffle"),
        negative_fill=cfg.get("negative_fill", "random_pool"),
        auto_ids=bool(cfg.get("auto_ids", True)),
        canonicalize_doc_ids=bool(cfg.get("canonicalize_doc_ids", True)),
        auto_positive_ids=bool(cfg.get("auto_positive_ids", True)),
        avoid_related_negatives=bool(cfg.get("avoid_related_negatives", True)),
        seed=int(cfg.get("seed", 42)),
        max_samples=int(cfg.get(f"max_{split}_samples", -1)),
        hard_negatives=cfg.get("hard_negatives") or {},
        multiview_negatives=bool(cfg.get("multiview_negatives", False)),
        progress=progress,
    )
    if split != "train":
        kwargs.pop("num_negatives", None)
        kwargs.pop("negative_fill", None)
        # No hard-negative mining for the eval set: it has no negative slots.
        kwargs.pop("hard_negatives", None)
        kwargs.pop("progress", None)
        # Evaluation does not gather across GPUs, so shapes need not be regular:
        # by default all views are used (no truncation, no expansion). Set
        # `eval_use_all_views: false` to evaluate with exactly the training view layout.
        if bool(cfg.get("eval_use_all_views", True)):
            kwargs["num_query_views"] = -1
            kwargs["num_positives"] = -1
            kwargs["view_strategy"] = "first"
            kwargs["view_pad"] = "mask"
    return DATASETS.build(ds_type, path, **kwargs)


def build_collator(cfg, encoder, collect_uids: bool = False) -> ContrastiveCollator:
    """`cfg` is the `data` section. Shares view parsing with `build_dataset`."""
    from ..config import to_plain

    cfg = to_plain(cfg)
    views = view_config(cfg)
    return ContrastiveCollator(
        encoder=encoder,
        num_negatives=int(cfg.get("num_negatives", 1)),
        collect_uids=collect_uids,
        num_query_views=views["num_query_views"],
        num_positives=views["num_positives"],
        dedupe=bool(cfg.get("dedupe_records", False)),
        # Example uids are needed only when one row expands into several groups (one extra all_gather_object).
        collect_example_uids=views["expands_views"],
        # Identity info for false-negative masking (a few string lists; negligible cost).
        collect_identity=bool(cfg.get("collect_identity", True)),
        # Must match the rule in ContrastiveJsonlDataset: with k > 1, negatives are always
        # k-view groups; a mismatch would make the doc count differ from B*G.
        multiview_negatives=(
            bool(cfg.get("multiview_negatives", False)) or int(views["num_positives"]) > 1
        ),
    )
