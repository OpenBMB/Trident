#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
embed_jsonl_unified_multigpu.py

Unified multi-model, multi-GPU embedding script.

This file only contains model-agnostic logic: JSONL record parsing and
validation, sharding by rank, a two-stage CPU/GPU pipeline (build_inputs_cpu in
a background thread pool, compute_from_inputs on the GPU in the main thread),
failure retries, shard saving and merging, multi-job execution (--jobs_json:
the model is loaded once and all datasets are processed in sequence), and the
CLI. Adding a model only requires a new file under ../embedding_backends/; this
file needs no changes.

See build_argument_parser at the bottom for all options, or
scripts/run_all_datasets.sh for typical usage.
"""

import abc  # noqa: F401  (kept for EmbeddingBackend subtype annotations)
import argparse
import gc
import json
import os
import shutil
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

# Make `src/` importable (embedding_backends / mmemb) when the script is run directly.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from embedding_backends import BACKEND_REGISTRY  # noqa: E402
from embedding_backends.base import EmbeddingBackend
from embedding_backends.common import (
    ensure_tokenizers_thread_safety,
    ensure_transformers_special_tokens_compat,
    is_remote_path,
)

IMAGE_PLACEHOLDER = "<image>"


# =============================================================================
# field normalization (model-agnostic, shared by all backends)
# =============================================================================

def normalize_text_field(text: Any) -> Optional[str]:
    if text is None:
        return None

    if isinstance(text, str):
        text = text.strip()
        return text if text else None

    if isinstance(text, (int, float, bool)):
        return str(text)

    return None


def normalize_image_field(image: Any) -> list[str]:
    """
    Always return a list of image paths.

        None            -> []
        "a.jpg"         -> ["a.jpg"]
        ["a.jpg", ...]  -> ["a.jpg", ...]
    """
    if image is None:
        return []

    if isinstance(image, str):
        image = image.strip()
        return [image] if image else []

    if isinstance(image, (list, tuple)):
        paths = []

        for item in image:
            if isinstance(item, str):
                item = item.strip()
                if item:
                    paths.append(item)

        return paths

    return []


def resolve_image_paths(
    paths: list[str],
    image_root: Optional[str],
) -> list[str]:
    """Resolve relative paths to absolute paths. Remote URLs and absolute paths are returned unchanged."""
    resolved = []

    for path in paths:
        if is_remote_path(path) or os.path.isabs(path):
            resolved.append(path)
            continue

        if image_root is None:
            resolved.append(path)
        else:
            resolved.append(
                os.path.normpath(os.path.join(image_root, path))
            )

    return resolved


def check_image_paths(paths: list[str]):
    """Raise immediately if a local path is missing; handled by the failure-recording logic."""
    for path in paths:
        if is_remote_path(path):
            continue

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Image file does not exist: {path}"
            )


def build_model_text(
    text: Optional[str],
    image_paths: list[str],
    placeholder_mode: str,
) -> tuple[Optional[str], Optional[str]]:
    """
    Handle "<image>" placeholders in the text and return (model_text, error).

    A non-empty error means the record is inconsistent and must fail (no silent truncation / guessing).
    """
    if text is None:
        return None, None

    if IMAGE_PLACEHOLDER not in text:
        return text, None

    segments = text.split(IMAGE_PLACEHOLDER)
    num_placeholders = len(segments) - 1
    num_images = len(image_paths)

    if num_placeholders != num_images:
        return None, (
            f"Text contains {num_placeholders} '<image>' placeholder(s) "
            f"but {num_images} image path(s) were provided; "
            f"counts must match exactly."
        )

    if placeholder_mode == "keep":
        return text, None

    cleaned = " ".join(
        segment.strip()
        for segment in segments
        if segment.strip()
    )

    return (cleaned or None), None


# =============================================================================
# record parsing
# =============================================================================

def parse_record(
    obj: dict,
    idx: int,
    line_number: int,
    image_root: Optional[str],
    id_field: str,
    role: str,
    instruction: Optional[str],
    placeholder_mode: str,
    backend: EmbeddingBackend,
) -> dict:
    """Convert one JSONL record into a task."""
    record_id = obj.get(id_field)

    if record_id is not None:
        record_id = str(record_id)

    raw_text = normalize_text_field(obj.get("text"))
    raw_image_paths = normalize_image_field(obj.get("image"))

    image_paths = resolve_image_paths(
        paths=raw_image_paths,
        image_root=image_root,
    )

    model_text, error = build_model_text(
        text=raw_text,
        image_paths=image_paths,
        placeholder_mode=placeholder_mode,
    )

    truncated = False
    if len(image_paths) > backend.max_images_per_record:
        image_paths = image_paths[: backend.max_images_per_record]
        truncated = True

    usable = error is None and (
        model_text is not None or bool(image_paths)
    )

    if error is None and not usable:
        error = "Both text and image fields are missing or empty."

    if error is None and usable:
        backend_error = backend.validate_item(
            text=model_text,
            images=image_paths,
            role=role,
        )
        if backend_error is not None:
            error = backend_error
            usable = False

    return {
        "_idx": idx,
        "_line_number": line_number,
        "_id": record_id,
        "_raw_text": raw_text,
        "_text": model_text,
        "_image_paths": image_paths,
        "_raw_image_paths": raw_image_paths,
        "_role": role,
        "_instruction": instruction,
        "_usable": usable,
        "_error": error,
        "_metadata": {
            "id": record_id,
            "line_number": line_number,
            "role": role,
            "has_text": model_text is not None,
            "num_images": len(image_paths),
            "images_truncated": truncated,
        },
    }


def scan_input(
    input_jsonl: str,
    id_field: str,
) -> tuple[int, list[Optional[str]]]:
    """Pre-scan: return the number of records and the ordered id list, validating JSON and id uniqueness."""
    ids: list[Optional[str]] = []
    seen: dict[str, int] = {}
    duplicate_count = 0

    with open(input_jsonl, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at physical line {line_number}: {exc}"
                ) from exc

            if not isinstance(obj, dict):
                raise TypeError(
                    f"Record at physical line {line_number} "
                    f"must be a JSON object."
                )

            record_id = obj.get(id_field)

            if record_id is not None:
                record_id = str(record_id)

                if record_id in seen:
                    duplicate_count += 1
                else:
                    seen[record_id] = line_number

            ids.append(record_id)

    if duplicate_count:
        print(
            f"[warn] found {duplicate_count} duplicated ids in "
            f"{input_jsonl}; embeddings remain aligned by line order.",
            flush=True,
        )

    return len(ids), ids


def load_rank_tasks(
    input_jsonl: str,
    rank: int,
    world_size: int,
    image_root: Optional[str],
    id_field: str,
    role: str,
    instruction: Optional[str],
    placeholder_mode: str,
    backend: EmbeddingBackend,
) -> list[dict]:
    """Assign tasks to GPUs by idx % world_size == rank; the original order is restored by idx when merging."""
    rank_tasks: list[dict] = []
    idx = 0

    with open(input_jsonl, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            obj = json.loads(line)

            if idx % world_size == rank:
                rank_tasks.append(
                    parse_record(
                        obj=obj,
                        idx=idx,
                        line_number=line_number,
                        image_root=image_root,
                        id_field=id_field,
                        role=role,
                        instruction=instruction,
                        placeholder_mode=placeholder_mode,
                        backend=backend,
                    )
                )

            idx += 1

    return rank_tasks


# =============================================================================
# batch inference (backend-agnostic)
# =============================================================================

def make_failed_record(
    task: dict,
    error: Exception | str,
    batch_error: Optional[Exception | str] = None,
) -> dict:
    item = {
        "idx": int(task["_idx"]),
        "id": task.get("_id"),
        "line_number": int(task["_line_number"]),
        "image": task.get("_raw_image_paths"),
        "error": str(error),
    }

    if batch_error is not None:
        item["batch_error"] = str(batch_error)

    return item


def prepare_batch(
    backend: EmbeddingBackend,
    batch_tasks: list[dict],
    role: str,
) -> tuple[list[dict], list[dict], Optional[Any], Optional[Exception], list[dict]]:
    """CPU stage: validate records + backend.build_inputs_cpu; never touches the GPU."""
    valid_tasks: list[dict] = []
    valid_inputs: list[dict] = []
    failed_records: list[dict] = []

    for task in batch_tasks:
        if not task.get("_usable"):
            failed_records.append(
                make_failed_record(
                    task,
                    task.get("_error") or "Record is not usable.",
                )
            )
            continue

        try:
            check_image_paths(task["_image_paths"])
        except Exception as exc:
            failed_records.append(make_failed_record(task, exc))
            continue

        valid_tasks.append(task)
        valid_inputs.append(
            {
                "text": task.get("_text"),
                "images": task.get("_image_paths"),
                "instruction": task.get("_instruction"),
            }
        )

    prepared: Optional[Any] = None
    build_error: Optional[Exception] = None

    if valid_inputs:
        try:
            # Share backend.cpu_lock with backend.process() (the per-item retry path) to
            # serialize all tokenizer access; otherwise the prefetch threads and the
            # main-thread retries could call the same tokenizer instance concurrently and
            # crash with `Already borrowed` (HuggingFace fast tokenizers are not
            # thread-safe). See EmbeddingBackend.__init__ in base.py.
            with backend.cpu_lock:
                prepared = backend.build_inputs_cpu(valid_inputs)
        except Exception as exc:  # noqa: BLE001
            build_error = exc

    return valid_tasks, valid_inputs, prepared, build_error, failed_records


def run_batch_on_gpu(
    backend: EmbeddingBackend,
    valid_tasks: list[dict],
    valid_inputs: list[dict],
    prepared: Optional[Any],
    build_error: Optional[Exception],
    failed_records: list[dict],
    role: str,
) -> tuple[
    list[int],
    list[np.ndarray],
    list[int],
    list[int],
    list[int],
    list[dict],
    list[dict],
]:
    """GPU stage: run the forward on the output of prepare_batch. Must be called from the main thread."""
    failed_records = list(failed_records)

    if not valid_inputs:
        return [], [], [], [], [], [], failed_records

    success_indices: list[int] = []
    embeddings: list[np.ndarray] = []
    total_counts: list[int] = []
    text_counts: list[int] = []
    image_counts: list[int] = []
    success_metadata: list[dict] = []

    if prepared is not None:
        try:
            batch_embeddings, token_infos = backend.compute_from_inputs(
                prepared, valid_inputs, role
            )

            for offset, task in enumerate(valid_tasks):
                success_indices.append(int(task["_idx"]))
                embeddings.append(batch_embeddings[offset])
                total_counts.append(token_infos[offset][0])
                text_counts.append(token_infos[offset][1])
                image_counts.append(token_infos[offset][2])
                success_metadata.append(dict(task["_metadata"]))

            return (
                success_indices,
                embeddings,
                total_counts,
                text_counts,
                image_counts,
                success_metadata,
                failed_records,
            )

        except Exception as batch_exc:  # noqa: BLE001
            print(
                f"[warn] batch inference failed (role={role}); "
                f"retrying individually: {batch_exc}",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            build_error = batch_exc
    else:
        print(
            f"[warn] batch build_inputs failed (role={role}); "
            f"retrying individually: {build_error}",
            flush=True,
        )

    for task, model_input in zip(valid_tasks, valid_inputs):
        try:
            single_embeddings, single_tokens = backend.process([model_input], role=role)

            success_indices.append(int(task["_idx"]))
            embeddings.append(single_embeddings[0])

            total_counts.append(single_tokens[0][0])
            text_counts.append(single_tokens[0][1])
            image_counts.append(single_tokens[0][2])

            success_metadata.append(dict(task["_metadata"]))

        except Exception as exc:
            failed_records.append(
                make_failed_record(
                    task=task,
                    error=exc,
                    batch_error=build_error,
                )
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return (
        success_indices,
        embeddings,
        total_counts,
        text_counts,
        image_counts,
        success_metadata,
        failed_records,
    )


# =============================================================================
# shard saving / merging
# =============================================================================

def save_rank_shard(
    shard_path: str,
    success_indices: list[int],
    embeddings: list[np.ndarray],
    total_token_counts: list[int],
    text_token_counts: list[int],
    image_token_counts: list[int],
    success_metadata: list[dict],
    failed_records: list[dict],
    save_dtype: np.dtype,
    matryoshka_pack_dims: Optional[list[int]] = None,
):
    if embeddings:
        embeddings_array = np.stack(embeddings, axis=0).astype(save_dtype)
    else:
        embeddings_array = np.empty((0, 0), dtype=save_dtype)

    metadata_json = np.asarray(
        [json.dumps(item, ensure_ascii=False) for item in success_metadata],
        dtype=np.str_,
    )

    failed_json = np.asarray(
        [json.dumps(item, ensure_ascii=False) for item in failed_records],
        dtype=np.str_,
    )

    np.savez(
        shard_path,
        indices=np.asarray(success_indices, dtype=np.int64),
        embeddings=embeddings_array,
        total_token_counts=np.asarray(total_token_counts, dtype=np.int64),
        text_token_counts=np.asarray(text_token_counts, dtype=np.int64),
        image_token_counts=np.asarray(image_token_counts, dtype=np.int64),
        metadata_json=metadata_json,
        failed_json=failed_json,
        # With --matryoshka_dims each row is a concatenation of segments; record the segment
        # dims so merge_shards can unpack them. Empty array otherwise.
        matryoshka_pack_dims=np.asarray(matryoshka_pack_dims or [], dtype=np.int64),
    )


MATRYOSHKA_SUBDIR = "matryoshka"
MATRYOSHKA_MANIFEST = "dims.json"
# Files of the top-level directory copied verbatim into every matryoshka/dim_<d>/
# (the evaluation EmbeddingSet only needs embeddings.npy / ids.txt /
# token_counts.jsonl; the rest is copied for debugging).
MATRYOSHKA_SIDECAR_FILES = (
    "ids.txt",
    "token_counts.jsonl",
    "embedding_metadata.jsonl",
    "failed_records.jsonl",
    "instructions_used.json",
)


def matryoshka_dim_dir(output_dir: str, dim: int) -> str:
    return os.path.join(output_dir, MATRYOSHKA_SUBDIR, f"dim_{int(dim)}")


def matryoshka_outputs_complete(output_dir: str, requested_dims: list[int]) -> bool:
    """For --skip_existing: whether a previous run completed with the same list of dims."""
    manifest_path = os.path.join(output_dir, MATRYOSHKA_SUBDIR, MATRYOSHKA_MANIFEST)

    if not os.path.isfile(manifest_path):
        return False

    try:
        with open(manifest_path, "r", encoding="utf-8") as file:
            manifest = json.load(file)
    except (OSError, json.JSONDecodeError):
        return False

    if sorted(manifest.get("requested_dims") or []) != sorted(requested_dims):
        return False

    return all(
        os.path.isfile(os.path.join(matryoshka_dim_dir(output_dir, d), "embeddings.npy"))
        for d in manifest.get("dims") or []
    )


def _write_embeddings_npy(
    path: str,
    shard_data: list[dict],
    num_records: int,
    col_start: int,
    col_end: int,
    save_dtype: np.dtype,
) -> None:
    """Write embeddings[:, col_start:col_end] of all shards into one .npy, in original row order."""
    final_embeddings = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=save_dtype,
        shape=(num_records, col_end - col_start),
    )

    final_embeddings[:] = 0

    for data in shard_data:
        indices = data["indices"]

        if len(indices) == 0:
            continue

        final_embeddings[indices] = data["embeddings"][:, col_start:col_end]

    final_embeddings.flush()
    del final_embeddings


def merge_shards(
    shard_dir: str,
    world_size: int,
    num_records: int,
    ordered_ids: list[Optional[str]],
    output_dir: str,
    save_dtype: np.dtype,
    matryoshka_dims: Optional[list[int]] = None,
):
    shard_data: list[dict] = []
    packed_dim: Optional[int] = None
    pack_dims: Optional[list[int]] = None
    all_failed_records: list[dict] = []

    for rank in range(world_size):
        shard_path = os.path.join(shard_dir, f"rank_{rank:03d}.npz")

        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Missing shard file: {shard_path}")

        with np.load(shard_path, allow_pickle=False) as data:
            indices = data["indices"]
            embeddings = data["embeddings"]

            if embeddings.ndim == 2 and embeddings.shape[0] > 0:
                current_dim = embeddings.shape[1]

                if packed_dim is None:
                    packed_dim = current_dim
                elif packed_dim != current_dim:
                    raise ValueError(
                        f"Inconsistent embedding dimensions: "
                        f"{packed_dim} and {current_dim}."
                    )

            if "matryoshka_pack_dims" in data.files:
                current_pack = [int(d) for d in data["matryoshka_pack_dims"].tolist()]
                if current_pack:
                    if pack_dims is None:
                        pack_dims = current_pack
                    elif pack_dims != current_pack:
                        raise ValueError(
                            f"inconsistent matryoshka dims across ranks: {pack_dims} vs {current_pack}"
                        )

            for item in data["failed_json"].tolist():
                all_failed_records.append(json.loads(str(item)))

            shard_data.append(
                {
                    "indices": indices.copy(),
                    "embeddings": embeddings.copy(),
                    "total_token_counts": data["total_token_counts"].copy(),
                    "text_token_counts": data["text_token_counts"].copy(),
                    "image_token_counts": data["image_token_counts"].copy(),
                    "metadata_json": data["metadata_json"].copy(),
                }
            )

    if packed_dim is None:
        raise RuntimeError(
            "All records failed. Unable to determine embedding dimension."
        )

    # ---------------------------------------------------------- unpacking layout
    # Without matryoshka: one segment (the whole row).
    # With matryoshka: each row is [d_1 | d_2 | ... | d_full]; d_full goes to the top-level
    # directory (identical to a run without matryoshka) and each requested dim to matryoshka/dim_<d>/.
    if pack_dims:
        if sum(pack_dims) != packed_dim:
            raise ValueError(
                f"matryoshka packed width {packed_dim} does not match the sum of segment dims "
                f"{sum(pack_dims)} ({pack_dims})"
            )
        offsets = np.cumsum([0] + pack_dims).tolist()
        segments = {
            dim: (offsets[i], offsets[i + 1]) for i, dim in enumerate(pack_dims)
        }
        embedding_dim = pack_dims[-1]
        full_start, full_end = segments[embedding_dim]
    else:
        segments = {}
        embedding_dim = packed_dim
        full_start, full_end = 0, packed_dim

    embeddings_path = os.path.join(output_dir, "embeddings.npy")
    _write_embeddings_npy(
        embeddings_path, shard_data, num_records, full_start, full_end, save_dtype,
    )

    total_token_counts = np.zeros(num_records, dtype=np.int64)
    text_token_counts = np.zeros(num_records, dtype=np.int64)
    image_token_counts = np.zeros(num_records, dtype=np.int64)
    success_mask = np.zeros(num_records, dtype=bool)

    ordered_metadata: list[Optional[dict]] = [None] * num_records

    for data in shard_data:
        indices = data["indices"]

        if len(indices) == 0:
            continue

        total_token_counts[indices] = data["total_token_counts"]
        text_token_counts[indices] = data["text_token_counts"]
        image_token_counts[indices] = data["image_token_counts"]
        success_mask[indices] = True

        for idx, metadata_str in zip(
            indices.tolist(),
            data["metadata_json"].tolist(),
        ):
            ordered_metadata[idx] = json.loads(str(metadata_str))

    failed_by_idx = {int(item["idx"]): item for item in all_failed_records}

    token_counts_path = os.path.join(output_dir, "token_counts.jsonl")
    metadata_path = os.path.join(output_dir, "embedding_metadata.jsonl")
    ids_path = os.path.join(output_dir, "ids.txt")

    with open(
        token_counts_path, "w", encoding="utf-8",
    ) as token_file, open(
        metadata_path, "w", encoding="utf-8",
    ) as metadata_file, open(
        ids_path, "w", encoding="utf-8",
    ) as ids_file:

        for idx in range(num_records):
            record_id = ordered_ids[idx]

            ids_file.write(("" if record_id is None else record_id) + "\n")

            token_item = {
                "idx": idx,
                "id": record_id,
                "token_count": int(total_token_counts[idx]),
                "text_token_count": int(text_token_counts[idx]),
                "image_token_count": int(image_token_counts[idx]),
                "success": bool(success_mask[idx]),
            }

            token_file.write(json.dumps(token_item, ensure_ascii=False) + "\n")

            if ordered_metadata[idx] is not None:
                metadata_item = {
                    "idx": idx,
                    **ordered_metadata[idx],
                    "success": True,
                }
            else:
                failed = failed_by_idx.get(idx, {})

                metadata_item = {
                    "idx": idx,
                    "id": failed.get("id", record_id),
                    "line_number": failed.get("line_number"),
                    "has_text": None,
                    "num_images": None,
                    "success": False,
                    "error": failed.get("error", "Unknown processing failure."),
                }

            metadata_file.write(json.dumps(metadata_item, ensure_ascii=False) + "\n")

    failed_records_path = os.path.join(output_dir, "failed_records.jsonl")

    all_failed_records.sort(key=lambda item: item.get("idx", -1))

    with open(failed_records_path, "w", encoding="utf-8") as file:
        for item in all_failed_records:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    success_count = int(success_mask.sum())
    failed_count = num_records - success_count

    print(f"[merge] embeddings: {embeddings_path}")
    print(f"[merge] shape: ({num_records}, {embedding_dim})")
    print(f"[merge] dtype: {save_dtype}")
    print(f"[merge] metadata: {metadata_path}")
    print(f"[merge] ids: {ids_path}")
    print(f"[merge] token counts: {token_counts_path}")
    print(f"[merge] failed records: {failed_records_path}")
    print(f"[merge] success={success_count}, failed={failed_count}")

    # ---------------------------------------------------------- matryoshka
    # Always remove old matryoshka subdirectories: the top-level embeddings.npy was just
    # overwritten, and stale truncated results would no longer match it.
    matryoshka_root = os.path.join(output_dir, MATRYOSHKA_SUBDIR)
    if os.path.isdir(matryoshka_root):
        shutil.rmtree(matryoshka_root)

    if not pack_dims:
        if matryoshka_dims:
            raise RuntimeError(
                f"--matryoshka_dims={matryoshka_dims} was requested, but the shards contain no "
                f"packing information (did the backend call setup_matryoshka?)"
            )
        return

    requested = sorted({int(d) for d in (matryoshka_dims or [])})
    # The full dim is only packed for the top-level directory; no subdirectory unless requested.
    out_dims = [d for d in pack_dims if d in requested]

    for dim in out_dims:
        dim_dir = matryoshka_dim_dir(output_dir, dim)
        os.makedirs(dim_dir, exist_ok=True)

        col_start, col_end = segments[dim]
        _write_embeddings_npy(
            os.path.join(dim_dir, "embeddings.npy"),
            shard_data, num_records, col_start, col_end, save_dtype,
        )

        for name in MATRYOSHKA_SIDECAR_FILES:
            src = os.path.join(output_dir, name)
            if os.path.isfile(src):
                shutil.copyfile(src, os.path.join(dim_dir, name))

    with open(
        os.path.join(matryoshka_root, MATRYOSHKA_MANIFEST), "w", encoding="utf-8",
    ) as file:
        json.dump(
            {
                "requested_dims": requested,
                "dims": out_dims,
                "skipped_dims": [d for d in requested if d not in out_dims],
                "full_dim": embedding_dim,
                "dirs": {
                    str(d): os.path.relpath(matryoshka_dim_dir(output_dir, d), output_dir)
                    for d in out_dims
                },
            },
            file, ensure_ascii=False, indent=2,
        )

    print(f"[merge] matryoshka dims: {out_dims} -> {matryoshka_root}/dim_<d>/")


# =============================================================================
# single-GPU worker process
# =============================================================================

def limit_cpu_threads(world_size: int) -> int:
    cpu_count = os.cpu_count() or world_size
    per_rank = max(1, cpu_count // world_size)

    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = str(per_rank)

    torch.set_num_threads(per_rank)

    return per_rank


def run_job_batches(
    rank: int,
    world_size: int,
    physical_gpu_id: int,
    backend: EmbeddingBackend,
    job: dict,
    args_dict: dict,
) -> tuple[
    list[int], list[np.ndarray], list[int], list[int], list[int], list[dict], list[dict]
]:
    """
    Process the part of one job (the query or doc file of one dataset) assigned to this rank.

    Instead of double buffering (one prefetched batch, one background thread), a
    ThreadPoolExecutor(max_workers=prefetch_depth) with a pending queue of length
    prefetch_depth keeps prefetch_depth batches in preparation (each batch also loads
    its images in parallel with load_images_parallel), so the GPU main thread
    almost always has a ready batch.
    """
    role = job["role"]
    instruction = job["instruction"]

    rank_tasks = load_rank_tasks(
        input_jsonl=job["input_jsonl"],
        rank=rank,
        world_size=world_size,
        image_root=job["image_root"],
        id_field=job["id_field"],
        role=role,
        instruction=instruction,
        placeholder_mode=args_dict["image_placeholder"],
        backend=backend,
    )

    if args_dict["sort_by_length"]:
        rank_tasks.sort(
            key=lambda task: (
                len(task["_text"]) if task["_text"] else 0,
                len(task["_image_paths"]),
            )
        )

    print(
        f"[rank {rank}] job='{job['tag']}' assigned {len(rank_tasks)} records",
        flush=True,
    )

    if rank == 0 and args_dict.get("preview"):
        preview_n = int(args_dict["preview"])
        print("\n" + "=" * 70, flush=True)
        print(f"[job={job['tag']}] text actually seen by the model (first {preview_n} records)", flush=True)
        print("=" * 70, flush=True)
        shown = 0
        for task in rank_tasks:
            if shown >= preview_n:
                break
            if not task.get("_usable"):
                continue
            rendered = backend.render_prompt(
                task.get("_text"),
                task.get("_image_paths") or [],
                task.get("_instruction"),
                role,
            )
            print(
                f"[idx={task['_idx']}] id={task.get('_id')} "
                f"role={role} num_images={len(task['_image_paths'])}",
                flush=True,
            )
            print(f"    {rendered!r}", flush=True)
            shown += 1
        print("=" * 70 + "\n", flush=True)

    all_success_indices: list[int] = []
    all_embeddings: list[np.ndarray] = []
    all_total_token_counts: list[int] = []
    all_text_token_counts: list[int] = []
    all_image_token_counts: list[int] = []
    all_success_metadata: list[dict] = []
    all_failed_records: list[dict] = []

    batch_size = args_dict["batch_size"]
    prefetch_depth = max(1, int(args_dict.get("prefetch_depth", 3)))

    batches = [
        rank_tasks[start:start + batch_size]
        for start in range(0, len(rank_tasks), batch_size)
    ]

    progress = tqdm(
        total=len(rank_tasks),
        desc=f"GPU{physical_gpu_id}|{job['tag']}",
        unit="rec",
        position=rank,
        leave=True,
        dynamic_ncols=True,
    )

    if batches:
        executor = ThreadPoolExecutor(max_workers=prefetch_depth)
        pending: deque = deque()

        try:
            for i in range(min(prefetch_depth, len(batches))):
                pending.append(
                    executor.submit(prepare_batch, backend, batches[i], role)
                )

            for i, batch_tasks in enumerate(batches):
                valid_tasks, valid_inputs, prepared, build_error, failed_records = (
                    pending.popleft().result()
                )

                next_idx = i + prefetch_depth
                if next_idx < len(batches):
                    pending.append(
                        executor.submit(
                            prepare_batch, backend, batches[next_idx], role
                        )
                    )

                (
                    success_indices,
                    batch_embeddings,
                    total_counts,
                    text_counts,
                    image_counts,
                    success_metadata,
                    batch_failed_records,
                ) = run_batch_on_gpu(
                    backend=backend,
                    valid_tasks=valid_tasks,
                    valid_inputs=valid_inputs,
                    prepared=prepared,
                    build_error=build_error,
                    failed_records=failed_records,
                    role=role,
                )

                all_success_indices.extend(success_indices)
                all_embeddings.extend(batch_embeddings)
                all_total_token_counts.extend(total_counts)
                all_text_token_counts.extend(text_counts)
                all_image_token_counts.extend(image_counts)
                all_success_metadata.extend(success_metadata)
                all_failed_records.extend(batch_failed_records)

                progress.update(len(batch_tasks))
        finally:
            executor.shutdown(wait=True)

    progress.close()

    return (
        all_success_indices,
        all_embeddings,
        all_total_token_counts,
        all_text_token_counts,
        all_image_token_counts,
        all_success_metadata,
        all_failed_records,
    )


def worker_main(
    rank: int,
    world_size: int,
    gpu_ids: list[int],
    args_dict: dict,
):
    physical_gpu_id = gpu_ids[rank]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    repo_root = args_dict.get("repo_root")
    if repo_root and repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA unavailable in rank {rank}, physical GPU={physical_gpu_id}."
        )

    torch.cuda.set_device(0)

    # This patch must be applied before any backend.load() (which calls
    # AutoProcessor/AutoTokenizer.from_pretrained); see common.py. It fixes
    # `AttributeError: 'list' object has no attribute 'keys'` for checkpoints whose
    # tokenizer_config.json stores extra_special_tokens in the v5 list format while
    # transformers v4.x is installed. Workers are separate mp.spawn processes, so every
    # rank applies the patch itself.
    ensure_transformers_special_tokens_compat()

    # Fix "Already borrowed": wrap the encode methods of the Rust tokenizer with a
    # process-wide lock, covering backend libraries that call the same tokenizer from
    # their own thread pools. Must also be called in every worker before backend.load();
    # see common.py.
    ensure_tokenizers_thread_safety()

    per_rank_threads = limit_cpu_threads(world_size)

    save_dtype = np.dtype(args_dict["save_dtype"])
    jobs = args_dict["jobs"]

    print(
        f"[rank {rank}/{world_size}] "
        f"physical_gpu={physical_gpu_id}, local_device=cuda:0, "
        f"cpu_threads={per_rank_threads}, model_type={args_dict['model_type']}, "
        f"jobs={len(jobs)}",
        flush=True,
    )

    backend_cls = BACKEND_REGISTRY[args_dict["model_type"]]
    backend = backend_cls(args_dict, device="cuda:0")
    if args_dict.get("clip_fusion_alpha") is not None:
        backend.clip_fusion_alpha = args_dict["clip_fusion_alpha"]

    # The model is loaded once here; the `for job in jobs` loop below reuses the same
    # backend already on the GPU instead of reloading it for every dataset.
    backend.load()

    for job in jobs:
        # Some backends (trident_qwen3vl / jina_v5_omni) read self.args_dict["role"] in
        # build_inputs_cpu; update it in place so they see the role of the current job.
        args_dict["role"] = job["role"]

        shard_path = os.path.join(job["shard_dir"], f"rank_{rank:03d}.npz")

        (
            success_indices,
            embeddings,
            total_counts,
            text_counts,
            image_counts,
            success_metadata,
            failed_records,
        ) = run_job_batches(
            rank=rank,
            world_size=world_size,
            physical_gpu_id=physical_gpu_id,
            backend=backend,
            job=job,
            args_dict=args_dict,
        )

        save_rank_shard(
            shard_path=shard_path,
            success_indices=success_indices,
            embeddings=embeddings,
            total_token_counts=total_counts,
            text_token_counts=text_counts,
            image_token_counts=image_counts,
            success_metadata=success_metadata,
            failed_records=failed_records,
            save_dtype=save_dtype,
            matryoshka_pack_dims=backend.matryoshka_pack_dims,
        )

        print(
            f"[rank {rank}] job='{job['tag']}' finished: "
            f"success={len(success_indices)}, failed={len(failed_records)}, "
            f"shard={shard_path}",
            flush=True,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del backend
    gc.collect()
    torch.cuda.empty_cache()


# =============================================================================
# arguments
# =============================================================================

def parse_gpu_ids(gpus: Optional[str]) -> list[int]:
    if gpus is None or not gpus.strip():
        visible_count = torch.cuda.device_count()

        if visible_count <= 0:
            raise RuntimeError("No CUDA GPUs are visible.")

        return list(range(visible_count))

    gpu_ids = []

    for value in gpus.split(","):
        value = value.strip()

        if value:
            gpu_ids.append(int(value))

    if not gpu_ids:
        raise ValueError("No valid GPU ids were provided.")

    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"Duplicate GPU ids are not allowed: {gpu_ids}")

    return gpu_ids


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate embeddings from JSONL using multiple GPUs, with the "
            "model backend selected via --model_type. Supports either a "
            "single (--input_jsonl/--output_dir/--role) job, or a batch of "
            "jobs via --jobs_json so the model is loaded only once."
        )
    )

    parser.add_argument(
        "--model_type",
        required=True,
        choices=sorted(BACKEND_REGISTRY.keys()),
        help=(
            "Model backend to use (defined in the embedding_backends package). Choices: "
            f"{sorted(BACKEND_REGISTRY.keys())}"
        ),
    )

    parser.add_argument(
        "--jobs_json",
        default=None,
        help=(
            "Path to a JSON array where each element is a job: "
            '{"input_jsonl":..., "output_dir":..., "role":"query"/"doc", '
            '"tag":..., "image_root":..., "id_field":..., "instruction":...}'
            " (all fields except input_jsonl/output_dir/role are optional and fall back to the "
            "global arguments). With this option the model is loaded once per GPU and all jobs "
            "run in sequence; without it, a single job is defined by --input_jsonl/--output_dir/--role."
        ),
    )

    parser.add_argument(
        "--input_jsonl",
        "--testdata",
        dest="input_jsonl",
        default=None,
        help="Required in single-job mode. queries.jsonl or corpus.jsonl.",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Model path / HuggingFace (or ModelScope) repo id. For trident_qwen3vl / trident_jinaclip"
            " it must be a directory written by save_pretrained of this repository (with"
            " mmemb_model.json); other backends accept a repo id or a local snapshot directory."
        ),
    )

    parser.add_argument(
        "--base_model",
        default=None,
        help="[trident_qwen3vl / trident_jinaclip only] base weights, used when the checkpoint is a LoRA adapter",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        help="Required in single-job mode.",
    )

    parser.add_argument(
        "--image_root",
        default=None,
        help="Root directory of relative image paths; defaults to the directory of each job's JSONL file.",
    )

    parser.add_argument(
        "--id_field",
        default="id",
    )

    parser.add_argument(
        "--repo_root",
        default=None,
        help="[trident_qwen3vl / trident_jinaclip only] directory containing the mmemb package; defaults to the parent directory of this script (src/)",
    )

    parser.add_argument(
        "--role",
        default=None,
        choices=["query", "doc"],
        help="Required in single-job mode. Whether the file is encoded as queries or documents.",
    )

    parser.add_argument(
        "--instruction",
        default=None,
        help="Override the default instruction (normally unused; for ablations). Can also be "
             "overridden per job in --jobs_json",
    )

    parser.add_argument(
        "--no_instruction",
        action="store_true",
        help="Do not add any instruction (ablation control)",
    )

    parser.add_argument(
        "--image_placeholder",
        choices=["strip", "keep"],
        default="strip",
        help="How '<image>' placeholders in text are handled: strip = validate and remove (default), keep = keep as is",
    )

    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Rank 0 prints the first N 'texts actually seen by the model' of each job, to check prompts",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only validate inputs and print statistics; do not load the model or run inference",
    )

    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip a job if its output_dir/embeddings.npy already exists",
    )

    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="e.g. 0,1,2,3",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size per GPU. 4-8 is recommended for image-text models; CLIP-style models can use more",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
        help="inference precision",
    )

    parser.add_argument(
        "--save_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="storage precision of embeddings.npy",
    )

    parser.add_argument(
        "--max_pixels",
        type=int,
        default=None,
        help="[trident_qwen3vl / trident_jinaclip only] override the image pixel limit used in training; defaults to the checkpoint config",
    )

    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="Matryoshka truncation dim, e.g. 1024. Supported by: "
             "trident_qwen3vl / trident_jinaclip / jina_v5_omni / jina_clip_v2. "
             "Defaults to the full dim. Mutually exclusive with --matryoshka_dims",
    )

    parser.add_argument(
        "--matryoshka_dims",
        type=parse_matryoshka_dims,
        default=None,
        help="[trident_qwen3vl / trident_jinaclip / qwen3vl_official only] Matryoshka evaluation: "
             "one forward produces several truncation dims, e.g. 128,256,512,1024,2048 "
             "(the yaml style '[128, 256, 512]' is also accepted). The top-level embeddings.npy "
             "keeps the full dim; each truncation dim is written to <output_dir>/matryoshka/dim_<d>/; "
             "dims larger than the model's full dim are ignored.",
    )

    parser.add_argument(
        "--sort_by_length",
        action="store_true",
        help="Sort by length within each GPU before batching to reduce padding; output order is unchanged.",
    )

    parser.add_argument(
        "--no_merge_lora",
        action="store_true",
        help="[trident_qwen3vl / trident_jinaclip only] do not merge LoRA into the base model (merging is faster unless you need to swap adapters)",
    )

    parser.add_argument(
        "--keep_shards",
        action="store_true",
    )

    parser.add_argument(
        "--image_load_workers",
        type=int,
        default=8,
        help=(
            "Threads for parallel image loading within a batch (one pool of this size per rank). "
            "Increase (e.g. 16) for remote URLs / large images / large batches, decrease for "
            "small local images. Default 8."
        ),
    )

    parser.add_argument(
        "--prefetch_depth",
        type=int,
        default=3,
        help=(
            "Number of batches prepared ahead on the CPU. Larger values absorb CPU-side "
            "jitter (e.g. slow remote image downloads) at the cost of more memory; 2-4 is "
            "usually enough. Default 3."
        ),
    )

    parser.add_argument(
        "--jina_task",
        default=None,
        help="[jina_v5_omni] task name, default 'retrieval'.",
    )

    parser.add_argument(
        "--clip_fusion_alpha",
        type=float,
        default=None,
        help="[jina_v5_omni / unime_phi35v / jina_clip_v2 / ...] alpha of the CLIP-style "
             "late fusion (image weight), default 0.5.",
    )

    parser.add_argument(
        "--qwen3vl_official_script",
        default=None,
        help="[qwen3vl_official] path to qwen3_vl_embedding.py; by default imported from src/.",
    )

    parser.add_argument(
        "--attn_implementation",
        default=None,
        help="[qwen3vl_official / unime_phi35v and other backends supporting it] "
             "e.g. flash_attention_2, eager",
    )


    return parser


def parse_matryoshka_dims(text: str) -> list[int]:
    """'128,256' / '128 256' / '[128, 256]' -> [128, 256] (deduplicated, ascending)."""
    cleaned = str(text).strip().strip("[]()")
    parts = [p for p in cleaned.replace(",", " ").split() if p]

    try:
        dims = sorted({int(p) for p in parts})
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--matryoshka_dims must be a list of integers, got {text!r}"
        ) from exc

    if not dims:
        raise argparse.ArgumentTypeError("--matryoshka_dims must not be empty")
    if dims[0] <= 0:
        raise argparse.ArgumentTypeError(f"--matryoshka_dims must all be > 0, got {dims}")

    return dims


def resolve_instruction(args, backend: EmbeddingBackend) -> Optional[str]:
    if args.no_instruction:
        return ""

    if args.instruction is not None:
        return args.instruction

    return backend.default_instruction(args.role)


# =============================================================================
# main
# =============================================================================

def main():
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be greater than 0.")
    if args.max_pixels is not None and args.max_pixels <= 0:
        raise ValueError("--max_pixels must be greater than 0.")
    if args.dim is not None and args.dim <= 0:
        raise ValueError("--dim must be greater than 0.")
    if args.matryoshka_dims:
        if args.dim is not None:
            raise ValueError("--dim and --matryoshka_dims cannot be used together.")
        if not getattr(BACKEND_REGISTRY[args.model_type], "supports_matryoshka", False):
            supported = sorted(
                name for name, cls in BACKEND_REGISTRY.items()
                if getattr(cls, "supports_matryoshka", False)
            )
            raise ValueError(
                f"--matryoshka_dims is only supported by these backends: {supported}; "
                f"got --model_type={args.model_type}"
            )
    if args.clip_fusion_alpha is not None and not (0.0 <= args.clip_fusion_alpha <= 1.0):
        raise ValueError("--clip_fusion_alpha must be within [0, 1].")
    if args.image_load_workers <= 0:
        raise ValueError("--image_load_workers must be greater than 0.")
    if args.prefetch_depth <= 0:
        raise ValueError("--prefetch_depth must be greater than 0.")

    checkpoint = args.checkpoint
    if os.path.exists(checkpoint):
        checkpoint = os.path.abspath(checkpoint)

    if args.model_type in ("trident_qwen3vl", "trident_jinaclip") and not os.path.isdir(checkpoint):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    repo_root = None
    if args.model_type in ("trident_qwen3vl", "trident_jinaclip"):
        repo_root = args.repo_root or os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        repo_root = os.path.abspath(repo_root)

        if not os.path.isdir(os.path.join(repo_root, "mmemb")):
            raise NotADirectoryError(
                f"mmemb package not found under {repo_root}; specify the directory containing it with --repo_root"
            )

        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        old_pythonpath = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = (
            repo_root if not old_pythonpath
            else repo_root + os.pathsep + old_pythonpath
        )

    # ------------------------------------------------------------ assemble jobs
    if args.jobs_json:
        with open(args.jobs_json, "r", encoding="utf-8") as f:
            raw_jobs = json.load(f)
        if not isinstance(raw_jobs, list) or not raw_jobs:
            raise ValueError(f"--jobs_json {args.jobs_json} must be a non-empty JSON array")
    else:
        if not args.input_jsonl or not args.output_dir or not args.role:
            raise ValueError(
                "Without --jobs_json, --input_jsonl / --output_dir / --role are all required"
            )
        raw_jobs = [
            {
                "input_jsonl": args.input_jsonl,
                "output_dir": args.output_dir,
                "role": args.role,
                "tag": "single",
            }
        ]

    backend_cls = BACKEND_REGISTRY[args.model_type]

    jobs: list[dict] = []
    for job_idx, raw_job in enumerate(raw_jobs):
        input_jsonl = os.path.abspath(raw_job["input_jsonl"])
        output_dir = os.path.abspath(raw_job["output_dir"])
        role = raw_job.get("role") or args.role
        tag = raw_job.get("tag") or f"job{job_idx}"

        if role not in ("query", "doc"):
            raise ValueError(f"job '{tag}': role must be query or doc, got {role!r}")

        if not os.path.isfile(input_jsonl):
            raise FileNotFoundError(f"job '{tag}': input_jsonl does not exist: {input_jsonl}")

        if args.skip_existing and os.path.isfile(
            os.path.join(output_dir, "embeddings.npy")
        ) and (
            not args.matryoshka_dims
            or matryoshka_outputs_complete(output_dir, args.matryoshka_dims)
        ):
            print(f"[skip] job '{tag}': {output_dir}/embeddings.npy already exists; skipping")
            continue

        image_root = raw_job.get("image_root") or args.image_root
        image_root = (
            os.path.abspath(image_root) if image_root
            else os.path.dirname(input_jsonl)
        )
        if not os.path.isdir(image_root):
            raise NotADirectoryError(f"job '{tag}': image root does not exist: {image_root}")

        id_field = raw_job.get("id_field") or args.id_field

        probe_args_dict = {
            "model_type": args.model_type,
            "checkpoint": checkpoint,
            "role": role,
        }
        probe_backend = backend_cls(probe_args_dict, device="cpu")
        if args.clip_fusion_alpha is not None:
            probe_backend.clip_fusion_alpha = args.clip_fusion_alpha

        if raw_job.get("instruction") is not None:
            instruction = raw_job["instruction"]
        else:
            instruction = resolve_instruction(
                argparse.Namespace(**{**vars(args), "role": role}), probe_backend
            )

        num_records, ordered_ids = scan_input(
            input_jsonl=input_jsonl, id_field=id_field,
        )
        if num_records == 0:
            raise RuntimeError(f"job '{tag}': {input_jsonl} contains no non-empty records")

        probe_tasks = load_rank_tasks(
            input_jsonl=input_jsonl, rank=0, world_size=1,
            image_root=image_root, id_field=id_field, role=role,
            instruction=instruction, placeholder_mode=args.image_placeholder,
            backend=probe_backend,
        )
        usable_count = sum(1 for t in probe_tasks if t.get("_usable"))
        mixed = sum(
            1 for t in probe_tasks
            if t["_usable"] and t["_text"] and t["_image_paths"]
        )

        print(
            f"\n[init] job '{tag}': role={role}, records={num_records}, "
            f"usable={usable_count}, input={input_jsonl}"
        )
        if mixed and not probe_backend.supports_fused_text_image:
            print(
                f"[init] job '{tag}': {mixed} records have both text and images; "
                f"{args.model_type} has no native fusion, so CLIP-style late fusion is used "
                f"(alpha={probe_backend.clip_fusion_alpha})."
            )
        print(f"[init] job '{tag}': instruction={instruction!r}")

        os.makedirs(output_dir, exist_ok=True)
        shard_dir = os.path.join(output_dir, "shards")
        if os.path.exists(shard_dir):
            shutil.rmtree(shard_dir)
        os.makedirs(shard_dir, exist_ok=True)

        instructions_path = os.path.join(output_dir, "instructions_used.json")
        with open(instructions_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_type": args.model_type, "role": role,
                    "instruction": instruction,
                    "image_placeholder": args.image_placeholder,
                    "clip_fusion_alpha": probe_backend.clip_fusion_alpha,
                },
                f, ensure_ascii=False, indent=2,
            )

        jobs.append({
            "tag": tag,
            "input_jsonl": input_jsonl,
            "output_dir": output_dir,
            "shard_dir": shard_dir,
            "role": role,
            "instruction": instruction,
            "id_field": id_field,
            "image_root": image_root,
            "num_records": num_records,
            "ordered_ids": ordered_ids,
        })

    if not jobs:
        print("[all done] no job to run (all skipped by --skip_existing?)")
        return

    if args.dry_run:
        print(f"\n[dry_run] {len(jobs)} jobs; only data validation was done, the model was not loaded.")
        for job in jobs:
            print(f"  - {job['tag']}: {job['input_jsonl']} -> {job['output_dir']}")
            shutil.rmtree(job["shard_dir"], ignore_errors=True)
        return

    gpu_ids = parse_gpu_ids(args.gpus)
    max_records = max(job["num_records"] for job in jobs)
    if len(gpu_ids) > max_records:
        gpu_ids = gpu_ids[:max_records]
        print(f"[init] fewer records than GPUs; using GPUs {gpu_ids}")
    world_size = len(gpu_ids)

    print(f"\n[init] model_type={args.model_type}, checkpoint={checkpoint}")
    print(f"[init] GPUs={gpu_ids}, workers={world_size}, batch_size={args.batch_size}")
    print(f"[init] jobs to run: {len(jobs)}")
    if args.matryoshka_dims:
        print(f"[init] matryoshka dims: {args.matryoshka_dims}")
    print(
        f"[init] image_load_workers={args.image_load_workers}, "
        f"prefetch_depth={args.prefetch_depth}\n"
    )

    worker_args = {
        "model_type": args.model_type,
        "checkpoint": checkpoint,
        "base_model": args.base_model,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "save_dtype": args.save_dtype,
        "max_pixels": args.max_pixels,
        "dim": args.dim,
        "matryoshka_dims": args.matryoshka_dims,
        "image_placeholder": args.image_placeholder,
        "sort_by_length": args.sort_by_length,
        "merge_lora": not args.no_merge_lora,
        "repo_root": repo_root,
        "preview": args.preview,
        "jina_task": args.jina_task,
        "clip_fusion_alpha": args.clip_fusion_alpha,
        "qwen3vl_official_script": args.qwen3vl_official_script,
        "attn_implementation": args.attn_implementation,
        "image_load_workers": args.image_load_workers,
        "prefetch_depth": args.prefetch_depth,
        "jobs": jobs,
    }

    mp.spawn(
        worker_main,
        args=(world_size, gpu_ids, worker_args),
        nprocs=world_size,
        join=True,
    )

    for job in jobs:
        merge_shards(
            shard_dir=job["shard_dir"],
            world_size=world_size,
            num_records=job["num_records"],
            ordered_ids=job["ordered_ids"],
            output_dir=job["output_dir"],
            save_dtype=np.dtype(args.save_dtype),
            matryoshka_dims=args.matryoshka_dims,
        )
        if not args.keep_shards:
            shutil.rmtree(job["shard_dir"])

    print("\n[done] all jobs finished")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()