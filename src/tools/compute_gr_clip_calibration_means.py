#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compute_gr_clip_calibration_means.py

Compute the mean-shift calibration vectors of GR-CLIP ("Closing the Modality
Gap for Mixed Modality Search", Algorithm 1), bucketed by **(role, modality)**.
GR-CLIP is used as a post-hoc baseline in our paper (the "+ GR-CLIP" rows).

===============================================================================
Scope: CLIP-based models only
===============================================================================
  GR-CLIP's justification ("the modality gap is approximately a constant vector
  c_perp orthogonal to the image / text subspaces") is a property of
  contrastive dual-tower models (CLIP / OpenCLIP / SigLIP / AltCLIP /
  jina-clip ...). This script therefore only accepts --model_type values whose
  backend has is_clip_based=True
  (clip_vit_l14 / siglip2 / altclip / jina_clip_v2 / trident_jinaclip) and
  exits otherwise, since the inference side (embedding_backends/gr_clip.py) also
  skips calibration for non-CLIP backends. For control experiments pass
  --allow_non_clip_backend and export GR_CLIP_ALLOW_NON_CLIP=1 at evaluation time.

===============================================================================
Buckets: query and doc sides are symmetric over three modalities
===============================================================================
    bucket        role    modality  npz key            notation   when
    ------------  ------  --------  -----------------  ---------  ------------------------------------
    query         query   text      query_mean         e_bar_q    required
    query_image   query   image     query_image_mean   e_bar_qI   if the data has query images
    query_fused   query   fused     query_fused_mean   e_bar_qF   native-fusion backend + text+image queries
    text          doc     text      text_mean          e_bar_T    required
    image         doc     image     image_mean         e_bar_I    required
    fused         doc     fused     fused_mean         e_bar_F    native-fusion backend

  Bucket names are also the subdirectory names under --out_dir.

  Query images / fused queries (e.g. OVEN: image + text; Nights: image only)
  follow a different distribution than document images, just as query text
  differs from document text (e_bar_q vs. e_bar_T in GR-CLIP), so the query
  side is bucketed by modality as well.

Late fusion vs. native fusion:
  - late-fusion backends (supports_fused_text_image=False, e.g. clip_vit_l14 /
    siglip2 / altclip / jina_clip_v2): components are separable, and fused
    records are calibrated by interpolation (Algorithm 1):
        α·(f_I(d_I) - image_mean) + (1-α)·(f_T(d_T) - text_mean)
    so the fused / query_fused buckets are not computed.
  - native-fusion backends (supports_fused_text_image=True, e.g. trident_jinaclip):
    a text+image record is encoded into one vector by the model, so the
    interpolation is not its true mean. The fused bucket (doc side, positive[2])
    and query_fused bucket (queries with both text and image) are computed.
  This is decided automatically from the backend class of --model_type.

===============================================================================
Input format (one JSON object per line)
===============================================================================
  {
    "example_id": "ex_107798",
    "query_id": "q_102616",
    "query": [                                       # list of content pieces
        {"text": "...", "image": "oven/123.jpg"}     # text only / image only / both
    ],
    "doc_id": "d_75853",
    "positive": [
        {"text": "..."},                             # [0] doc text  -> text_mean
        {"image": "infovqa/41432.jpeg"},             # [1] doc image -> image_mean
        {"text": "...", "image": "infovqa/41432.jpeg"}# [2] doc fused -> fused_mean (native-fusion backends only)
    ],
    "task": "default",
    "dataset": "infovqa"
  }

  This is the same layout as the training data (see README, "Data Preparation").

  Requirements (all --model_type values):
    - `query` is a non-empty list from which at least one non-empty text or one
      existing image can be extracted (all pieces are scanned; the first
      non-empty text and the first image are used)
    - positive[0]['text'] is non-empty
    - positive[1]['image'] exists

  For native-fusion backends, positive[2] must additionally exist with
  non-empty text and an existing image.

  Rows that do not satisfy these requirements are skipped with a [skip] log.

  **Buckets are independent**: a row without a query image still contributes to
  the query (text) / text / image buckets, just not to query_image / query_fused.
  Optional buckets without any sample are not computed and their keys are not
  written to the npz; the inference side treats them as optional and falls
  back to the doc-side mean of the same modality with a one-time notice.

===============================================================================
Implementation
===============================================================================
  The multi-GPU pipeline is not reimplemented: this script imports
  src/eval/embed_jsonl_unified_multigpu.py ("core") and reuses its
  BACKEND_REGISTRY / worker_main / merge_shards / scan_input / parse_gpu_ids /
  normalize_text_field / normalize_image_field / is_remote_path.

Usage:
  python src/tools/compute_gr_clip_calibration_means.py \
      --data data/Train/train.jsonl --image_root data/images \
      --model_type jina_clip_v2 --checkpoint jinaai/jina-clip-v2 \
      --gpus 0,1,2,3 --batch_size 16 \
      --out_dir outputs/gr_clip_calib/jina_clip_v2

Outputs:
  <out_dir>/gr_clip_means.npz        the mean vectors (see "Storage format" below)
  <out_dir>/gr_clip_means.meta.json  the same information as readable JSON (without vectors)

===============================================================================
Storage format of gr_clip_means.npz (read by the inference side):
===============================================================================
  np.load("gr_clip_means.npz") contains the following keys:

    query_mean        : (D,) float32   e_bar_q   query text mean           [always]
    query_image_mean  : (D,) float32   e_bar_qI  query image mean          [optional]
    query_fused_mean  : (D,) float32   e_bar_qF  query native fused mean   [optional]
    text_mean         : (D,) float32   e_bar_T   doc text mean             [always]
    image_mean        : (D,) float32   e_bar_I   doc image mean            [always]
    fused_mean        : (D,) float32   e_bar_F   doc native fused mean     [optional]

    <one *_count per mean above>  : ()  int64   number of samples used for that mean
        query_count / query_image_count / query_fused_count /
        text_count / image_count / fused_count

    dim               : ()  int64      embedding dimension D
    model_type        : string         value of --model_type
    checkpoint        : string         value of --checkpoint
    query_instruction : string         instruction used on the query side
    doc_instruction   : string         instruction used on the doc side
    created_at        : string         ISO8601 creation time
    source_data       : string         path of --data

  Note: strings / scalars in the npz are 0-d or 1-element numpy arrays; read them
  with `str(d["model_type"])` or `d["dim"].item()`.

  Inference (pseudo code; Algorithm 1 extended to three query modalities):

    d = np.load("gr_clip_means.npz")

    # ---- query side ----
    q_emb = f_T(query_text) - d["query_mean"]                  # text
    q_emb = f_I(query_image) - d["query_image_mean"]           # image
    q_emb = (alpha * (f_I(q_I) - d["query_image_mean"])        # text+image (late fusion)
             + (1-alpha) * (f_T(q_T) - d["query_mean"]))
    q_emb = f_fuse(q_T, q_I) - d["query_fused_mean"]           # text+image (native fusion)

    # ---- doc side (symmetric) ----
    d_emb = f_T(doc_text) - d["text_mean"]
    d_emb = f_I(doc_image) - d["image_mean"]
    d_emb = alpha * (f_I(d_I) - d["image_mean"]) + (1-alpha) * (f_T(d_T) - d["text_mean"])
    d_emb = f_fuse(d_T, d_I) - d["fused_mean"]

    score = cosine_similarity(q_emb, d_emb)

  You do not need to implement this yourself: embedding_backends/gr_clip.py and
  base.py pick the right mean per (role, modality); just export GR_CLIP_MEANS=<npz path>.
===============================================================================
"""

import argparse
import datetime
import json
import os
import shutil
from typing import Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# The means must be computed from *uncalibrated* embeddings; otherwise they would be
# "means after one calibration" (calibrating twice). Calibration is therefore forced
# off before importing the backends; the environment variable is inherited by the
# mp.spawn workers, so an exported GR_CLIP_MEANS (needed for evaluation) has no effect
# here. See embedding_backends/gr_clip.py.
if os.environ.get("GR_CLIP_MEANS"):
    print(
        "[gr-clip] GR_CLIP_MEANS is set; calibration is forced off while computing the means "
        "(to avoid calibrating twice).",
        flush=True,
    )
os.environ["GR_CLIP_DISABLE"] = "1"

import numpy as np

# The shared multi-GPU pipeline lives in src/eval/embed_jsonl_unified_multigpu.py.
import sys  # noqa: E402

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval")
)
import embed_jsonl_unified_multigpu as core

import torch.multiprocessing as mp

# =============================================================================
# Buckets: Cartesian product of (role, modality)
# =============================================================================
# The query and doc sides are symmetric over three modalities (text / image / fused).
# Bucket names are also the subdirectory names under out_dir.
#
#   bucket          role    modality  npz key              note
#
#   query           query   text      query_mean           e_bar_q  (required)
#   query_image     query   image     query_image_mean     e_bar_qI (only if the data has query images)
#   query_fused     query   fused     query_fused_mean     e_bar_qF (native-fusion backend + text+image queries)
#   text            doc     text      text_mean            e_bar_T  (required)
#   image           doc     image     image_mean           e_bar_I  (required)
#   fused           doc     fused     fused_mean           e_bar_F  (native-fusion backend)
#
# A required bucket without any sample raises an error: the --data format does not
# match the expectations and the means would be unusable.
# Buckets with needs_native_fusion=True are only computed when the backend class has
# supports_fused_text_image=True (late-fusion backends interpolate with α as in
# Algorithm 1 and need no fused mean).
BUCKET_SPECS: dict[str, dict] = {
    "query": {
        "role": "query", "modality": "text",
        "npz_mean": "query_mean", "npz_count": "query_count",
        "required": True, "needs_native_fusion": False,
    },
    "query_image": {
        "role": "query", "modality": "image",
        "npz_mean": "query_image_mean", "npz_count": "query_image_count",
        "required": False, "needs_native_fusion": False,
    },
    "query_fused": {
        "role": "query", "modality": "fused",
        "npz_mean": "query_fused_mean", "npz_count": "query_fused_count",
        "required": False, "needs_native_fusion": True,
    },
    "text": {
        "role": "doc", "modality": "text",
        "npz_mean": "text_mean", "npz_count": "text_count",
        "required": True, "needs_native_fusion": False,
    },
    "image": {
        "role": "doc", "modality": "image",
        "npz_mean": "image_mean", "npz_count": "image_count",
        "required": True, "needs_native_fusion": False,
    },
    "fused": {
        "role": "doc", "modality": "fused",
        "npz_mean": "fused_mean", "npz_count": "fused_count",
        "required": True, "needs_native_fusion": True,
    },
}

# Fixed bucket order (stable ordering of logs / npz keys)
BUCKET_ORDER = ("query", "query_image", "query_fused", "text", "image", "fused")

# Short-hand names
BASE_BUCKETS = ("query", "text", "image")
FUSED_BUCKET = "fused"

ROLE_OF_BUCKET = {name: spec["role"] for name, spec in BUCKET_SPECS.items()}


def candidate_buckets(need_fused: bool) -> tuple[str, ...]:
    """Buckets this --model_type may compute (they still need samples in the data)."""
    return tuple(
        name for name in BUCKET_ORDER
        if need_fused or not BUCKET_SPECS[name]["needs_native_fusion"]
    )


def bucket_manifest_record(bucket: str, item: dict) -> Optional[dict]:
    """
    Manifest content of a usable record for a bucket; None if the record does not
    match the bucket's modality (e.g. no query image for the query_image bucket),
    i.e. it does not contribute to that bucket's mean.

    The sample sets of the buckets are **independent**: query images do not exist in
    many datasets, so requiring them for every record would drop the whole dataset.
    Records are filtered per bucket; an empty bucket is simply not computed and its
    key is not written.
    """
    spec = BUCKET_SPECS[bucket]
    role, modality = spec["role"], spec["modality"]

    if role == "query":
        text, image = item.get("query_text"), item.get("query_image")
    elif modality == "fused":
        text, image = item.get("fused_text"), item.get("fused_image")
    else:
        text, image = item.get("doc_text"), item.get("doc_image")

    if modality == "text":
        return {"text": text} if text else None
    if modality == "image":
        return {"image": image} if image else None
    # modality == "fused"
    if text and image:
        return {"text": text, "image": image}
    return None


def model_is_clip_based(model_type: str) -> bool:
    """Whether the backend of --model_type is CLIP-based (contrastive dual tower)."""
    backend_cls = core.BACKEND_REGISTRY[model_type]
    return bool(getattr(backend_cls, "is_clip_based", False))


def check_backend_is_clip_based(args) -> None:
    """
    GR-CLIP mean-shift calibration is only defined for CLIP-based models (it relies
    on "the modality gap is approximately a constant vector orthogonal to the
    image / text subspaces", a property of contrastive dual-tower models).

    By default means are therefore only computed for CLIP-based backends; otherwise
    the npz would be rejected by gr_clip.resolve_calibrator at inference anyway.
    Pass --allow_non_clip_backend for control experiments with non-CLIP models.
    """
    if model_is_clip_based(args.model_type):
        return

    clip_models = sorted(
        m for m in core.BACKEND_REGISTRY if model_is_clip_based(m)
    )
    if not args.allow_non_clip_backend:
        raise SystemExit(
            f"[gr-clip] --model_type={args.model_type!r} is not a CLIP-based model "
            f"(is_clip_based=False). GR-CLIP mean calibration is only defined for contrastive "
            f"dual-tower models; supported: {clip_models}.\n"
            f"        The inference side also skips calibration for non-CLIP backends, so this "
            f"run is stopped early.\n"
            f"        For control experiments: pass --allow_non_clip_backend to this script "
            f"and export GR_CLIP_ALLOW_NON_CLIP=1 at evaluation time."
        )

    print(
        f"[gr-clip][warn] --model_type={args.model_type!r} is not a CLIP-based model "
        f"but was allowed by --allow_non_clip_backend. The mean shift is not validated for "
        f"such models; use the results for control experiments only and set "
        f"GR_CLIP_ALLOW_NON_CLIP=1 at inference, otherwise calibration is skipped.",
        flush=True,
    )


def model_supports_native_fusion(model_type: str) -> bool:
    """Whether the backend class of --model_type natively encodes one fused text+image vector
    per record (supports_fused_text_image=True); decides whether the fused buckets are computed."""
    backend_cls = core.BACKEND_REGISTRY[model_type]
    return bool(getattr(backend_cls, "supports_fused_text_image", False))


# =============================================================================
# Part 1: read the query/positive JSONL and split it into per-bucket manifests
# =============================================================================

def load_json_records(jsonl_path: str) -> list[dict]:
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _resolve_image_path(raw_path: str, image_root_abs: str) -> Optional[str]:
    """Join relative paths with image_root; remote paths (http / oss ...) unchanged; None if a local file is missing."""
    if core.is_remote_path(raw_path):
        return raw_path
    path = raw_path if os.path.isabs(raw_path) else os.path.normpath(
        os.path.join(image_root_abs, raw_path)
    )
    if not os.path.isfile(path):
        return None
    return path


def _extract_query_parts(
    query_list: list,
    image_root_abs: str,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract (query_text, query_image, error) from the `query` field.

    `query` is a list of content pieces, each with text, an image, or both.
    Queries come in three forms:
        - text T          : Google WIT / MSCOCO / VisualNews / ...
        - image + text T+I: OVEN (an image + "What is the name of this building?")
        - image I         : Nights
    so taking only query[0]['text'] is not enough.

    All pieces are scanned; the first non-empty text and the first resolvable image
    are used (every backend supports at most one query image).

    A non-empty error means a hard error (an image path that does not exist) and the
    caller should skip the record; text and image both None (no content) should also
    be skipped.
    """
    query_text: Optional[str] = None
    query_image: Optional[str] = None

    for part in query_list:
        if not isinstance(part, dict):
            continue

        if query_text is None:
            text = core.normalize_text_field(part.get("text"))
            if text:
                query_text = text

        if query_image is None:
            raw_image = part.get("image")
            if raw_image:
                # Some data stores `image` as a list; reuse the driver's normalization.
                candidates = core.normalize_image_field(raw_image)
                if not candidates:
                    continue
                resolved = _resolve_image_path(candidates[0], image_root_abs)
                if resolved is None:
                    return None, None, f"query image not found: {candidates[0]}"
                query_image = resolved

    return query_text, query_image, None


def collect_usable_records(args, need_fused: bool) -> list[dict]:
    """
    Parse every line, validate the query/positive structure, and return:
      [{"eid", "query_text"?, "query_image"?, "doc_text", "doc_image",
        "fused_text"?, "fused_image"?}, ...]

    Validation (failing rows are skipped with a [skip] reason; the run continues):
      - `query` is a non-empty list with at least text or an image
        (text only / image only / image + text, see _extract_query_parts)
      - `positive` has at least 2 entries
      - positive[0] has non-empty text -> text_mean
      - positive[1] has an image (existing path or remote) -> image_mean
      - need_fused=False (late-fusion backend): positive[2] (fused document), if
        present, is ignored.
      - need_fused=True (native-fusion backend): positive[2] must exist with
        non-empty text and an existing image -> fused_mean; otherwise the row is skipped.

    Whether the query has an image does **not** affect usability: rows without query
    images still contribute to the query (text) / text / image buckets, just not to
    query_image / query_fused. Per-bucket filtering happens in write_manifests.
    """
    records = load_json_records(args.data)
    if args.max_samples:
        records = records[: args.max_samples]
    print(f"[data] loaded {len(records)} records from {args.data}", flush=True)

    image_root_abs = os.path.abspath(args.image_root)

    usable: list[dict] = []
    seen_ids: dict[str, int] = {}
    duplicate_count = 0
    skip_count = 0

    for i, r in enumerate(records):
        raw_id = r.get("example_id")
        eid = str(raw_id) if raw_id is not None else str(i)

        # ---- query (text only / image only / image + text) ----
        query_list = r.get("query")
        if not query_list or not isinstance(query_list, list):
            print(f"[skip] id={eid}: 'query' is missing or not a list", flush=True)
            skip_count += 1
            continue

        query_text, query_image, query_error = _extract_query_parts(
            query_list, image_root_abs
        )
        if query_error:
            print(f"[skip] id={eid}: {query_error}", flush=True)
            skip_count += 1
            continue
        if not query_text and not query_image:
            print(
                f"[skip] id={eid}: query has neither non-empty text nor an image",
                flush=True,
            )
            skip_count += 1
            continue

        # ---- positive ----
        positive = r.get("positive")
        if not positive or len(positive) < 2:
            print(
                f"[skip] id={eid}: 'positive' has fewer than 2 entries "
                f"(got {len(positive) if positive else 0}); at least "
                f"positive[0] (text) and positive[1] (image) are required",
                flush=True,
            )
            skip_count += 1
            continue

        p_text_doc, p_image_doc = positive[0], positive[1]

        doc_text = core.normalize_text_field(p_text_doc.get("text"))
        if not doc_text:
            print(f"[skip] id={eid}: positive[0]['text'] is missing / empty", flush=True)
            skip_count += 1
            continue

        raw_doc_image = p_image_doc.get("image")
        if not raw_doc_image:
            print(f"[skip] id={eid}: positive[1]['image'] is missing", flush=True)
            skip_count += 1
            continue
        doc_image = _resolve_image_path(raw_doc_image, image_root_abs)
        if doc_image is None:
            print(f"[skip] id={eid}: image not found: {raw_doc_image}", flush=True)
            skip_count += 1
            continue

        # ---- positive[2] (fused document), only needed for native-fusion backends ----
        fused_text = None
        fused_image = None
        if need_fused:
            if len(positive) < 3:
                print(
                    f"[skip] id={eid}: the current --model_type natively fuses text and images "
                    f"and needs positive[2] as the fused document, but 'positive' only has "
                    f"{len(positive)}",
                    flush=True,
                )
                skip_count += 1
                continue

            p_fused_doc = positive[2]
            fused_text = core.normalize_text_field(p_fused_doc.get("text"))
            if not fused_text:
                print(f"[skip] id={eid}: positive[2]['text'] is missing / empty", flush=True)
                skip_count += 1
                continue

            raw_fused_image = p_fused_doc.get("image")
            if not raw_fused_image:
                print(f"[skip] id={eid}: positive[2]['image'] is missing", flush=True)
                skip_count += 1
                continue
            fused_image = _resolve_image_path(raw_fused_image, image_root_abs)
            if fused_image is None:
                print(f"[skip] id={eid}: image not found: {raw_fused_image}", flush=True)
                skip_count += 1
                continue

        if eid in seen_ids:
            duplicate_count += 1
            eid = f"{eid}__dup{duplicate_count}"
        seen_ids[eid] = 1

        entry = {
            "eid": eid,
            "query_text": query_text,
            "query_image": query_image,
            "doc_text": doc_text,
            "doc_image": doc_image,
        }
        if need_fused:
            entry["fused_text"] = fused_text
            entry["fused_image"] = fused_image
        usable.append(entry)

    if duplicate_count:
        print(f"[warn] {duplicate_count} records have duplicated example_id; suffixes were added", flush=True)
    if skip_count:
        print(f"[warn] skipped {skip_count} records that do not match the expected format", flush=True)

    if not usable:
        raise RuntimeError("No valid record; please check that --data has the expected format")

    num_query_text = sum(1 for e in usable if e.get("query_text"))
    num_query_image = sum(1 for e in usable if e.get("query_image"))
    num_query_both = sum(
        1 for e in usable if e.get("query_text") and e.get("query_image")
    )
    print(
        f"[data] {len(usable)} valid records "
        f"(query: text={num_query_text}, image={num_query_image}, "
        f"text+image={num_query_both})",
        flush=True,
    )
    return usable


def write_manifests(
    usable: list[dict],
    manifest_dir: str,
    buckets: tuple[str, ...],
) -> tuple[dict[str, str], dict[str, int]]:
    """
    Write one manifest per candidate bucket; return (manifest path per bucket, row count per bucket).

    Each bucket only contains **its own** records, so row counts can differ across
    buckets (e.g. 10000 query-text rows but query_image rows only for OVEN / Nights).
    Empty buckets still get an (empty) file for debugging, but the caller drops them.

    The manifest schema is identical to embed_jsonl_unified_multigpu.py
    ({"id", "text"?, "image"?}); a fused-bucket record carries both text and image,
    like the text+image records of a mix_corpus.
    """
    os.makedirs(manifest_dir, exist_ok=True)
    paths = {b: os.path.join(manifest_dir, f"{b}.jsonl") for b in buckets}
    counts = {b: 0 for b in buckets}

    handles = {b: open(paths[b], "w", encoding="utf-8") for b in buckets}
    try:
        for item in usable:
            for bucket in buckets:
                record = bucket_manifest_record(bucket, item)
                if record is None:
                    continue
                record = {"id": item["eid"], **record}
                handles[bucket].write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[bucket] += 1
    finally:
        for h in handles.values():
            h.close()

    for bucket in buckets:
        spec = BUCKET_SPECS[bucket]
        print(
            f"[data] bucket='{bucket}' (role={spec['role']}, "
            f"modality={spec['modality']}) -> {counts[bucket]} samples",
            flush=True,
        )

    return paths, counts


def select_buckets_to_compute(
    counts: dict[str, int],
    buckets: tuple[str, ...],
) -> tuple[str, ...]:
    """
    Drop candidate buckets without samples; an empty required bucket raises an error.

    Typical case: all queries of --data are text-only (MSCOCO / Google WIT /
    VisualNews ...), so query_image / query_fused are empty; they are not computed
    and query_image_mean / query_fused_mean are not written. The inference side
    (gr_clip.py) treats these keys as optional and falls back to the doc-side
    image_mean / fused_mean (with a one-time notice) for datasets with image
    queries (OVEN / Nights).
    """
    selected: list[str] = []
    for bucket in buckets:
        spec = BUCKET_SPECS[bucket]
        if counts.get(bucket, 0) > 0:
            selected.append(bucket)
            continue
        if spec["required"]:
            raise RuntimeError(
                f"bucket '{bucket}'(role={spec['role']}, "
                f"modality={spec['modality']}) has no sample, but it is required. "
                f"Please check that --data has the expected format."
            )
        print(
            f"[data] bucket='{bucket}' has no sample; skipping. "
            f"{spec['npz_mean']} will not be written to the npz (the inference side falls back, see gr_clip.py).",
            flush=True,
        )
    return tuple(selected)


# =============================================================================
# Part 2: build jobs / reuse the multi-GPU pipeline of embed_jsonl_unified_multigpu
# =============================================================================

def resolve_checkpoint_and_repo_root(args) -> tuple[str, Optional[str]]:
    checkpoint = args.checkpoint
    if os.path.exists(checkpoint):
        checkpoint = os.path.abspath(checkpoint)

    if args.model_type in ("trident_qwen3vl", "trident_jinaclip") and not os.path.isdir(checkpoint):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    repo_root = None
    if args.model_type in ("trident_qwen3vl", "trident_jinaclip"):
        repo_root = args.repo_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        repo_root = os.path.abspath(repo_root)
        if not os.path.isdir(os.path.join(repo_root, "mmemb")):
            raise NotADirectoryError(
                f"mmemb package not found under {repo_root}; specify the directory containing it with --repo_root"
            )
        import sys
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        old_pythonpath = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = (
            repo_root if not old_pythonpath else repo_root + os.pathsep + old_pythonpath
        )

    return checkpoint, repo_root


def build_all_jobs(args, checkpoint: str, manifest_paths: dict[str, str], manifest_dir: str,
                    buckets_to_run: list[str]) -> tuple[list[dict], dict[str, str]]:
    """
    The role of each bucket comes from BUCKET_SPECS (query / query_image / query_fused
    use the role='query' instruction; text / image / fused use role='doc'). Each job
    carries its own role / instruction and worker_main processes jobs one by one, so
    jobs with different roles can run within one mp.spawn.

    Using the right role matters: the instruction is prepended to the text / fed to
    the model and differs between queries and documents; encoding query images with
    the doc instruction would yield means that do not match the real query distribution.
    """
    backend_cls = core.BACKEND_REGISTRY[args.model_type]

    def _probe_instruction(role: str) -> str:
        if args.no_instruction:
            return ""
        if args.instruction is not None:
            return args.instruction
        probe_args_dict = {"model_type": args.model_type, "checkpoint": checkpoint, "role": role}
        probe_backend = backend_cls(probe_args_dict, device="cpu")
        if args.clip_fusion_alpha is not None:
            probe_backend.clip_fusion_alpha = args.clip_fusion_alpha
        return probe_backend.default_instruction(role)

    instructions = {
        "query": _probe_instruction("query"),
        "doc": _probe_instruction("doc"),
    }
    print(f"[init] query instruction={instructions['query']!r}", flush=True)
    print(f"[init] doc   instruction={instructions['doc']!r}", flush=True)

    jobs: list[dict] = []
    for bucket in buckets_to_run:
        role = BUCKET_SPECS[bucket]["role"]
        input_jsonl = os.path.abspath(manifest_paths[bucket])
        output_dir = os.path.abspath(os.path.join(args.out_dir, bucket))
        os.makedirs(output_dir, exist_ok=True)

        num_records, ordered_ids = core.scan_input(input_jsonl=input_jsonl, id_field="id")
        if num_records == 0:
            raise RuntimeError(f"bucket '{bucket}': {input_jsonl} contains no non-empty records")

        shard_dir = os.path.join(output_dir, "shards")
        if os.path.exists(shard_dir):
            shutil.rmtree(shard_dir)
        os.makedirs(shard_dir, exist_ok=True)

        jobs.append(
            {
                "tag": bucket,
                "input_jsonl": input_jsonl,
                "output_dir": output_dir,
                "shard_dir": shard_dir,
                "image_root": manifest_dir,
                "role": role,
                "instruction": instructions[role],
                "id_field": "id",
                "num_records": num_records,
                "ordered_ids": ordered_ids,
            }
        )

    return jobs, instructions


def run_jobs(args, checkpoint: str, repo_root: Optional[str], jobs: list[dict]) -> None:
    gpu_ids = core.parse_gpu_ids(args.gpus)
    max_records = max(job["num_records"] for job in jobs)
    if len(gpu_ids) > max_records:
        gpu_ids = gpu_ids[:max_records]
        print(f"[init] fewer records than GPUs; using GPUs {gpu_ids}", flush=True)
    world_size = len(gpu_ids)

    print(f"\n[init] model_type={args.model_type}, checkpoint={checkpoint}", flush=True)
    print(f"[init] GPUs={gpu_ids}, workers={world_size}, batch_size={args.batch_size}", flush=True)
    print(f"[init] buckets to run: {[j['tag'] for j in jobs]}\n", flush=True)

    worker_args = {
        "model_type": args.model_type,
        "checkpoint": checkpoint,
        "base_model": args.base_model,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "save_dtype": args.save_dtype,
        "max_pixels": args.max_pixels,
        "dim": args.dim,
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

    mp.spawn(core.worker_main, args=(world_size, gpu_ids, worker_args), nprocs=world_size, join=True)

    for job in jobs:
        core.merge_shards(
            shard_dir=job["shard_dir"],
            world_size=world_size,
            num_records=job["num_records"],
            ordered_ids=job["ordered_ids"],
            output_dir=job["output_dir"],
            save_dtype=np.dtype(args.save_dtype),
        )
        if not args.keep_shards:
            shutil.rmtree(job["shard_dir"])

    print("\n[done] all buckets finished", flush=True)


# =============================================================================
# Part 3: load each bucket's embeddings.npy, compute the means, save
# =============================================================================

def bucket_cache_is_valid(output_dir: str, expected_num_records: int) -> bool:
    embeddings_path = os.path.join(output_dir, "embeddings.npy")
    ids_path = os.path.join(output_dir, "ids.txt")
    if not (os.path.isfile(embeddings_path) and os.path.isfile(ids_path)):
        return False
    try:
        embeddings = np.load(embeddings_path, mmap_mode="r")
    except Exception:
        return False
    return embeddings.shape[0] == expected_num_records


def compute_bucket_mean(output_dir: str) -> tuple[np.ndarray, int]:
    embeddings = np.load(os.path.join(output_dir, "embeddings.npy")).astype(np.float32)
    mean_vec = embeddings.mean(axis=0)
    return mean_vec, embeddings.shape[0]


# =============================================================================
# CLI
# =============================================================================

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute GR-CLIP (Algorithm 1) mean-shift calibration vectors from a "
            "query/positive JSONL corpus, bucketed by (role, modality) "
            "(e_bar_q / e_bar_T / e_bar_I, plus optional ones), saved into one .npz file."
        )
    )

    parser.add_argument("--data", type=str, required=True, help="path of the query/positive JSONL")
    parser.add_argument("--image_root", type=str, default=".", help="root directory of relative image paths")
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--model_type", required=True, choices=sorted(core.BACKEND_REGISTRY.keys()))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--repo_root", default=None)
    parser.add_argument("--instruction", default=None, help="override the default instruction (used for both query and doc; normally unused)")
    parser.add_argument("--no_instruction", action="store_true")
    parser.add_argument("--image_placeholder", choices=["strip", "keep"], default="strip")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                         choices=["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--save_dtype", type=str, default="float32",
                         choices=["float16", "float32"],
                         help="storage precision of the intermediate embeddings.npy (the means are always saved as float32)")
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--no_merge_lora", action="store_true")
    parser.add_argument("--jina_task", default=None)
    parser.add_argument("--clip_fusion_alpha", type=float, default=None)
    parser.add_argument("--qwen3vl_official_script", default=None)
    parser.add_argument("--attn_implementation", default=None)

    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--sort_by_length", action="store_true")
    parser.add_argument("--image_load_workers", type=int, default=8)
    parser.add_argument("--prefetch_depth", type=int, default=3)
    parser.add_argument("--preview", type=int, default=0)
    parser.add_argument("--keep_shards", action="store_true")

    parser.add_argument("--out_dir", type=str, default="./gr_clip_calib_out")
    parser.add_argument("--force", action="store_true", help="ignore the cache and re-encode all buckets")
    parser.add_argument("--dry_run", action="store_true", help="only validate data / build manifests, no inference")
    parser.add_argument(
        "--allow_non_clip_backend",
        action="store_true",
        help=(
            "Allow a non-CLIP-based --model_type (rejected by default). GR-CLIP's mean shift "
            "is only defined for contrastive dual-tower models; evaluation then also needs "
            "export GR_CLIP_ALLOW_NON_CLIP=1."
        ),
    )

    return parser


def validate_args(args) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be greater than 0.")
    if args.max_pixels is not None and args.max_pixels <= 0:
        raise ValueError("--max_pixels must be greater than 0.")
    if args.dim is not None and args.dim <= 0:
        raise ValueError("--dim must be greater than 0.")
    if args.clip_fusion_alpha is not None and not (0.0 <= args.clip_fusion_alpha <= 1.0):
        raise ValueError("--clip_fusion_alpha must be within [0, 1].")


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    validate_args(args)

    check_backend_is_clip_based(args)

    os.makedirs(args.out_dir, exist_ok=True)

    need_fused = model_supports_native_fusion(args.model_type)
    candidates = candidate_buckets(need_fused)
    if need_fused:
        print(
            f"[init] model_type={args.model_type!r} natively fuses text and images "
            f"(supports_fused_text_image=True); fused means are computed as well "
            f"(doc side: positive[2]; query side: queries with text and image; native "
            f"fused forward).",
            flush=True,
        )

    usable = collect_usable_records(args, need_fused=need_fused)

    manifest_dir = os.path.join(args.out_dir, "manifests")
    manifest_paths, manifest_counts = write_manifests(usable, manifest_dir, candidates)
    print(f"[data] manifests written to {manifest_dir}", flush=True)

    buckets = select_buckets_to_compute(manifest_counts, candidates)
    print(f"[init] buckets to compute: {list(buckets)}", flush=True)

    if args.dry_run:
        print(f"\n[dry_run] {len(usable)} records; manifests written, the model was not loaded.")
        return

    buckets_to_run = []
    for bucket in buckets:
        output_dir = os.path.join(args.out_dir, bucket)
        # Buckets have different sample counts, so the cache check uses **this bucket's**
        # row count instead of len(usable).
        if not args.force and bucket_cache_is_valid(output_dir, manifest_counts[bucket]):
            print(f"[cache] bucket='{bucket}': reusing existing embeddings.npy, skipping re-encoding.")
        else:
            buckets_to_run.append(bucket)

    instructions = {"query": None, "doc": None}
    if buckets_to_run:
        checkpoint, repo_root = resolve_checkpoint_and_repo_root(args)
        jobs, instructions = build_all_jobs(args, checkpoint, manifest_paths, manifest_dir, buckets_to_run)
        run_jobs(args, checkpoint, repo_root, jobs)
    else:
        print(
            f"[cache] embeddings of all {len(buckets)} buckets exist; reusing the cache "
            f"(pass --force to recompute).",
            flush=True,
        )
        checkpoint = args.checkpoint

    # ---- compute and save the means ----
    means = {}
    counts = {}
    dim = None
    for bucket in buckets:
        output_dir = os.path.join(args.out_dir, bucket)
        mean_vec, cnt = compute_bucket_mean(output_dir)
        means[bucket] = mean_vec
        counts[bucket] = cnt
        if dim is None:
            dim = mean_vec.shape[0]
        elif dim != mean_vec.shape[0]:
            raise RuntimeError(
                f"embedding dim of bucket '{bucket}' ({mean_vec.shape[0]}) differs from the other buckets ({dim})"
            )
        spec = BUCKET_SPECS[bucket]
        print(
            f"[mean] bucket='{bucket}' -> {spec['npz_mean']}: "
            f"count={cnt}, dim={mean_vec.shape[0]}",
            flush=True,
        )

    created_at = datetime.datetime.now().isoformat(timespec="seconds")
    out_npz = os.path.join(args.out_dir, "gr_clip_means.npz")

    # Means + counts are written per bucket. Which keys exist depends on the buckets that
    # were actually computed: the three required ones (query_mean / text_mean / image_mean)
    # always exist; query_image_mean / query_fused_mean / fused_mean are optional and the
    # inference side (gr_clip.py) treats them as optional.
    npz_payload: dict = {}
    npz_keys: list[str] = []
    for bucket in BUCKET_ORDER:
        if bucket not in means:
            continue
        spec = BUCKET_SPECS[bucket]
        npz_payload[spec["npz_mean"]] = means[bucket].astype(np.float32)
        npz_payload[spec["npz_count"]] = np.int64(counts[bucket])
        npz_keys += [spec["npz_mean"], spec["npz_count"]]

    npz_payload.update(
        dim=np.int64(dim),
        model_type=np.array(args.model_type),
        checkpoint=np.array(args.checkpoint),
        query_instruction=np.array(instructions.get("query") or ""),
        doc_instruction=np.array(instructions.get("doc") or ""),
        created_at=np.array(created_at),
        source_data=np.array(os.path.abspath(args.data)),
    )
    npz_keys += [
        "dim", "model_type", "checkpoint",
        "query_instruction", "doc_instruction", "created_at", "source_data",
    ]

    np.savez(out_npz, **npz_payload)

    meta = {
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "dim": dim,
        "counts": {BUCKET_SPECS[b]["npz_count"]: counts[b] for b in means},
        "means": {BUCKET_SPECS[b]["npz_mean"]: BUCKET_SPECS[b]["role"] + "/" +
                  BUCKET_SPECS[b]["modality"] for b in means},
        "has_query_image_mean": "query_image" in means,
        "has_query_fused_mean": "query_fused" in means,
        "has_fused_mean": "fused" in means,
        "query_instruction": instructions.get("query") or "",
        "doc_instruction": instructions.get("doc") or "",
        "created_at": created_at,
        "source_data": os.path.abspath(args.data),
        "npz_keys": npz_keys,
    }
    out_meta = os.path.join(args.out_dir, "gr_clip_means.meta.json")
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nSaved GR-CLIP calibration means to {out_npz}", flush=True)
    print(f"Saved readable metadata to {out_meta}", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()