#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_mix_retrieval.py

Mixed-modality ("Mix") corpus evaluation.

Procedure:
  1. each dataset's data/<DATASET>/mix_corpus.jsonl is embedded separately
     (by embed_jsonl_unified_multigpu.py; this script only reads the outputs);
  2. the mix corpora of all datasets are concatenated into one large corpus;
  3. for each dataset, its own queries are retrieved against the large corpus,
     reporting Recall / nDCG / MRR / MAP / Precision / Success;
  4. per-dataset metrics + macro / micro (query-weighted) averages are written,
     together with the fraction of retrieved docs from other datasets
     (cross_dataset_rate), which shows how much interference there is.

Id collisions (important):
query / doc ids can collide across datasets (e.g. "0", "q_1"), and a naive
merge would mix up qrels. Each dataset therefore gets a namespace (its tag by
default) and all ids are rewritten as

        <tag>::<raw id>

Corpus row ids, query ids, and the query / doc ids in qrels all use the same
rewriting, so:
  - a query of dataset A never counts a doc of dataset B with the same raw id as a hit;
  - each dataset is evaluated with its own qrels only, and gold docs can only lie
    in its own namespace.

Duplicated doc ids within one dataset (one doc on several rows) are handled as
well: results are deduplicated by doc id keeping the best rank, and
max_duplicate times more rows are fetched so top-k is still full after dedup.

Usage:
    python src/eval/eval_mix_retrieval.py \
        --manifest   eval/xxx/eval_manifest.json \
        --k_values   1,5,10 \
        --device     cuda \
        --output_json eval/xxx/mix_result.json

The manifest is a JSON array whose elements look like:
    {
      "name": "ChartQA",
      "tag": "cqa",
      "queries_dir": "embedding/xxx/cqa_queries",
      "mix_corpus_dir": "embedding/xxx/cqa_mix_corpus",
      "qrels": "data/ChartQA/qrels.jsonl"
    }
"""

import argparse
import json
import os

import numpy as np

from eval_common import (
    CorpusIndex,
    EmbeddingSet,
    dump_run_file,
    evaluate_queries,
    format_metric_table,
    load_qrels,
    parse_k_values,
    resolve_device,
    split_global_id,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mix-corpus (merged datasets) retrieval evaluation")

    parser.add_argument(
        "--manifest", required=True,
        help="dataset manifest JSON (name / tag / queries_dir / mix_corpus_dir / qrels)",
    )
    parser.add_argument("--k_values", default="1,5,10")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--query_batch_size", type=int, default=1024)
    parser.add_argument(
        "--corpus_chunk_size", type=int, default=20000,
        help="corpus rows moved to the GPU per chunk; reduce for very large corpora / small GPUs",
    )
    parser.add_argument(
        "--drop_missing_gold", action="store_true",
        help="remove gold docs that do not exist in the corpus from qrels (default: keep them)",
    )
    parser.add_argument(
        "--dump_run_dir", default=None,
        help="optional: write the top-N retrieval results of each dataset to this directory",
    )
    parser.add_argument("--dump_run_topn", type=int, default=10)
    parser.add_argument(
        "--datasets", default=None,
        help="optional: only evaluate these datasets of the manifest (comma-separated names)",
    )
    parser.add_argument(
        "--embedding_subdir", default="",
        help="optional: subdirectory appended to queries_dir / mix_corpus_dir of the manifest, "
             "e.g. matryoshka/dim_256 (for Matryoshka evaluation)",
    )

    return parser


def _with_subdir(directory: str, subdir: str) -> str:
    return os.path.join(directory, subdir) if subdir else directory


def load_manifest(
    path: str, only: list[str] | None, subdir: str = "",
) -> list[dict]:
    with open(path, "r", encoding="utf-8") as file:
        entries = json.load(file)

    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path} must be a non-empty JSON array")

    normalized: list[dict] = []
    seen_tags: set[str] = set()

    for entry in entries:
        name = entry["name"]

        if only is not None and name not in only:
            continue

        tag = entry.get("tag") or name

        if tag in seen_tags:
            raise ValueError(
                f"duplicated tag in manifest: {tag!r}. Tags are used as id namespaces and must be unique."
            )

        seen_tags.add(tag)

        normalized.append(
            {
                "name": name,
                "tag": tag,
                "queries_dir": _with_subdir(entry["queries_dir"], subdir),
                "mix_corpus_dir": _with_subdir(
                    entry.get("mix_corpus_dir") or entry["corpus_dir"], subdir,
                ),
                "qrels": entry["qrels"],
            }
        )

    if not normalized:
        raise ValueError("manifest is empty after filtering")

    return normalized


def cross_dataset_rate(
    ranked_ids: list[list[str]], own_tag: str, k: int,
) -> float:
    """Fraction of top-k docs that come from other datasets."""
    total = 0
    foreign = 0

    for ranked in ranked_ids:
        for doc_id in ranked[:k]:
            namespace, _ = split_global_id(doc_id)
            total += 1

            if namespace != own_tag:
                foreign += 1

    return foreign / total if total else 0.0


def main():
    args = build_argument_parser().parse_args()

    k_values = parse_k_values(args.k_values)
    device = resolve_device(args.device)
    only = (
        [item.strip() for item in args.datasets.split(",") if item.strip()]
        if args.datasets else None
    )

    entries = load_manifest(args.manifest, only, args.embedding_subdir)

    print(f"[init] device={device}, k_values={k_values}")
    print(f"[init] datasets in the Mix corpus: {[entry['name'] for entry in entries]}")

    # ------------------------------------------------------------------
    # 1. load each dataset's mix corpus, namespaced by tag, and merge into one corpus
    # ------------------------------------------------------------------

    shards: list[EmbeddingSet] = []

    for entry in entries:
        shard = EmbeddingSet(
            entry["mix_corpus_dir"],
            namespace=entry["tag"],
            name=entry["name"],
        )
        shards.append(shard)

        print(
            f"[corpus] {entry['name']:<16} rows={shard.num_rows:>9}  "
            f"usable={shard.num_usable:>9}  failed={shard.num_failed:>6}  "
            f"dim={shard.dim}"
        )

    corpus = CorpusIndex(shards, chunk_size=args.corpus_chunk_size)

    print(
        f"[corpus] merged: rows={corpus.num_rows}, "
        f"unique_docs={corpus.num_unique_docs}, dim={corpus.dim}"
    )

    if corpus.duplicated_doc_ids:
        print(
            f"[warn] {corpus.duplicated_doc_ids} rows still have duplicated doc ids after merging "
            f"(duplicates within a dataset); results are deduplicated by doc id keeping the best rank"
        )

    # ------------------------------------------------------------------
    # 2. evaluate each dataset
    # ------------------------------------------------------------------

    per_dataset: dict[str, dict] = {}
    table_rows: list[tuple[str, dict[str, float]]] = []

    for entry in entries:
        name = entry["name"]
        tag = entry["tag"]

        print("")
        print("=" * 62)
        print(f"[eval] {name} (tag={tag}) on mix corpus")
        print("=" * 62)

        queries = EmbeddingSet(entry["queries_dir"], namespace=tag, name=name)

        # Prefix the query and doc ids in qrels with the dataset's namespace, so gold docs
        # only match this dataset's rows in the merged corpus.
        qrels = load_qrels(
            entry["qrels"], query_namespace=tag, doc_namespace=tag,
        )

        print(
            f"[eval] queries rows={queries.num_rows}, usable={queries.num_usable}; "
            f"qrels queries={len(qrels)}"
        )

        result = evaluate_queries(
            queries=queries,
            corpus=corpus,
            qrels=qrels,
            k_values=k_values,
            device=device,
            query_batch_size=args.query_batch_size,
            drop_missing_gold=args.drop_missing_gold,
        )

        ranked_ids = result["_ranked_ids"]

        interference = {
            f"cross_dataset_rate@{k}": cross_dataset_rate(ranked_ids, tag, k)
            for k in k_values
        }

        if args.dump_run_dir:
            os.makedirs(args.dump_run_dir, exist_ok=True)
            dump_path = os.path.join(args.dump_run_dir, f"{tag}_mix_run.jsonl")
            dump_run_file(
                dump_path,
                result["_query_ids"],
                ranked_ids,
                top_n=args.dump_run_topn,
            )
            print(f"[dump] {dump_path}")

        per_dataset[name] = {
            "tag": tag,
            "queries_dir": os.path.abspath(entry["queries_dir"]),
            "mix_corpus_dir": os.path.abspath(entry["mix_corpus_dir"]),
            "qrels": os.path.abspath(entry["qrels"]),
            "num_evaluated_queries": result["num_evaluated_queries"],
            "metrics": result["metrics"],
            "interference": interference,
            "stats": result["stats"],
        }

        table_rows.append((name, result["metrics"]))

        print("")
        print(format_metric_table([(name, result["metrics"])], k_values))
        print(
            "[eval] cross_dataset_rate: "
            + ", ".join(f"@{k}={interference[f'cross_dataset_rate@{k}']:.4f}"
                        for k in k_values)
        )

    # ------------------------------------------------------------------
    # 3. aggregate: macro + micro averages
    # ------------------------------------------------------------------

    metric_keys = sorted(next(iter(per_dataset.values()))["metrics"].keys())

    macro = {
        key: float(np.mean([item["metrics"][key] for item in per_dataset.values()]))
        for key in metric_keys
    }

    weights = np.asarray(
        [item["num_evaluated_queries"] for item in per_dataset.values()],
        dtype=np.float64,
    )

    micro = {
        key: float(
            np.average(
                [item["metrics"][key] for item in per_dataset.values()],
                weights=weights,
            )
        )
        for key in metric_keys
    }

    payload = {
        "mode": "mix",
        "manifest": os.path.abspath(args.manifest),
        "embedding_subdir": args.embedding_subdir,
        "k_values": k_values,
        "id_namespace_separator": "::",
        "corpus": corpus.describe(),
        "datasets": per_dataset,
        "macro_average": macro,
        "micro_average": micro,
        "total_evaluated_queries": int(weights.sum()),
    }

    os.makedirs(
        os.path.dirname(os.path.abspath(args.output_json)) or ".", exist_ok=True,
    )

    with open(args.output_json, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    print("")
    print("=" * 62)
    print(f"[summary] mix corpus: {corpus.num_rows} rows / {corpus.num_unique_docs} unique docs")
    print("=" * 62)
    print(
        format_metric_table(
            table_rows + [("MACRO_AVG", macro), ("MICRO_AVG", micro)],
            k_values,
        )
    )
    print("")
    print(f"[done] results written to: {args.output_json}")


if __name__ == "__main__":
    main()
