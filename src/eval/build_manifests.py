#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_manifests.py

Manifest generator for scripts/run_all_datasets.sh: builds the job and evaluation
manifests in Python and checks that all input files exist.

Outputs two files:

  jobs_manifest.json   for embed_jsonl_unified_multigpu.py --jobs_json
                       up to 3 jobs per dataset:
                         queries.jsonl       -> <tag>_queries      (role=query)
                         origin_corpus.jsonl -> <tag>_origin_corpus(role=doc)
                         mix_corpus.jsonl    -> <tag>_mix_corpus   (role=doc)

  eval_manifest.json   for eval_retrieval.py / eval_mix_retrieval.py
                       [{name, tag, queries_dir, corpus_dir, mix_corpus_dir, qrels}]

Usage:
    python src/eval/build_manifests.py \
        --data_root data \
        --embed_root embedding/xxx \
        --datasets ChartQA,DocVQA \
        --tags ChartQA=cqa,DocVQA=dqa \
        --jobs_json  eval/xxx/jobs_manifest.json \
        --eval_manifest eval/xxx/eval_manifest.json
"""

import argparse
import json
import os
import sys


def parse_tags(text: str) -> dict[str, str]:
    tags: dict[str, str] = {}

    for item in text.split(","):
        item = item.strip()

        if not item:
            continue

        if "=" not in item:
            raise ValueError(f"--tags entries must look like DATASET=tag, got {item!r}")

        name, tag = item.split("=", 1)
        tags[name.strip()] = tag.strip()

    return tags


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build embedding job / evaluation manifests")

    parser.add_argument("--data_root", required=True)
    parser.add_argument("--embed_root", required=True)
    parser.add_argument("--datasets", required=True, help="comma-separated dataset directory names")
    parser.add_argument("--tags", default="", help="DATASET=tag,DATASET=tag")
    parser.add_argument("--jobs_json", required=True)
    parser.add_argument("--eval_manifest", required=True)

    parser.add_argument("--queries_name", default="queries.jsonl")
    parser.add_argument("--origin_corpus_name", default="origin_corpus.jsonl")
    parser.add_argument("--mix_corpus_name", default="mix_corpus.jsonl")
    parser.add_argument("--qrels_name", default="qrels.jsonl")

    parser.add_argument(
        "--no_mix", action="store_true", help="do not create mix_corpus jobs",
    )
    parser.add_argument(
        "--only_mix", action="store_true",
        help="only create queries + mix_corpus jobs (skip origin_corpus)",
    )

    return parser


def main() -> int:
    args = build_argument_parser().parse_args()

    tags = parse_tags(args.tags)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]

    jobs: list[dict] = []
    eval_entries: list[dict] = []
    skipped: list[str] = []

    for dataset in datasets:
        tag = tags.get(dataset, dataset.lower())

        dataset_dir = os.path.join(args.data_root, dataset)
        queries_jsonl = os.path.join(dataset_dir, args.queries_name)
        origin_jsonl = os.path.join(dataset_dir, args.origin_corpus_name)
        mix_jsonl = os.path.join(dataset_dir, args.mix_corpus_name)
        qrels = os.path.join(dataset_dir, args.qrels_name)

        missing = [
            path for path in (queries_jsonl, qrels) if not os.path.isfile(path)
        ]

        if not args.only_mix and not os.path.isfile(origin_jsonl):
            missing.append(origin_jsonl)

        if not args.no_mix and not os.path.isfile(mix_jsonl):
            missing.append(mix_jsonl)

        if missing:
            print(f"[skip] {dataset}: missing files", file=sys.stderr)

            for path in missing:
                print(f"        missing: {path}", file=sys.stderr)

            skipped.append(f"{dataset} (missing input files)")
            continue

        queries_out = os.path.join(args.embed_root, f"{tag}_queries")
        origin_out = os.path.join(args.embed_root, f"{tag}_origin_corpus")
        mix_out = os.path.join(args.embed_root, f"{tag}_mix_corpus")

        jobs.append(
            {
                "input_jsonl": queries_jsonl,
                "output_dir": queries_out,
                "role": "query",
                "tag": f"{dataset}/query",
            }
        )

        if not args.only_mix:
            jobs.append(
                {
                    "input_jsonl": origin_jsonl,
                    "output_dir": origin_out,
                    "role": "doc",
                    "tag": f"{dataset}/origin_doc",
                }
            )

        if not args.no_mix:
            jobs.append(
                {
                    "input_jsonl": mix_jsonl,
                    "output_dir": mix_out,
                    "role": "doc",
                    "tag": f"{dataset}/mix_doc",
                }
            )

        eval_entries.append(
            {
                "name": dataset,
                "tag": tag,
                "queries_dir": queries_out,
                "corpus_dir": origin_out,
                "mix_corpus_dir": mix_out,
                "qrels": qrels,
            }
        )

    if not jobs:
        print("[error] no dataset has all required files; cannot build manifests", file=sys.stderr)
        return 1

    for path, payload in (
        (args.jobs_json, jobs),
        (args.eval_manifest, eval_entries),
    ):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    print(
        f"[init] jobs={len(jobs)}, datasets={len(eval_entries)}, "
        f"skipped={len(skipped)}",
        file=sys.stderr,
    )

    # stdout only lists the available dataset names so bash can read them into an array.
    for entry in eval_entries:
        print(entry["name"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
