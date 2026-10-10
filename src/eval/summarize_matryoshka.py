#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
summarize_matryoshka.py

Summarize the per-dimension results produced by scripts/run_all_datasets.sh in
Matryoshka mode into a "dataset x dimension" table; writes
matryoshka_summary.json and prints it.

Expected layout (created by run_all_datasets.sh):
    <eval_root>/matryoshka/dim_<d>/<tag>_result.json   per dataset
    <eval_root>/matryoshka/dim_<d>/mix_result.json     Mix (macro / micro average)

Usage:
    python src/eval/summarize_matryoshka.py \\
        --eval_root eval/xxx \\
        --dims 128,256,512,1024,2048 \\
        --metrics ndcg@10,recall@10
"""

import argparse
import json
import os
import re


def parse_int_list(text: str) -> list[int]:
    parts = re.split(r"[,\s\[\]]+", text.strip())
    return sorted({int(p) for p in parts if p})


def load_json(path: str):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def format_table(title: str, rows: dict[str, dict[int, float]], dims: list[int]) -> str:
    headers = [title] + [f"dim={d}" for d in dims]
    table = [headers]
    for name, values in rows.items():
        table.append(
            [name] + [f"{values[d]:.4f}" if d in values else "-" for d in dims]
        )
    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]
    lines = []
    for idx, row in enumerate(table):
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
        if idx == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize Matryoshka evaluation results across dims")
    parser.add_argument("--eval_root", required=True)
    parser.add_argument("--dims", required=True, help="e.g. 128,256,512 or [128, 256]")
    parser.add_argument(
        "--metrics", default=None,
        help="metrics to print, comma separated; default ndcg@K,recall@K (K = largest k in the results)",
    )
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    dims = parse_int_list(args.dims)
    root = os.path.join(args.eval_root, "matryoshka")

    # single[dataset][dim] = metrics;mix[row_name][dim] = metrics
    single: dict[str, dict[int, dict]] = {}
    mix: dict[str, dict[int, dict]] = {}
    k_values: set[int] = set()

    for dim in dims:
        dim_dir = os.path.join(root, f"dim_{dim}")
        if not os.path.isdir(dim_dir):
            continue

        for name in sorted(os.listdir(dim_dir)):
            if not name.endswith("_result.json") or name == "mix_result.json":
                continue
            payload = load_json(os.path.join(dim_dir, name))
            if not payload or "metrics" not in payload:
                continue
            dataset = payload.get("dataset") or name[: -len("_result.json")]
            single.setdefault(dataset, {})[dim] = payload["metrics"]
            k_values.update(payload.get("k_values") or [])

        payload = load_json(os.path.join(dim_dir, "mix_result.json"))
        if payload:
            k_values.update(payload.get("k_values") or [])
            for dataset, item in (payload.get("datasets") or {}).items():
                mix.setdefault(dataset, {})[dim] = item["metrics"]
            for key, label in (("macro_average", "MACRO_AVG"), ("micro_average", "MICRO_AVG")):
                if key in payload:
                    mix.setdefault(label, {})[dim] = payload[key]

    if not single and not mix:
        raise SystemExit(f"[error] no evaluation results found under {root}")

    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    else:
        k = max(k_values) if k_values else 10
        metrics = [f"ndcg@{k}", f"recall@{k}"]

    def pick(block: dict[str, dict[int, dict]], metric: str) -> dict[str, dict[int, float]]:
        return {
            name: {d: m[metric] for d, m in per_dim.items() if metric in m}
            for name, per_dim in block.items()
        }

    for metric in metrics:
        if single:
            print("")
            print(f"[matryoshka] per-dataset {metric}")
            print(format_table("dataset", pick(single, metric), dims))
        if mix:
            print("")
            print(f"[matryoshka] mix corpus {metric}")
            print(format_table("dataset", pick(mix, metric), dims))

    output_json = args.output_json or os.path.join(root, "matryoshka_summary.json")
    os.makedirs(os.path.dirname(os.path.abspath(output_json)) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(
            {
                "dims": dims,
                "single": {n: {str(d): m for d, m in v.items()} for n, v in single.items()},
                "mix": {n: {str(d): m for d, m in v.items()} for n, v in mix.items()},
            },
            file, ensure_ascii=False, indent=2,
        )
    print("")
    print(f"[done] summary written to: {output_json}")


if __name__ == "__main__":
    main()
