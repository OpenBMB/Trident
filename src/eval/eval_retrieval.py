#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_retrieval.py

Per-dataset retrieval evaluation ("Origin" setting): queries are retrieved
against the dataset's own original corpus (embeddings of
data/<DATASET>/origin_corpus.jsonl).

Metrics: Recall@k / nDCG@k / MRR@k / MAP@k / Precision@k / Success@k

Usage:
    python src/eval/eval_retrieval.py \
        --queries_dir embedding/xxx/cqa_queries \
        --corpus_dir  embedding/xxx/cqa_origin_corpus \
        --qrels       data/ChartQA/qrels.jsonl \
        --k_values    1,5,10 \
        --device      cuda \
        --output_json eval/xxx/cqa_result.json
"""

import argparse
import json
import os

from eval_common import (
    CorpusIndex,
    EmbeddingSet,
    dump_run_file,
    evaluate_queries,
    format_metric_table,
    load_qrels,
    parse_k_values,
    resolve_device,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Per-dataset retrieval evaluation")

    parser.add_argument("--queries_dir", required=True, help="query embedding directory")
    parser.add_argument("--corpus_dir", required=True, help="corpus embedding directory")
    parser.add_argument("--qrels", required=True, help="qrels.jsonl")
    parser.add_argument("--k_values", default="1,5,10")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--dataset_name", default=None, help="dataset name written into the result json (optional)",
    )
    parser.add_argument("--query_batch_size", type=int, default=1024)
    parser.add_argument(
        "--corpus_chunk_size", type=int, default=20000,
        help="corpus rows moved to the GPU per chunk; reduce if GPU memory is tight",
    )
    parser.add_argument(
        "--drop_missing_gold", action="store_true",
        help="remove gold docs that do not exist in the corpus from qrels (default: keep them)",
    )
    parser.add_argument(
        "--dump_run", default=None, help="optional: write the top-N retrieval results as jsonl",
    )
    parser.add_argument("--dump_run_topn", type=int, default=10)

    return parser


def main():
    args = build_argument_parser().parse_args()

    k_values = parse_k_values(args.k_values)
    device = resolve_device(args.device)

    dataset_name = args.dataset_name or os.path.basename(
        os.path.dirname(os.path.abspath(args.qrels))
    )

    print(f"[init] dataset={dataset_name}, device={device}, k_values={k_values}")

    # No namespace prefix is needed for a single dataset: ids are only used within it.
    queries = EmbeddingSet(args.queries_dir, namespace=None, name="queries")
    corpus_shard = EmbeddingSet(args.corpus_dir, namespace=None, name="corpus")
    corpus = CorpusIndex([corpus_shard], chunk_size=args.corpus_chunk_size)

    print(
        f"[init] queries: rows={queries.num_rows}, usable={queries.num_usable}, "
        f"failed={queries.num_failed}"
    )
    print(
        f"[init] corpus : rows={corpus.num_rows}, unique_docs={corpus.num_unique_docs}, "
        f"failed={corpus_shard.num_failed}, dim={corpus.dim}"
    )

    if corpus.duplicated_doc_ids:
        print(
            f"[warn] the corpus contains {corpus.duplicated_doc_ids} rows with duplicated doc ids; "
            f"results are deduplicated by doc id, keeping the best rank"
        )

    qrels = load_qrels(args.qrels)
    print(f"[init] qrels: {len(qrels)} queries with annotations")

    result = evaluate_queries(
        queries=queries,
        corpus=corpus,
        qrels=qrels,
        k_values=k_values,
        device=device,
        query_batch_size=args.query_batch_size,
        drop_missing_gold=args.drop_missing_gold,
    )

    if args.dump_run:
        dump_run_file(
            args.dump_run,
            result["_query_ids"],
            result["_ranked_ids"],
            top_n=args.dump_run_topn,
        )
        print(f"[dump] retrieval results: {args.dump_run}")

    payload = {
        "mode": "single",
        "dataset": dataset_name,
        "queries_dir": os.path.abspath(args.queries_dir),
        "corpus_dir": os.path.abspath(args.corpus_dir),
        "qrels": os.path.abspath(args.qrels),
        "k_values": k_values,
        "num_evaluated_queries": result["num_evaluated_queries"],
        "metrics": result["metrics"],
        "stats": result["stats"],
    }

    os.makedirs(
        os.path.dirname(os.path.abspath(args.output_json)) or ".", exist_ok=True,
    )

    with open(args.output_json, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    stats = result["stats"]
    print("")
    print(format_metric_table([(dataset_name, result["metrics"])], k_values))
    print("")
    print(
        f"[done] evaluated_queries={result['num_evaluated_queries']}, "
        f"no_qrels={stats['queries_without_qrels']}, "
        f"embedding_failed={stats['queries_embedding_failed']}, "
        f"gold_missing_in_corpus={stats['gold_docs_missing_in_corpus']}"
    )
    print(f"[done] results written to: {args.output_json}")


if __name__ == "__main__":
    main()
