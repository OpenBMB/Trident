#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_common.py

Shared retrieval-evaluation utilities used by two scripts:

    eval_retrieval.py        per-dataset evaluation (corpus = origin_corpus.jsonl)
    eval_mix_retrieval.py    Mix evaluation (the mix_corpus.jsonl of all datasets is
                             merged into one large corpus, then each dataset is evaluated)

This module:
  1. reads embedding directories produced by embed_jsonl_unified_multigpu.py
     (embeddings.npy / ids.txt / token_counts.jsonl) and detects rows whose
     embedding failed (all-zero vectors);
  2. parses qrels.jsonl (several common field names, list / dict forms);
  3. namespaces ids: when several datasets are merged, query / doc ids may
     collide across datasets, so ids are prefixed with "<dataset tag>::" to make
     them globally unique; qrels are rewritten with the same prefix;
  4. chunked retrieval: the corpus is streamed from disk to the GPU in chunks
     with a running top-k, so very large corpora need a single pass and never
     have to fit in GPU memory at once;
  5. metrics: Recall@k / nDCG@k / MRR@k / MAP@k / Precision@k / Success@k.
"""

import json
import os
from typing import Iterable, Optional

import numpy as np
import torch

# Namespace separator of global ids; chosen to almost never occur in raw ids.
NS_SEP = "::"


# =============================================================================
# global id helpers
# =============================================================================

def make_global_id(namespace: Optional[str], raw_id: Optional[str]) -> Optional[str]:
    """Turn a raw id into a globally unique id. Returned unchanged if namespace is empty."""
    if raw_id is None:
        return None

    raw_id = str(raw_id)

    if not namespace:
        return raw_id

    return f"{namespace}{NS_SEP}{raw_id}"


def split_global_id(global_id: str) -> tuple[Optional[str], str]:
    """Global id -> (namespace, raw id). namespace is None if there is no prefix."""
    if NS_SEP in global_id:
        namespace, raw_id = global_id.split(NS_SEP, 1)
        return namespace, raw_id

    return None, global_id


# =============================================================================
# embedding directory reader
# =============================================================================

class EmbeddingSet:
    """
    One embedding output directory (used for both queries and corpus).

    Layout (see merge_shards in embed_jsonl_unified_multigpu.py):
      - embeddings.npy       (num_records, dim); failed rows are all zeros
      - ids.txt              one id per line, aligned with the rows of embeddings.npy
      - token_counts.jsonl   one line per row with a `success` field (optional)
    """

    def __init__(
        self,
        directory: str,
        namespace: Optional[str] = None,
        name: Optional[str] = None,
    ):
        self.directory = directory
        self.namespace = namespace
        self.name = name or os.path.basename(directory.rstrip("/"))

        self.embeddings_path = os.path.join(directory, "embeddings.npy")
        ids_path = os.path.join(directory, "ids.txt")

        if not os.path.isfile(self.embeddings_path):
            raise FileNotFoundError(f"missing embeddings.npy: {self.embeddings_path}")

        if not os.path.isfile(ids_path):
            raise FileNotFoundError(f"missing ids.txt: {ids_path}")

        # Open with mmap instead of loading the whole matrix into memory.
        self.embeddings = np.load(self.embeddings_path, mmap_mode="r")

        if self.embeddings.ndim != 2:
            raise ValueError(
                f"{self.embeddings_path} must be a 2-d array, got {self.embeddings.shape}"
            )

        self.num_rows, self.dim = self.embeddings.shape

        raw_ids: list[Optional[str]] = []

        with open(ids_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.rstrip("\n")
                raw_ids.append(line if line else None)

        if len(raw_ids) != self.num_rows:
            raise ValueError(
                f"{directory}: number of lines in ids.txt ({len(raw_ids)}) does not match "
                f"the number of rows of embeddings.npy ({self.num_rows})"
            )

        self.raw_ids = raw_ids
        self.ids = [make_global_id(namespace, rid) for rid in raw_ids]

        self.success_mask = self._load_success_mask()

    # -- internal -----------------------------------------------------------

    def _load_success_mask(self) -> np.ndarray:
        """
        Mark which rows have a usable embedding.

        Uses the `success` field of token_counts.jsonl if available; otherwise
        falls back to "vector norm != 0". Rows with an empty id are always unusable.
        """
        mask = np.ones(self.num_rows, dtype=bool)

        token_counts_path = os.path.join(self.directory, "token_counts.jsonl")

        if os.path.isfile(token_counts_path):
            with open(token_counts_path, "r", encoding="utf-8") as file:
                for line_number, line in enumerate(file):
                    line = line.strip()

                    if not line:
                        continue

                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    idx = item.get("idx", line_number)

                    if not isinstance(idx, int) or not (0 <= idx < self.num_rows):
                        continue

                    if item.get("success") is False:
                        mask[idx] = False
        else:
            # No token_counts.jsonl: check for zero vectors chunk by chunk.
            chunk = 100000

            for start in range(0, self.num_rows, chunk):
                end = min(start + chunk, self.num_rows)
                block = np.asarray(self.embeddings[start:end], dtype=np.float32)
                norms = np.linalg.norm(block, axis=1)
                mask[start:end] = norms > 0

        for idx, rid in enumerate(self.ids):
            if rid is None:
                mask[idx] = False

        return mask

    # -- public -------------------------------------------------------------

    @property
    def num_usable(self) -> int:
        return int(self.success_mask.sum())

    @property
    def num_failed(self) -> int:
        return int(self.num_rows - self.success_mask.sum())

    def load_rows(self, rows: np.ndarray) -> np.ndarray:
        """Fetch embeddings (float32) by row index; row indices must be ascending."""
        return np.asarray(self.embeddings[rows], dtype=np.float32)

    def describe(self) -> dict:
        return {
            "dir": self.directory,
            "namespace": self.namespace,
            "num_rows": self.num_rows,
            "num_usable": self.num_usable,
            "num_failed": self.num_failed,
            "dim": self.dim,
        }


def l2_normalize(matrix: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(matrix.float(), p=2, dim=-1)


# =============================================================================
# qrels parsing
# =============================================================================

QUERY_ID_KEYS = (
    "query_id", "query-id", "qid", "q_id", "queryid",
    "question_id", "query", "question",
)

DOC_ID_KEYS = (
    "doc_id", "doc-id", "docid", "corpus-id", "corpus_id", "cid",
    "pid", "passage_id", "document_id", "did", "doc",
)

DOC_ID_LIST_KEYS = (
    "doc_ids", "docids", "corpus_ids", "positive_ids", "positive_doc_ids",
    "positives", "relevant_docs", "relevant_doc_ids", "gold_ids", "gt_ids",
    "answer_doc_ids", "pos_ids",
)

SCORE_KEYS = ("score", "relevance", "rel", "label", "grade", "gain")


def _first_present(obj: dict, keys: Iterable[str]):
    for key in keys:
        if key in obj and obj[key] is not None:
            return obj[key]

    return None


def load_qrels(
    path: str,
    query_namespace: Optional[str] = None,
    doc_namespace: Optional[str] = None,
) -> dict[str, dict[str, float]]:
    """
    Parse qrels.jsonl into {query_global_id: {doc_global_id: relevance}}.

    Supported formats:
        {"query_id": "q1", "doc_id": "d1", "score": 1}
        {"qid": "q1", "corpus-id": "d1"}                       # relevance defaults to 1
        {"query_id": "q1", "positive_ids": ["d1", "d2"]}
        {"query_id": "q1", "docs": {"d1": 2, "d2": 1}}
        q1\td1\t1                                              # TSV fallback
    """
    qrels: dict[str, dict[str, float]] = {}

    def add(raw_qid, raw_did, score: float):
        qid = make_global_id(query_namespace, str(raw_qid))
        did = make_global_id(doc_namespace, str(raw_did))
        bucket = qrels.setdefault(qid, {})
        # Keep the higher relevance if the same (q, d) appears several times.
        bucket[did] = max(bucket.get(did, 0.0), float(score))

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            if not line.startswith("{"):
                # TSV / whitespace-separated fallback: qid docid [score]
                parts = line.split()

                if len(parts) >= 2:
                    score = float(parts[2]) if len(parts) >= 3 else 1.0
                    add(parts[0], parts[1], score)
                    continue

                raise ValueError(f"{path}:{line_number} cannot be parsed: {line[:120]}")

            obj = json.loads(line)

            if not isinstance(obj, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")

            raw_qid = _first_present(obj, QUERY_ID_KEYS)

            if raw_qid is None:
                raise KeyError(
                    f"{path}:{line_number} has no query id field; "
                    f"supported keys: {QUERY_ID_KEYS}"
                )

            default_score = _first_present(obj, SCORE_KEYS)
            default_score = 1.0 if default_score is None else float(default_score)

            handled = False

            # form 3: {"docs": {"d1": 2}} / {"qrels": {...}}
            for key in ("docs", "qrels", "relevance", "doc_scores"):
                value = obj.get(key)

                if isinstance(value, dict):
                    for raw_did, score in value.items():
                        add(raw_qid, raw_did, float(score))

                    handled = True

            # form 2: list
            raw_list = _first_present(obj, DOC_ID_LIST_KEYS)

            if isinstance(raw_list, (list, tuple)):
                for item in raw_list:
                    if isinstance(item, dict):
                        raw_did = _first_present(item, DOC_ID_KEYS)
                        score = _first_present(item, SCORE_KEYS)
                        score = default_score if score is None else float(score)

                        if raw_did is not None:
                            add(raw_qid, raw_did, score)
                    else:
                        add(raw_qid, item, default_score)

                handled = True

            # form 1: single entry
            raw_did = _first_present(obj, DOC_ID_KEYS)

            if raw_did is not None and not isinstance(raw_did, (list, tuple, dict)):
                add(raw_qid, raw_did, default_score)
                handled = True

            if not handled:
                raise KeyError(
                    f"{path}:{line_number} has no doc id field; "
                    f"supported keys: {DOC_ID_KEYS} / {DOC_ID_LIST_KEYS}"
                )

    return qrels


# =============================================================================
# corpus index (several embedding directories can form one large corpus)
# =============================================================================

class CorpusIndex:
    """
    Concatenate one or more EmbeddingSets into one logical corpus.

    - only successful rows are kept;
    - the row -> global doc id mapping is stored in self.doc_ids;
    - a doc id may occupy several rows (duplicates within a dataset); retrieval
      results are deduplicated by doc id, keeping the best rank.
    """

    def __init__(self, shards: list[EmbeddingSet], chunk_size: int = 20000):
        if not shards:
            raise ValueError("CorpusIndex needs at least one EmbeddingSet")

        dims = {shard.dim for shard in shards}

        if len(dims) != 1:
            raise ValueError(f"corpus shards have different dimensions: {dims}")

        self.dim = dims.pop()
        self.shards = shards
        self.chunk_size = chunk_size

        self.doc_ids: list[str] = []
        self._blocks: list[tuple[EmbeddingSet, np.ndarray, int]] = []

        offset = 0

        for shard in shards:
            valid_rows = np.where(shard.success_mask)[0]

            for start in range(0, len(valid_rows), chunk_size):
                rows = valid_rows[start:start + chunk_size]
                self._blocks.append((shard, rows, offset))

                for row in rows.tolist():
                    self.doc_ids.append(shard.ids[row])

                offset += len(rows)

        self.num_rows = offset

        if self.num_rows == 0:
            raise RuntimeError("the corpus has no usable embeddings (did all embeddings fail?)")

        # doc id -> number of occurrences; decides how many extra rows top-k must fetch to get k unique docs.
        counts: dict[str, int] = {}

        for doc_id in self.doc_ids:
            counts[doc_id] = counts.get(doc_id, 0) + 1

        self.num_unique_docs = len(counts)
        self.max_duplicate = max(counts.values())
        self.duplicated_doc_ids = self.num_rows - self.num_unique_docs

    def id_set(self) -> set[str]:
        return set(self.doc_ids)

    def iter_chunks(self, device: torch.device):
        """Yield (normalized embeddings, start row of the chunk in the global corpus)."""
        for shard, rows, offset in self._blocks:
            block = shard.load_rows(rows)
            tensor = torch.from_numpy(block).to(device=device, non_blocking=True)
            yield l2_normalize(tensor), offset

    def describe(self) -> dict:
        return {
            "num_rows": self.num_rows,
            "num_unique_docs": self.num_unique_docs,
            "duplicated_rows": self.duplicated_doc_ids,
            "dim": self.dim,
            "shards": [shard.describe() for shard in self.shards],
        }


# =============================================================================
# retrieval
# =============================================================================

def search_topk(
    query_embeddings: torch.Tensor,
    corpus: CorpusIndex,
    topk: int,
    device: torch.device,
    query_batch_size: int = 1024,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cosine-similarity retrieval. The corpus is read from disk once with a running top-k.

    Returns (scores, rows), both (num_queries, topk); rows are global CorpusIndex
    row indices, padded with -1 when fewer than topk are available.
    """
    num_queries = query_embeddings.shape[0]
    topk = int(min(topk, corpus.num_rows))

    best_scores = torch.full(
        (num_queries, topk), float("-inf"), device=device, dtype=torch.float32,
    )
    best_rows = torch.full(
        (num_queries, topk), -1, device=device, dtype=torch.long,
    )

    num_blocks = len(corpus._blocks)

    for block_idx, (chunk, offset) in enumerate(corpus.iter_chunks(device)):
        chunk_t = chunk.t().contiguous()

        for start in range(0, num_queries, query_batch_size):
            end = min(start + query_batch_size, num_queries)
            scores = query_embeddings[start:end] @ chunk_t

            k = int(min(topk, scores.shape[1]))
            chunk_scores, chunk_idx = torch.topk(scores, k, dim=1)
            chunk_idx = chunk_idx + offset

            merged_scores = torch.cat([best_scores[start:end], chunk_scores], dim=1)
            merged_rows = torch.cat([best_rows[start:end], chunk_idx], dim=1)

            new_scores, positions = torch.topk(merged_scores, topk, dim=1)
            best_scores[start:end] = new_scores
            best_rows[start:end] = torch.gather(merged_rows, 1, positions)

        del chunk, chunk_t

        if verbose and (block_idx + 1) % 20 == 0:
            print(
                f"    [search] corpus block {block_idx + 1}/{num_blocks}",
                flush=True,
            )

    return best_scores.cpu().numpy(), best_rows.cpu().numpy()


def rows_to_ranked_ids(
    rows: np.ndarray,
    corpus: CorpusIndex,
    max_k: int,
) -> list[list[str]]:
    """Convert retrieved rows into a deduplicated, ranked list of doc ids (best rank per doc)."""
    ranked: list[list[str]] = []

    for row_list in rows:
        seen: set[str] = set()
        ordered: list[str] = []

        for row in row_list.tolist():
            if row < 0:
                continue

            doc_id = corpus.doc_ids[row]

            if doc_id in seen:
                continue

            seen.add(doc_id)
            ordered.append(doc_id)

            if len(ordered) >= max_k:
                break

        ranked.append(ordered)

    return ranked


# =============================================================================
# metrics
# =============================================================================

def compute_metrics(
    ranked_ids: list[list[str]],
    gold: list[dict[str, float]],
    k_values: list[int],
) -> dict[str, float]:
    """
    ranked_ids[i]: retrieval results of query i (doc ids, deduplicated, by descending score)
    gold[i]:       {doc id: relevance} of query i; only relevance > 0 counts as relevant

    Metrics: Recall@k / Precision@k / nDCG@k / MRR@k / MAP@k / Success@k (hit rate)
    """
    if len(ranked_ids) != len(gold):
        raise ValueError("ranked_ids and gold have different lengths")

    num_queries = len(ranked_ids)
    sums = {
        f"{name}@{k}": 0.0
        for k in k_values
        for name in ("recall", "precision", "ndcg", "mrr", "map", "success")
    }

    if num_queries == 0:
        return {key: 0.0 for key in sums}

    discounts = np.log2(np.arange(2, max(k_values) + 2))

    for ranked, relevant in zip(ranked_ids, gold):
        positives = {doc for doc, rel in relevant.items() if rel > 0}
        num_positives = len(positives)

        ideal_gains = sorted(
            (rel for rel in relevant.values() if rel > 0), reverse=True,
        )

        for k in k_values:
            topk = ranked[:k]

            hits = 0
            dcg = 0.0
            average_precision = 0.0
            reciprocal_rank = 0.0

            for rank, doc_id in enumerate(topk):
                relevance = relevant.get(doc_id, 0.0)

                if relevance > 0:
                    hits += 1
                    dcg += relevance / discounts[rank]
                    average_precision += hits / (rank + 1)

                    if reciprocal_rank == 0.0:
                        reciprocal_rank = 1.0 / (rank + 1)

            idcg = sum(
                gain / discounts[rank]
                for rank, gain in enumerate(ideal_gains[:k])
            )

            sums[f"recall@{k}"] += hits / num_positives if num_positives else 0.0
            sums[f"precision@{k}"] += hits / k
            sums[f"ndcg@{k}"] += dcg / idcg if idcg > 0 else 0.0
            sums[f"mrr@{k}"] += reciprocal_rank
            sums[f"success@{k}"] += 1.0 if hits > 0 else 0.0

            denominator = min(num_positives, k)
            sums[f"map@{k}"] += (
                average_precision / denominator if denominator else 0.0
            )

    return {key: float(value / num_queries) for key, value in sums.items()}


# =============================================================================
# end-to-end evaluation of one dataset (shared by per-dataset and Mix evaluation)
# =============================================================================

def evaluate_queries(
    queries: EmbeddingSet,
    corpus: CorpusIndex,
    qrels: dict[str, dict[str, float]],
    k_values: list[int],
    device: torch.device,
    query_batch_size: int = 1024,
    drop_missing_gold: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Retrieve and score all queries that have qrels and a successful embedding.

    With drop_missing_gold=True, gold docs that do not exist in the corpus are
    removed (queries left without gold are skipped). The default False keeps them
    in the denominator, which lowers recall but avoids inflated metrics.
    """
    max_k = max(k_values)

    corpus_ids = corpus.id_set()

    selected_rows: list[int] = []
    selected_qids: list[str] = []
    selected_gold: list[dict[str, float]] = []

    num_no_qrels = 0
    num_failed_embedding = 0
    num_dropped_no_gold = 0
    num_missing_gold_docs = 0
    num_gold_docs = 0

    for row in range(queries.num_rows):
        query_id = queries.ids[row]

        if query_id is None:
            num_failed_embedding += 1
            continue

        relevant = qrels.get(query_id)

        if not relevant:
            num_no_qrels += 1
            continue

        if not queries.success_mask[row]:
            num_failed_embedding += 1
            continue

        kept: dict[str, float] = {}

        for doc_id, relevance in relevant.items():
            num_gold_docs += 1

            if doc_id not in corpus_ids:
                num_missing_gold_docs += 1

                if drop_missing_gold:
                    continue

            kept[doc_id] = relevance

        if not kept:
            num_dropped_no_gold += 1
            continue

        selected_rows.append(row)
        selected_qids.append(query_id)
        selected_gold.append(kept)

    if not selected_rows:
        raise RuntimeError(
            f"{queries.directory}: no query to evaluate"
            f"(no_qrels={num_no_qrels}, failed_embedding={num_failed_embedding})"
        )

    rows_array = np.asarray(selected_rows, dtype=np.int64)
    query_matrix = torch.from_numpy(queries.load_rows(rows_array))
    query_matrix = l2_normalize(query_matrix.to(device))

    # A doc id may occupy several corpus rows; fetch extra rows so max_k unique docs remain after dedup.
    retrieve_k = min(corpus.num_rows, max_k * max(1, corpus.max_duplicate))

    scores, hit_rows = search_topk(
        query_embeddings=query_matrix,
        corpus=corpus,
        topk=retrieve_k,
        device=device,
        query_batch_size=query_batch_size,
        verbose=verbose,
    )

    ranked_ids = rows_to_ranked_ids(hit_rows, corpus, max_k=max_k)
    metrics = compute_metrics(ranked_ids, selected_gold, k_values)

    return {
        "metrics": metrics,
        "num_evaluated_queries": len(selected_rows),
        "stats": {
            "queries_total_rows": queries.num_rows,
            "queries_without_qrels": num_no_qrels,
            "queries_embedding_failed": num_failed_embedding,
            "queries_dropped_no_gold_in_corpus": num_dropped_no_gold,
            "gold_pairs_total": num_gold_docs,
            "gold_docs_missing_in_corpus": num_missing_gold_docs,
            "drop_missing_gold": drop_missing_gold,
            "corpus_rows": corpus.num_rows,
            "corpus_unique_docs": corpus.num_unique_docs,
            "retrieve_k_rows": int(retrieve_k),
        },
        "_ranked_ids": ranked_ids,
        "_query_ids": selected_qids,
        "_scores": scores,
    }


def dump_run_file(
    path: str,
    query_ids: list[str],
    ranked_ids: list[list[str]],
    top_n: int = 10,
):
    """Write retrieval results to disk for case studies. Ids are written as raw id + source dataset."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        for query_id, ranked in zip(query_ids, ranked_ids):
            query_ns, raw_qid = split_global_id(query_id)
            items = []

            for rank, doc_id in enumerate(ranked[:top_n], start=1):
                doc_ns, raw_did = split_global_id(doc_id)
                items.append(
                    {
                        "rank": rank,
                        "doc_id": raw_did,
                        "doc_dataset": doc_ns,
                    }
                )

            file.write(
                json.dumps(
                    {
                        "query_id": raw_qid,
                        "query_dataset": query_ns,
                        "retrieved": items,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def resolve_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA is not available; falling back to CPU")
        return torch.device("cpu")

    return torch.device(device)


def parse_k_values(text: str) -> list[int]:
    values = sorted({int(item) for item in text.split(",") if item.strip()})

    if not values or values[0] <= 0:
        raise ValueError(f"invalid k_values: {text}")

    return values


def format_metric_table(rows: list[tuple[str, dict[str, float]]], k_values: list[int]) -> str:
    """Print [(name, metrics)] as an aligned table."""
    metric_names = ["recall", "ndcg", "mrr", "map", "precision", "success"]
    headers = ["dataset"] + [
        f"{name}@{k}" for name in metric_names for k in k_values
    ]

    table = [headers]

    for name, metrics in rows:
        line = [name] + [
            f"{metrics.get(f'{metric}@{k}', 0.0):.4f}"
            for metric in metric_names
            for k in k_values
        ]
        table.append(line)

    widths = [
        max(len(row[col]) for row in table) for col in range(len(headers))
    ]

    lines = []

    for row_idx, row in enumerate(table):
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))

        if row_idx == 0:
            lines.append("  ".join("-" * width for width in widths))

    return "\n".join(lines)
