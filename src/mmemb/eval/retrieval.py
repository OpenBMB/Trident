"""Retrieval evaluation: Recall@K / MRR@K / MeanRank.

Candidate pool = positives of all samples (including multi-view positives) +
negatives provided with the samples (deduplicated). Any dev JSONL with a
`positive` per row can therefore be evaluated directly; providing negatives
makes the pool harder and closer to real ranking.

Multi-view data (several query views / positive views per sample):
  * each query view is a separate query (so text queries and image queries get separate scores);
  * the gold set of a query is **all positive views of its sample**; the best-ranked
    one determines the rank (standard multi-gold retrieval evaluation).
  For single-view data this reduces to standard single-gold evaluation.

A set of **modality-balance diagnostics** (`balance/*`, multi-view only) is also
computed: the dispersion inside each sample's nq x npos cosine block. It is the
same quantity logged by `mmemb/losses/modality_balance.py` during training,
measured on the eval set, so "how modality bias evolves" and "how retrieval
metrics evolve" can be plotted together.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..data.schema import Example, Record
from ..utils.misc import get_logger

logger = get_logger(__name__)


def _doc_key(rec: Record) -> str:
    return rec.uid or ("TXT:" + (rec.text or "") + "|IMG:" + ",".join(rec.images))


def build_corpus(
    examples: Iterable[Example],
) -> Tuple[List[Record], List[Record], List[List[int]]]:
    """Return (queries, corpus, golds); golds[i] lists all correct document indices of query i."""
    queries: List[Record] = []
    corpus: List[Record] = []
    key2idx: Dict[str, int] = {}
    golds: List[List[int]] = []

    def add(rec: Record) -> int:
        k = _doc_key(rec)
        if k not in key2idx:
            key2idx[k] = len(corpus)
            corpus.append(rec)
        return key2idx[k]

    for ex in examples:
        gold = [add(p) for p in ex.positive_views]
        for neg in ex.negatives:
            add(neg)
        for qv in ex.query_views:
            queries.append(qv)
            golds.append(gold)
    return queries, corpus, golds


def example_blocks(
    examples: Iterable[Example],
    golds: Sequence[Sequence[int]],
) -> List[Tuple[List[int], List[int]]]:
    """Recover "query-view row indices / positive-view column indices in the pool" per sample.

    `build_corpus` appends query views in sample order, so walking the samples in the
    same order recovers the block coordinates without changing build_corpus's return value.
    """
    blocks: List[Tuple[List[int], List[int]]] = []
    qi = 0
    for ex in examples:
        n = len(ex.query_views)
        rows = list(range(qi, qi + n))
        qi += n
        if not rows:
            continue
        cols = list(golds[rows[0]])
        if len(rows) * len(cols) >= 2:   # a 1x1 block has no dispersion
            blocks.append((rows, cols))
    return blocks


@torch.no_grad()
def balance_diagnostics(
    q_emb: torch.Tensor,
    d_emb: torch.Tensor,
    blocks: Sequence[Tuple[List[int], List[int]]],
    prefix: str = "balance/",
    chunk: int = 256,
    cell_means: bool = True,
) -> Dict[str, float]:
    """Modality-balance diagnostics on the eval set.

    For each sample, take its own nq x npos cosine block (query views x positive views) and report:

      {prefix}pos_range  mean max-min within blocks -- linear scale, the most direct measure of modality bias
      {prefix}pos_var    mean within-block variance -- same scale as the training-side metric=variance
      {prefix}pos_mean   mean of block means         -- overall level of positive scores
      {prefix}cell_i_j   mean cosine of query view i x positive view j
                         (only when all blocks have the same shape and <= 16 cells)
                         -- shows directly which modality pair scores higher

    Blocks of the same shape are batched; groups of different shapes are computed separately and averaged weighted by block count.
    """
    if not blocks:
        return {}
    qn = F.normalize(q_emb.float(), p=2, dim=-1)
    dn = F.normalize(d_emb.float(), p=2, dim=-1)

    groups: Dict[Tuple[int, int], List[Tuple[List[int], List[int]]]] = {}
    for rows, cols in blocks:
        groups.setdefault((len(rows), len(cols)), []).append((rows, cols))

    tot = 0
    acc = {"range": 0.0, "var": 0.0, "mean": 0.0}
    cells: Optional[torch.Tensor] = None
    uniform_shape = len(groups) == 1

    for (nq, npos), items in groups.items():
        cell_sum = torch.zeros(nq, npos, dtype=torch.float32, device=qn.device)
        for start in range(0, len(items), chunk):
            part = items[start : start + chunk]
            qi = torch.tensor([r for r, _ in part], dtype=torch.long, device=qn.device)
            di = torch.tensor([c for _, c in part], dtype=torch.long, device=dn.device)
            Q = qn[qi.reshape(-1)].view(len(part), nq, -1)
            Dp = dn[di.reshape(-1)].view(len(part), npos, -1)
            S = torch.einsum("mid,mjd->mij", Q, Dp).reshape(len(part), nq * npos)
            mean = S.mean(dim=-1)
            acc["range"] += float((S.amax(dim=-1) - S.amin(dim=-1)).sum().item())
            acc["var"] += float(((S - mean.unsqueeze(-1)) ** 2).mean(dim=-1).sum().item())
            acc["mean"] += float(mean.sum().item())
            cell_sum += S.view(len(part), nq, npos).sum(dim=0)
            tot += len(part)
        if uniform_shape and cell_means and nq * npos <= 16:
            cells = cell_sum / max(len(items), 1)

    out = {
        f"{prefix}pos_range": acc["range"] / max(tot, 1),
        f"{prefix}pos_var": acc["var"] / max(tot, 1),
        f"{prefix}pos_mean": acc["mean"] / max(tot, 1),
        f"{prefix}n_blocks": float(tot),
    }
    if cells is not None:
        for i in range(cells.size(0)):
            for j in range(cells.size(1)):
                out[f"{prefix}cell_{i}_{j}"] = float(cells[i, j].item())
    return out


@torch.no_grad()
def evaluate_retrieval(
    encoder,
    examples: Sequence[Example],
    batch_size: int = 8,
    k_list: Sequence[int] = (1, 5, 10),
    dim: Optional[int] = None,
    device: Optional[str] = None,
    chunk: int = 512,
    balance: bool = True,
) -> Dict[str, float]:
    queries, corpus, golds = build_corpus(examples)
    max_gold = max((len(g) for g in golds), default=1)
    logger.info(
        "Evaluation: %d queries / %d candidate documents (%.2f gold per query on average)",
        len(queries),
        len(corpus),
        sum(len(g) for g in golds) / max(len(golds), 1),
    )

    q_emb = encoder.encode(queries, role="query", batch_size=batch_size, dim=dim)
    d_emb = encoder.encode(corpus, role="doc", batch_size=batch_size, dim=dim)

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    q_emb, d_emb = q_emb.to(dev).float(), d_emb.to(dev).float()

    # Pad golds into a rectangle [Q, max_gold] + mask, so a query may have several correct answers
    gold_idx = torch.zeros(len(golds), max_gold, dtype=torch.long, device=dev)
    gold_mask = torch.zeros(len(golds), max_gold, dtype=torch.bool, device=dev)
    for i, g in enumerate(golds):
        # Pad by repeating the first gold (not 0), otherwise scatter would affect candidate 0
        gold_idx[i, :] = g[0]
        for j, col in enumerate(g):
            gold_idx[i, j] = col
            gold_mask[i, j] = True

    ranks: List[torch.Tensor] = []
    for i in range(0, q_emb.size(0), chunk):
        sims = q_emb[i : i + chunk] @ d_emb.t()
        idx = gold_idx[i : i + chunk]
        msk = gold_mask[i : i + chunk]
        gold_scores = sims.gather(1, idx).masked_fill(~msk, float("-inf"))
        best_gold = gold_scores.max(dim=1, keepdim=True).values  # the easiest-to-hit gold
        # Exclude other golds from "candidates ranked above me" so multiple golds do not push each other down
        better = (sims > best_gold)
        better.scatter_(1, idx, torch.zeros_like(idx, dtype=torch.bool))
        ranks.append(better.sum(dim=1) + 1)  # 1-based rank
    rank = torch.cat(ranks).float()

    metrics: Dict[str, float] = {"mean_rank": rank.mean().item()}
    for k in k_list:
        metrics[f"recall@{k}"] = (rank <= k).float().mean().item()
    max_k = max(k_list)
    rr = torch.where(rank <= max_k, 1.0 / rank, torch.zeros_like(rank))
    metrics[f"mrr@{max_k}"] = rr.mean().item()
    metrics["num_queries"] = float(len(queries))
    metrics["num_docs"] = float(len(corpus))

    # Modality-balance diagnostics: embeddings are already computed, this only costs a few einsums
    if balance:
        try:
            metrics.update(
                balance_diagnostics(q_emb, d_emb, example_blocks(examples, golds))
            )
        except Exception as e:  # a failing diagnostic must not fail the whole evaluation
            logger.warning("Modality-balance diagnostics failed (retrieval metrics unaffected): %s", e)
    return metrics


def collect_examples(dataset) -> List[Example]:
    return [dataset[i] for i in range(len(dataset))]
