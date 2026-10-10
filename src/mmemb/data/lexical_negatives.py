"""Mine hard negatives by **lexical overlap of the positive text slot**, with a
difficulty curriculum that follows training progress.

Note: this module is an optional extension and is **disabled** in all configs
used for the paper (`data.hard_negatives.enable: false`).

## Idea and risks

In a 1x3 layout, view #0 of `positive` is the text-only document (assuming
`view_strategy` is `first` / `chunk` and views are ordered consistently in every
row). We can therefore compute offline "which rows have the most similar text
document" and feed them as hard negatives, moving from "somewhat similar" to
"most similar" as training progresses.

**The main risk is introducing false negatives, not lack of effect.** The most
lexically similar documents are exactly the ones most likely to be truly
relevant (paraphrases, different descriptions of the same entity). This is a
well-known issue in dense retrieval (e.g. RocketQA denoises mined hard
negatives). `loss.modules.false_negative` masks **by id**, so a paraphrase with
a different doc_id cannot be recognized and masked.

Three safeguards are therefore applied:

  1. **Upper cut-off** `max_similarity`: pairs above it are treated as likely
     duplicates and are not used as negatives (better dropped than training in
     the wrong direction).
  2. **Relation exclusion**: reuse `RelationIndex`; any doc_id known to be
     related to the query is skipped (same criterion as `data.avoid_related_negatives`).
  3. **Lower cut-off** `min_similarity`: pairs below it are not "hard" and are
     left to random negatives.

Safeguard 1 needs tuning: inspect mined pairs with `describe_pairs()` before
choosing `max_similarity`.

**Limitations of the upper cut-off.** Lexical overlap measures what two texts
share, not how they differ. For example:

    row A  "yellow metal ring pendant, 3 cm diameter"
    row B  "yellow metal ring pendant, 5 cm diameter"   <- same item, different size: likely a false negative
    row C  "green metal ring pendant, 3 cm diameter"    <- different color: a desired hard negative

B and C have almost identical similarity to A because they share the same
number of tokens with A; they differ only in the one unshared token. Hence
`max_similarity` can only remove pairs with abnormally high overall overlap and
**cannot separate "same item, different spec" from "same category, different
attribute"**.

Conclusion: the reliable safeguard is #2 (id / relation exclusion), which
requires sufficiently complete `doc_id`s. On data with sparse ids, always use
this together with `loss.modules.false_negative`.

## Similarity

idf-weighted lexical cosine (BM25-like ranking, computed offline once and cacheable):

    score(a, b) = Σ_{t ∈ a∩b} idf(t)² / (‖a‖ · ‖b‖)

Chinese text is split into character bigrams and English / digits into words,
so mixed-language corpora need no external tokenizer. Tokens whose document
frequency exceeds `max_df_ratio` (stop words, template words) are dropped;
otherwise templated corpora would make every document look similar.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_WORD = re.compile(r"[a-zA-Z0-9_]+")


def _log_cache_warning(msg: str, *args: Any) -> None:
    """This module deliberately avoids depending on torch; logging is imported lazily and falls back to print."""
    try:
        from ..utils.misc import get_logger

        get_logger(__name__).warning(msg, *args)
    except Exception:  # noqa: BLE001
        print("[lexical_negatives] " + (msg % args if args else msg))


def _dist_state() -> Tuple[int, int, Callable[[], None]]:
    """Return (rank, world_size, barrier); (0, 1, noop) without torch / initialized distributed."""
    try:
        from ..utils.dist import barrier, get_rank, get_world_size

        return get_rank(), get_world_size(), barrier
    except Exception:  # noqa: BLE001
        return 0, 1, (lambda: None)


def tokenize(text: str) -> List[str]:
    """Chinese character bigrams + English / digit words. No tokenizer dependency."""
    if not text:
        return []
    text = text.lower()
    toks: List[str] = _WORD.findall(text)
    cjk = _CJK.findall(text)
    # Bigrams only within runs of CJK characters: split on non-CJK first, then slide a window
    for seg in re.split(r"[^\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+", text):
        if len(seg) == 1:
            toks.append(seg)
        for i in range(len(seg) - 1):
            toks.append(seg[i : i + 2])
    if not toks and cjk:
        toks = cjk
    return toks


@dataclass
class MinedNegative:
    row: int
    score: float


class LexicalNegativeIndex:
    """Offline mining: for each row, find the most textually similar rows (filtering likely duplicates / known related).

    Parameters
    ----
    texts        text of each row's "text slot" (one per row); empty strings do not participate.
    exclude      per-row sets of row indices that cannot be selected (known related / duplicates).
    top_k        number of candidates kept per row (ranked; consumed by the curriculum).
    min_similarity / max_similarity
                 similarity band: below = not hard enough, above = likely the same document.
    max_df_ratio tokens with document frequency above this ratio are dropped (stop / template words).
    """

    def __init__(
        self,
        texts: Sequence[str],
        exclude: Optional[Sequence[Iterable[int]]] = None,
        top_k: int = 50,
        min_similarity: float = 0.05,
        max_similarity: float = 0.9,
        max_df_ratio: float = 0.3,
        max_postings: int = 20000,
        similarity: str = "f1",
        block_size: int = 0,
        block_assign: str = "signature",
        block_rounds: int = 1,
        query_chunk: int = 2048,
        min_df: int = 2,
        backend: str = "auto",
        seed: int = 0,
    ) -> None:
        if similarity not in ("f1", "cosine"):
            raise ValueError("similarity must be f1 or cosine")
        if block_assign not in ("signature", "random", "none"):
            raise ValueError("block_assign must be signature / random / none")
        self.n = len(texts)
        self.top_k = int(top_k)
        self.min_similarity = float(min_similarity)
        self.max_similarity = float(max_similarity)
        self.max_df_ratio = float(max_df_ratio)
        self.max_postings = int(max_postings)
        self.similarity = similarity
        self.block_size = int(block_size)
        self.block_assign = block_assign
        self.block_rounds = max(1, int(block_rounds))
        self.query_chunk = max(1, int(query_chunk))
        self.min_df = max(1, int(min_df))
        self.backend = backend
        self.seed = int(seed)
        self._texts = list(texts)
        self._exclude = [set(e) for e in (exclude or [set() for _ in texts])]
        self.neighbors: List[List[MinedNegative]] = []
        self.stats: Dict[str, float] = {}

    # ------------------------------------------------------------------
    def build(self) -> "LexicalNegativeIndex":
        """Mining entry: vectorized block-wise path with numpy/scipy, otherwise pure Python."""
        use_fast = self.backend in ("auto", "scipy")
        if use_fast:
            try:
                import numpy  # noqa: F401
                import scipy.sparse  # noqa: F401
            except Exception:
                if self.backend == "scipy":
                    raise
                use_fast = False
        if use_fast:
            return self._build_vectorized()
        return self._build_python()

    # ------------------------------------------------------------------
    def _build_python(self) -> "LexicalNegativeIndex":
        tokens = [set(tokenize(t)) for t in self._texts]
        df: Dict[str, int] = defaultdict(int)
        for ts in tokens:
            for t in ts:
                df[t] += 1

        n_docs = max(1, sum(1 for t in self._texts if t))
        # Absolute floor of 20: on small corpora ratio*n can be so small that discriminative tokens vanish
        # (n=7, ratio=0.3 gives cut=2, removing keywords such as color names)
        df_cut = max(20, int(self.max_df_ratio * n_docs))
        idf = {
            t: math.log(1.0 + n_docs / d)
            for t, d in df.items()
            if d <= df_cut  # overly common tokens are excluded from the inverted index
        }

        postings: Dict[str, List[int]] = defaultdict(list)
        norms: List[float] = [0.0] * self.n
        kept: List[List[str]] = []
        for i, ts in enumerate(tokens):
            keep = [t for t in ts if t in idf]
            kept.append(keep)
            norms[i] = math.sqrt(sum(idf[t] ** 2 for t in keep)) or 1.0
            for t in keep:
                if len(postings[t]) < self.max_postings:
                    postings[t].append(i)

        n_dropped_dup = 0
        n_dropped_rel = 0
        for i in range(self.n):
            if not kept[i]:
                self.neighbors.append([])
                continue
            acc: Dict[int, float] = defaultdict(float)
            for t in kept[i]:
                w = idf[t] ** 2
                for j in postings[t]:
                    if j != i:
                        acc[j] += w
            scored: List[MinedNegative] = []
            for j, s in acc.items():
                if self.similarity == "f1":
                    # weighted F1 / Dice: 2·w(A∩B) / (w(A)+w(B)), w(t)=idf(t)²
                    sim = 2.0 * s / (norms[i] ** 2 + norms[j] ** 2)
                else:
                    sim = s / (norms[i] * norms[j])
                if sim < self.min_similarity:
                    continue
                if sim > self.max_similarity:
                    n_dropped_dup += 1      # likely the same document: skip
                    continue
                if j in self._exclude[i]:
                    n_dropped_rel += 1      # relation table says it is related to this query
                    continue
                scored.append(MinedNegative(row=j, score=sim))
            scored.sort(key=lambda m: -m.score)
            self.neighbors.append(scored[: self.top_k])

        covered = sum(1 for ns in self.neighbors if ns)
        self.stats = {
            "n_rows": float(self.n),
            "n_rows_with_candidates": float(covered),
            "coverage": covered / max(1, self.n),
            "mean_candidates": sum(len(ns) for ns in self.neighbors) / max(1, self.n),
            "dropped_near_duplicate": float(n_dropped_dup),
            "dropped_related": float(n_dropped_rel),
            "vocab_after_df_cut": float(len(idf)),
        }
        return self

    # ------------------------------------------------------------------
    # Vectorized + block-wise mining
    # ------------------------------------------------------------------
    def _build_matrix(self):
        """Build a CSR sparse matrix X[n, V] with X[i,t] = idf(t) if token t occurs in row i.

        Then entry (i,j) of `X @ X.T` equals Σ_{t∈A∩B} idf(t)², from which both
        similarities follow with a row-sum / row-norm vector:

            cosine(i,j) = dot / (‖i‖·‖j‖)
            f1(i,j)     = 2·dot / (Σᵢidf² + Σⱼidf²)      (weighted Dice / F1)

        Two pruning steps make the matmul fast (results unchanged or negligibly changed):
          * tokens with df > df_cut are dropped (stop / template words);
          * tokens with df < min_df (default 2) are **not put in the matrix**: they occur
            in a single document, contribute 0 to any i≠j dot product, but would be
            scanned repeatedly. They still count in the denominators (norm / rowsum),
            so scores are unchanged.
        """
        import numpy as np
        from scipy import sparse

        tokens = [set(tokenize(t)) for t in self._texts]
        df: Dict[str, int] = defaultdict(int)
        for ts in tokens:
            for t in ts:
                df[t] += 1

        n_docs = max(1, sum(1 for t in self._texts if t))
        df_cut = max(20, int(self.max_df_ratio * n_docs))
        idf = {t: math.log(1.0 + n_docs / d) for t, d in df.items() if d <= df_cut}
        # Only tokens with df >= min_df need to be in the matrix (others contribute 0 for i≠j)
        vocab: Dict[str, int] = {}
        for t in idf:
            if df[t] >= self.min_df:
                vocab[t] = len(vocab)

        indptr = np.zeros(self.n + 1, dtype=np.int64)
        indices: List[int] = []
        data: List[float] = []
        weight = np.zeros(self.n, dtype=np.float64)  # Σ idf² (including tokens not in the matrix)
        for i, ts in enumerate(tokens):
            w = 0.0
            for t in ts:
                v = idf.get(t)
                if v is None:
                    continue
                w += v * v
                col = vocab.get(t)
                if col is not None:
                    indices.append(col)
                    data.append(v)
            weight[i] = w
            indptr[i + 1] = len(indices)

        X = sparse.csr_matrix(
            (
                np.asarray(data, dtype=np.float32),
                np.asarray(indices, dtype=np.int32),
                indptr,
            ),
            shape=(self.n, max(1, len(vocab))),
        )
        return X, weight, len(idf)

    def _blocks(self, active, rank: int = 0) -> List[Any]:
        """Split participating rows into blocks; pairs are scored only within a block.

        Cost drops from O(N²) (strictly O(N x mean posting length)) to O(N x block_size).
        `block_assign`:
          * signature: sort by the rarest token, then chunk; documents sharing rare
            tokens land in the same block, so dropped pairs are mostly dissimilar anyway;
          * random: random chunks; unbiased but slightly lower recall;
          * none / block_size<=0: no blocking (exhaustive, most exact and slowest).
        """
        import numpy as np

        rows = np.asarray(active, dtype=np.int64)
        bs = self.block_size
        if bs <= 0 or self.block_assign == "none" or len(rows) <= bs:
            return [rows]
        rng = np.random.default_rng(self.seed)
        if self.block_assign == "random":
            rng.shuffle(rows)
        else:
            sig = self._signature(rows, rank)
            order = np.lexsort((rng.random(len(rows)), sig))
            rows = rows[order]
        return [rows[s : s + bs] for s in range(0, len(rows), bs)]

    def _signature(self, rows, rank: int = 0):
        """Stable hash of each row's `rank`-th rarest token, used to group similar documents.

        `rank` increases with `block_rounds`: round 2 re-blocks by the 2nd rarest token,
        so different rounds use genuinely different blockings and recall accumulates
        (similar to multiple minhash bands).
        """
        import numpy as np

        df: Dict[str, int] = defaultdict(int)
        toks = [set(tokenize(self._texts[i])) for i in rows]
        for ts in toks:
            for t in ts:
                df[t] += 1
        out = np.zeros(len(rows), dtype=np.uint64)
        for k, ts in enumerate(toks):
            if not ts:
                continue
            order = sorted(ts, key=lambda t: (df[t], t))
            rare = order[min(rank, len(order) - 1)]
            out[k] = int(hashlib.blake2b(rare.encode(), digest_size=8).hexdigest(), 16)
        return out

    def _build_vectorized(self) -> "LexicalNegativeIndex":
        import numpy as np

        X, weight, vocab_size = self._build_matrix()
        norms = np.sqrt(np.maximum(weight, 1e-12))
        active = [i for i in range(self.n) if X.indptr[i + 1] > X.indptr[i]]

        # One result bucket per row; rounds are merged, so deduplicate with a dict first
        best: List[Dict[int, float]] = [dict() for _ in range(self.n)]
        n_dup = 0
        n_rel = 0
        seeds = [self.seed + r * 7919 for r in range(self.block_rounds)]
        for r, sd in enumerate(seeds):
            saved, self.seed = self.seed, sd
            blocks = self._blocks(active, rank=r)
            self.seed = saved
            for blk in blocks:
                d, rl = self._mine_block(X, weight, norms, blk, best)
                n_dup += d
                n_rel += rl

        self.neighbors = []
        for i in range(self.n):
            items = [MinedNegative(row=j, score=float(s)) for j, s in best[i].items()]
            items.sort(key=lambda m: (-m.score, m.row))
            self.neighbors.append(items[: self.top_k])

        self._finish_stats(n_dup, n_rel, vocab_size)
        return self

    def _mine_block(self, X, weight, norms, rows, best) -> Tuple[int, int]:
        """Top-k within a block: Xq @ Xb.T computes intersection weights for the whole block, then vectorized scoring."""
        import numpy as np

        n_dup = n_rel = 0
        if len(rows) < 2:
            return 0, 0
        Xb = X[rows]
        wb = weight[rows]
        nb = norms[rows]
        BT = Xb.T.tocsc()
        # Keep extra candidates to leave room for duplicate / related filtering
        keep = max(self.top_k * 2, self.top_k + 16)
        for s in range(0, len(rows), self.query_chunk):
            qrows = rows[s : s + self.query_chunk]
            S = (X[qrows] @ BT).tocsr()
            for r in range(len(qrows)):
                i = int(qrows[r])
                lo, hi = S.indptr[r], S.indptr[r + 1]
                if lo == hi:
                    continue
                cols = S.indices[lo:hi]
                dot = S.data[lo:hi].astype(np.float64)
                if self.similarity == "f1":
                    sim = 2.0 * dot / np.maximum(weight[i] + wb[cols], 1e-12)
                else:
                    sim = dot / np.maximum(norms[i] * nb[cols], 1e-12)
                j = rows[cols]
                m = (j != i) & (sim >= self.min_similarity)
                n_dup += int(np.count_nonzero(m & (sim > self.max_similarity)))
                m &= sim <= self.max_similarity
                if not m.any():
                    continue
                j = j[m]
                sim = sim[m]
                if len(sim) > keep:  # coarse filter before the (more expensive) exclusion lookup
                    part = np.argpartition(-sim, keep)[:keep]
                    j, sim = j[part], sim[part]
                ex = self._exclude[i]
                if ex:
                    ok = np.fromiter((int(x) not in ex for x in j), dtype=bool, count=len(j))
                    n_rel += int(np.count_nonzero(~ok))
                    j, sim = j[ok], sim[ok]
                if not len(j):
                    continue
                if len(sim) > self.top_k:
                    part = np.argpartition(-sim, self.top_k)[: self.top_k]
                    j, sim = j[part], sim[part]
                bucket = best[i]
                for jj, ss in zip(j.tolist(), sim.tolist()):
                    if ss > bucket.get(jj, -1.0):
                        bucket[jj] = ss
                if len(bucket) > 4 * self.top_k:  # avoid unbounded growth across rounds
                    top = sorted(bucket.items(), key=lambda kv: -kv[1])[: self.top_k]
                    best[i] = dict(top)
        return n_dup, n_rel

    def _finish_stats(self, n_dup: int, n_rel: int, vocab_size: int) -> None:
        covered = sum(1 for ns in self.neighbors if ns)
        self.stats = {
            "n_rows": float(self.n),
            "n_rows_with_candidates": float(covered),
            "coverage": covered / max(1, self.n),
            "mean_candidates": sum(len(ns) for ns in self.neighbors) / max(1, self.n),
            "dropped_near_duplicate": float(n_dup),
            "dropped_related": float(n_rel),
            "vocab_after_df_cut": float(vocab_size),
        }

    # ------------------------------------------------------------------
    def describe_pairs(self, k: int = 5, max_chars: int = 60) -> str:
        """Print the hardest mined pairs; **inspect them before enabling this feature**.

        Check whether any of these "negatives" is actually a correct answer. If so,
        lower max_similarity or fill in missing doc_ids in the data first.
        """
        rows = [
            (i, ns[0].score, ns[0].row)
            for i, ns in enumerate(self.neighbors)
            if ns
        ]
        rows.sort(key=lambda r: -r[1])
        lines = ["Hardest mined negatives (please verify they are truly irrelevant):"]
        for i, score, j in rows[:k]:
            lines.append(f"  sim={score:.3f}")
            lines.append(f"    query row {i}: {self._texts[i][:max_chars]}")
            lines.append(f"    neg   row {j}: {self._texts[j][:max_chars]}")
        return "\n".join(lines)

    def summary(self) -> str:
        s = self.stats
        return (
            f"Lexical hard-negative mining: {int(s['n_rows_with_candidates'])}/{int(s['n_rows'])} rows have candidates "
            f"(coverage {s['coverage']:.1%}, mean {s['mean_candidates']:.1f} per row); "
            f"dropped {int(s['dropped_near_duplicate'])} likely-duplicate pairs, "
            f"dropped {int(s['dropped_related'])} known-related pairs"
        )

    # ------------------------------------------------------------------ cache
    def to_json(self) -> Dict[str, Any]:
        return {
            "neighbors": [[[m.row, round(m.score, 5)] for m in ns] for ns in self.neighbors],
            "stats": self.stats,
        }

    @classmethod
    def from_json(cls, obj: Dict[str, Any], texts: Sequence[str]) -> "LexicalNegativeIndex":
        idx = cls(texts)
        idx.neighbors = [[MinedNegative(int(r), float(s)) for r, s in ns] for ns in obj["neighbors"]]
        idx.stats = dict(obj.get("stats") or {})
        return idx

    @staticmethod
    def cache_key(texts: Sequence[str], params: Dict[str, Any]) -> str:
        h = hashlib.blake2b(digest_size=12)
        h.update(json.dumps(params, sort_keys=True).encode())
        h.update(str(len(texts)).encode())
        for t in texts:
            h.update(t.encode("utf-8", "ignore")[:512])
            h.update(b"\x00")
        return h.hexdigest()

    @classmethod
    def _try_load(
        cls, path: str, texts: Sequence[str]
    ) -> Optional["LexicalNegativeIndex"]:
        """Read the cache; returns None (cache miss) if the file is missing, truncated or malformed.

        This must be fault-tolerant: an interrupted or concurrent write can leave a
        syntactically invalid JSON file. Treating it as a miss (and rebuilding)
        avoids a crash on every subsequent run.
        """
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if not isinstance(obj, dict) or not isinstance(obj.get("neighbors"), list):
                raise ValueError("Malformed cache: missing `neighbors` field")
            if len(obj["neighbors"]) != len(texts):
                raise ValueError(
                    f"cache rows ({len(obj['neighbors'])}) != corpus rows ({len(texts)})"
                )
            return cls.from_json(obj, texts)
        except Exception as e:  # noqa: BLE001 -- any unreadable cache counts as a miss
            _log_cache_warning(
                "Hard-negative cache %s is corrupted or stale (%s); deleting and re-mining. "
                "Common causes: concurrent writes from several processes, or a write killed midway.",
                path,
                e,
            )
            try:
                os.remove(path)
            except OSError:
                pass
            return None

    def _write_cache(self, path: str) -> None:
        """Atomic write: the temp file name includes the pid and a random suffix, so
        **no two processes ever share a temp file**; `os.replace` then publishes it.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.to_json(), f)
                f.flush()
                os.fsync(f.fileno())  # flush to disk before renaming to avoid partial files
            os.replace(tmp, path)  # replace within the same directory is atomic
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise

    @classmethod
    def load_or_build(
        cls, texts: Sequence[str], cache_dir: Optional[str], **kwargs: Any
    ) -> "LexicalNegativeIndex":
        """Mining costs O(N x mean posting length) (tens of seconds for ~100k rows), so it is cached by default.

        The cache key covers all parameters and the corpus content; changing data or
        parameters invalidates it automatically.

        Multi-GPU: only rank 0 mines and writes the cache; other ranks wait at a barrier
        and then read it. This avoids W duplicate computations and concurrent writes.
        Without initialized distributed training (single GPU / direct call), each process
        mines on its own; writes remain atomic.
        """
        params = {k: v for k, v in kwargs.items() if k != "exclude"}
        # The exclusion table must be part of the key: if the corpus is unchanged but
        # auto_positive_ids / relations change, a stale cache must not be reused.
        ex = kwargs.get("exclude")
        if ex is not None:
            h = hashlib.blake2b(digest_size=8)
            for s in ex:
                h.update(",".join(str(x) for x in sorted(s)).encode())
                h.update(b"\x00")
            params["_exclude_fp"] = h.hexdigest()
        if not cache_dir:
            return cls(texts, **kwargs).build()

        key = cls.cache_key(texts, params)
        path = os.path.join(cache_dir, f"lexneg_{key}.json")

        hit = cls._try_load(path, texts)
        if hit is not None:
            return hit

        rank, world, barrier = _dist_state()
        if world > 1 and rank != 0:
            # Non-zero ranks: wait for rank 0, then read; if that fails (rank 0 failed too),
            # mine locally without writing (only rank 0 writes the cache).
            barrier()
            hit = cls._try_load(path, texts)
            if hit is not None:
                return hit
            return cls(texts, **kwargs).build()

        idx = cls(texts, **kwargs).build()
        try:
            idx._write_cache(path)
        except Exception as e:  # noqa: BLE001 -- failing to write the cache must not stop training
            _log_cache_warning("Failed to write hard-negative cache (%s); continuing without cache.", e)
        finally:
            if world > 1:
                barrier()  # release the other ranks (in `finally`, so a failed write never blocks them)
        return idx


# ---------------------------------------------------------------------------
class DifficultySchedule:
    """Map training progress p ∈ [0,1] to a rank window of candidates.

    Candidates are sorted by similarity in descending order (rank 0 = most similar =
    hardest). With window width `window`, the start slides from deep (easy) to 0
    (hardest) as training progresses:

        start(p) = round((K - window) · (1 - ramp(p)))

    p=0 takes the last window (relatively easy); p=1 takes [0, window) (hardest).
    A sliding window (instead of "random among the top N") ensures that different
    stages really sample different difficulty levels.
    """

    def __init__(
        self,
        window: int = 10,
        schedule: str = "linear",
        warmup_ratio: float = 0.0,
        mix_init: float = 0.3,
        mix_final: float = 1.0,
    ) -> None:
        if schedule not in ("linear", "cosine"):
            raise ValueError("schedule must be linear or cosine")
        self.window = max(1, int(window))
        self.schedule = schedule
        self.warmup_ratio = float(warmup_ratio)
        self.mix_init = float(mix_init)
        self.mix_final = float(mix_final)

    def ramp(self, p: float) -> float:
        p = min(1.0, max(0.0, float(p)))
        if self.warmup_ratio > 0:
            if p <= self.warmup_ratio:
                return 0.0
            p = (p - self.warmup_ratio) / max(1e-6, 1.0 - self.warmup_ratio)
        return p if self.schedule == "linear" else 0.5 * (1 - math.cos(math.pi * p))

    def band(self, n_candidates: int, p: float) -> Tuple[int, int]:
        """Return the [start, end) rank interval."""
        if n_candidates <= 0:
            return (0, 0)
        w = min(self.window, n_candidates)
        start = int(round((n_candidates - w) * (1.0 - self.ramp(p))))
        return (start, start + w)

    def mix_ratio(self, p: float) -> float:
        """Fraction of negative slots filled with mined hard negatives at this step (the rest use the original sources)."""
        r = self.ramp(p)
        return self.mix_init + (self.mix_final - self.mix_init) * r
