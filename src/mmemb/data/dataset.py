"""Dataset implementations.

`ContrastiveJsonlDataset` is responsible for:
  1. reading JSONL files (multiple files and sampling weights are supported);
  2. fixing the number of hard negatives per sample (`num_negatives`), filling
     missing ones according to a strategy;
  3. splitting each sample's query / positive **views** into view groups of a
     custom size **k** (`num_query_views` / `num_positives`, or `multiview_k`
     to set both);
  4. injecting instructions per task (and per modality);
  5. prefixing image paths with `image_root`.

Extension point: to support webdataset / HF datasets / multi-task mixing, write
a new class that returns `Example`s and register it with `@DATASETS.register("name")`.

Multi-view notes (k = block side length; the InfoNCE target block is Nq x Np):
  * `multiview_k: 1` (default) is plain single-view training;
  * `multiview_k: k` is equivalent to `num_query_views = num_positives = k`;
    the two can also be set separately (non-square, e.g. 2 query views x 4 positive views);
  * when a sample has M views and M != k, `view_strategy` decides what happens:

    | view_strategy   | when M > k                          | changes per epoch | views wasted?     |
    |-----------------|-------------------------------------|-------------------|-------------------|
    | `first`         | take the first k                    | no                | yes (drops M-k)   |
    | `shuffle`       | take k at random                    | yes               | yes (k per step)  |
    | `chunk`         | split into ceil(M/k) groups, one    | no                | **no**            |
    |                 | Example per group                   |                   |                   |
    | `shuffle_chunk` | shuffle, then chunk                 | yes               | **no**            |

    `chunk` / `shuffle_chunk` expand one row into several Examples (the dataset
    gets longer). Sibling groups from the same row share an `example_uid`, and
    the loss masks them from each other so they never become false negatives.

  * when M < k, `view_pad` decides:
      - `mask` (default): the collator duplicates views and masks them out of the loss;
      - `cycle`: fill with the sample's own views cyclically; these **count as valid
        views** (a view appearing twice in a block effectively gets a larger weight;
        with `dedupe_records` this costs no extra compute).
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from torch.utils.data import Dataset

from ..registry import DATASETS
from ..utils.misc import get_logger
from .instructions import resolve_instruction
from .identity import (
    DocIdCanonicalizer,
    RelationIndex,
    record_key,
)
from .lexical_negatives import DifficultySchedule, LexicalNegativeIndex
from .progress import SharedProgress
from .schema import Example, Record, count_views, parse_example, parse_record

VIEW_STRATEGIES = ("first", "shuffle", "chunk", "shuffle_chunk")
VIEW_PADS = ("mask", "cycle")


def _n_groups(n_views: int, k: int) -> int:
    """Number of groups obtained when chunking M views by k (k <= 0 means "all views form one group")."""
    if k <= 0 or n_views <= 0:
        return 1
    return max(1, -(-n_views // k))  # ceil


def resolve_view_counts(
    multiview_k: Optional[int] = None,
    num_query_views: Optional[int] = None,
    num_positives: Optional[int] = None,
) -> Tuple[int, int]:
    """Normalize `multiview_k` / `num_query_views` / `num_positives` into (Nq, Np).

    Priority: explicit num_query_views / num_positives > multiview_k > 1.
    So `multiview_k: 3` gives a 3x3 block; set one of the two explicitly for a
    non-square block (e.g. 2 query views with 4 positive views).

    Convention: -1 means "use all views without truncation" and is only allowed
    for evaluation sets (training gathers across GPUs and needs regular shapes).
    """
    k = int(multiview_k) if multiview_k is not None else None
    nq = int(num_query_views) if num_query_views is not None else (k if k is not None else 1)
    npos = int(num_positives) if num_positives is not None else (k if k is not None else 1)
    if nq == 0 or npos == 0:
        raise ValueError("num_query_views / num_positives must not be 0 (-1 = all views, >=1 = fixed k)")
    return (nq if nq > 0 else -1), (npos if npos > 0 else -1)

logger = get_logger(__name__)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i + 1} is not valid JSON: {e}") from e
    return rows


@DATASETS.register("jsonl")
class ContrastiveJsonlDataset(Dataset):
    def __init__(
        self,
        path: str | Sequence[str],
        num_negatives: int = 1,
        image_root: str = "",
        task_instructions: Optional[Dict[str, Dict[str, str]]] = None,
        negative_strategy: str = "shuffle",  # shuffle | topk | random_pool
        negative_fill: str = "random_pool",  # random_pool | repeat | drop
        max_samples: int = -1,
        seed: int = 42,
        multiview_k: Optional[int] = None,  # sets both values below at once (k x k block)
        num_query_views: Optional[int] = None,  # query views per sample (1 = single view)
        num_positives: Optional[int] = None,    # positive views per sample (1 = single view)
        view_strategy: str = "first",  # first | shuffle | chunk | shuffle_chunk
        view_pad: str = "mask",        # when fewer than k views: mask | cycle
        max_view_groups: int = -1,     # max view groups expanded per sample (-1 = unlimited)
        auto_ids: bool = True,          # fall back to content hashes for missing doc_id / query_id
        canonicalize_doc_ids: bool = True,  # merge "same content, different id" and "same id, different content"
        auto_positive_ids: bool = True, # global scan to collect query -> all relevant docs
        avoid_related_negatives: bool = True,  # skip docs related to the query when sampling random negatives
        hard_negatives: Optional[Dict[str, Any]] = None,  # lexical hard-negative mining + curriculum
        multiview_negatives: bool = False,  # whether a negative returns all views of its row as one group
        progress: Optional[SharedProgress] = None,        # training progress (written by a callback)
    ) -> None:
        paths = [path] if isinstance(path, str) else list(path)
        self.rows: List[Dict[str, Any]] = []
        for p in paths:
            rows = _read_jsonl(p)
            logger.info("Loaded %s: %d rows", p, len(rows))
            self.rows.extend(rows)
        if max_samples > 0:
            self.rows = self.rows[:max_samples]

        self.num_negatives = int(num_negatives)
        self.image_root = image_root or ""
        self.task_instructions = task_instructions or {}
        self.negative_strategy = negative_strategy
        self.negative_fill = negative_fill

        # ---- view config: multiview_k is a shortcut that sets both sides ----
        self.num_query_views, self.num_positives = resolve_view_counts(
            multiview_k, num_query_views, num_positives
        )
        if view_strategy not in VIEW_STRATEGIES:
            raise ValueError(f"view_strategy must be one of {VIEW_STRATEGIES}, got: {view_strategy}")
        if view_pad not in VIEW_PADS:
            raise ValueError(f"view_pad must be one of {VIEW_PADS}, got: {view_pad}")
        self.view_strategy = view_strategy
        self.view_pad = view_pad
        self.max_view_groups = int(max_view_groups)

        self.epoch = 0
        self._rng = random.Random(seed)
        self._seed = seed
        self._warned_missing_neg = False
        self._warned_missing_view = False
        self._warned_dropped_view = False

        # Global document pool (positives) used by negative_fill=random_pool.
        # In multi-view data `positive` may be a list; the raw field is stored here
        # and normalized into a single Record by _pool_record when sampled.
        self.auto_ids = bool(auto_ids)
        self.auto_positive_ids = bool(auto_positive_ids)
        self.avoid_related_negatives = bool(avoid_related_negatives)

        # Document identity canonicalization: same content / different id, same id / different views -> one doc_id
        self.canonicalize_doc_ids = bool(canonicalize_doc_ids)
        self._canon: Optional[DocIdCanonicalizer] = (
            DocIdCanonicalizer.from_rows(self.rows) if self.canonicalize_doc_ids else None
        )
        if self._canon is not None:
            cs = self._canon.stats()
            logger.info(
                "Document identity canonicalization: %d id/content nodes -> %d documents",
                int(cs["n_nodes"]),
                int(cs["n_documents"]),
            )

        # Global relation table: query_id -> all known relevant doc_ids (non-transitive, one hop)
        self._relations = RelationIndex(
            self.rows, enabled=self.auto_positive_ids, canonicalizer=self._canon
        )
        if self.auto_positive_ids:
            stats = self._relations.stats()
            if stats:
                logger.info(
                    "Relation index: %d unique queries, %d with multiple positives (max %d, mean %.2f); "
                    "%d rows share a query with other rows; they are masked from each other within a batch.",
                    int(stats["n_queries"]),
                    int(stats["n_multi_positive_queries"]),
                    int(stats["max_positives_per_query"]),
                    stats["mean_positives_per_query"],
                    int(stats["duplicate_query_rows"]),
                )

        # Global document pool used by negative_fill=random_pool.
        # Each entry stores (row_idx, raw positive field, doc_ids) so a row can reliably
        # exclude itself and its related documents.
        self._doc_pool: List[Tuple[int, Any, Tuple[str, ...]]] = []
        for i, r in enumerate(self.rows):
            raw = r.get("positive", r.get("pos"))
            if not raw:
                continue
            self._doc_pool.append((i, raw, tuple(self._relations.doc_keys(r))))

        # ---- lexical hard-negative mining + difficulty curriculum (off by default) ----
        self.progress = progress
        self._hn_cfg = dict(hard_negatives or {})
        self._hn_index: Optional[LexicalNegativeIndex] = None
        self._hn_schedule: Optional[DifficultySchedule] = None
        self.multiview_negatives = bool(multiview_negatives)
        # With k > 1, negatives must also be complete k-view groups: with k columns per
        # positive document but 1 column per negative, the collator's group_size would not
        # match and the k x k block structure of the block-diagonal InfoNCE would be misaligned.
        if self.num_positives > 1 and not self.multiview_negatives:
            logger.warning(
                "num_positives=%d but data.multiview_negatives=false: negatives would have a single view, "
                "inconsistent with the %d views of the positives (block-diagonal InfoNCE needs equal view counts). "
                "Multi-view negatives are enabled automatically; set num_positives back to 1 to avoid this.",
                self.num_positives,
                self.num_positives,
            )
            self.multiview_negatives = True
        if self._hn_cfg.get("enable"):
            self._build_hard_negatives()

        # ---- view-group index: "one row -> one sample" becomes "one row -> several view groups" ----
        self._index: List[Tuple[int, int]] = self._build_group_index()
        self._log_view_config()

    # ---------------- view-group index ----------------
    @property
    def expands_views(self) -> bool:
        """Whether one row is expanded into several Examples."""
        return self.view_strategy in ("chunk", "shuffle_chunk")

    def _groups_for_row(self, row: Dict[str, Any]) -> int:
        if not self.expands_views:
            return 1
        g = max(
            _n_groups(count_views(row, "query"), self.num_query_views),
            _n_groups(count_views(row, "positive"), self.num_positives),
        )
        if self.max_view_groups > 0:
            g = min(g, self.max_view_groups)
        return g

    def _build_group_index(self) -> List[Tuple[int, int]]:
        index: List[Tuple[int, int]] = []
        for i, row in enumerate(self.rows):
            for g in range(self._groups_for_row(row)):
                index.append((i, g))
        return index

    def _log_view_config(self) -> None:
        if self.num_query_views == 1 and self.num_positives == 1:
            return
        n_q = [count_views(r, "query") for r in self.rows]
        n_p = [count_views(r, "positive") for r in self.rows]
        logger.info(
            "Multi-view mode: num_query_views=%d, num_positives=%d, view_strategy=%s, view_pad=%s "
            "(InfoNCE target matrix is block-diagonal with %dx%d blocks)",
            self.num_query_views,
            self.num_positives,
            self.view_strategy,
            self.view_pad,
            self.num_query_views,
            self.num_positives,
        )
        logger.info(
            "Views per row: query max=%d / mean=%.2f, positive max=%d / mean=%.2f; "
            "samples after expansion %d -> %d",
            max(n_q or [0]),
            sum(n_q) / max(len(n_q), 1),
            max(n_p or [0]),
            sum(n_p) / max(len(n_p), 1),
            len(self.rows),
            len(self._index),
        )
        uses_all = self.num_query_views <= 0 and self.num_positives <= 0
        if not uses_all and not self.expands_views and (
            max(n_q or [0]) > self.num_query_views or max(n_p or [0]) > self.num_positives
        ):
            logger.warning(
                "Some rows have more views than k; view_strategy=%s only uses %d/%d of them per step. "
                "Set data.view_strategy: chunk (or shuffle_chunk) to use all views.",
                self.view_strategy,
                self.num_query_views,
                self.num_positives,
            )

    # ---------------- basic interface ----------------
    def __len__(self) -> int:
        return len(self._index)

    def set_epoch(self, epoch: int) -> None:
        """Vary hard-negative / view sampling across epochs."""
        self.epoch = epoch

    def __getitem__(self, index: int) -> Example:
        row_idx, group = self._index[index]
        ex = parse_example(self.rows[row_idx], line_id=str(row_idx))
        ex.view_group = group
        if not ex.example_uid:
            ex.example_uid = str(row_idx)
        # Identities must be filled **before** views are truncated: positive_ids must record
        # all positives of this row, including those dropped below (they are still relevant).
        self._assign_identity(ex, row_idx)
        ex.negatives = self._select_negatives(ex, row_idx, group)
        if self.auto_ids:
            self._canonicalize(ex.negatives)
        self._select_views(ex, row_idx, group)
        self._apply_image_root(ex)
        self._apply_instruction(ex)
        return ex

    # ---------------- identities ----------------
    def _assign_identity(self, ex: Example, row_idx: int) -> None:
        """Fill query_uid / group_id / positive uids / positive_ids.

        Together these determine how many false negatives the loss can mask. Missing any
        of them only means masking less, never masking wrongly (safe degradation).
        """
        if self.auto_ids:
            self._canonicalize(ex.positive_views)
        if not ex.query_uid:
            qk = self._relations.row_query_keys[row_idx] if self.auto_positive_ids else ""
            ex.query_uid = qk or (record_key(ex.query_views[0]) if self.auto_ids else None)
        if not ex.group_id:
            ex.group_id = ex.example_uid

        ids: List[str] = list(ex.positive_ids)
        for pid in [v.uid for v in ex.positive_views if v.uid]:
            if pid not in ids:
                ids.append(pid)
        for pid in sorted(self._relations.positives_for(ex.query_uid or "")):
            if pid not in ids:
                ids.append(pid)
        ex.positive_ids = ids

    def _canonicalize(self, records) -> None:
        """Replace Record.uid with the canonical doc_id (content hash if missing). In place.

        Negatives are List[List[Record]] (each negative is a view group), so this accepts
        both a list of Records and a list of lists of Records.
        """
        for item in records:
            if isinstance(item, (list, tuple)):
                for rec in item:
                    rec.uid = self._relations.canonical_record_id(rec)
            else:
                item.uid = self._relations.canonical_record_id(item)

    # ---------------- internals ----------------
    def _rng_for(self, index: int, salt: int = 0) -> random.Random:
        return random.Random(self._seed * 1000003 + index * 7919 + self.epoch + salt)

    def _pool_record(self, raw: Any, rng: random.Random) -> Record:
        """Pool entries may be a single record or a list of views; return one Record."""
        if isinstance(raw, (list, tuple)):
            if not raw:
                return Record(text="")
            return parse_record(raw[rng.randrange(len(raw))])
        return parse_record(raw)

    def _sample_pool_negative(
        self, ex: Example, row_idx: int, rng: random.Random
    ) -> Optional[Record]:
        """Sample a random document **unrelated to this query** from the global pool as a negative.

        It (1) never samples the row's own positive or any document related to the query,
        and (2) always assigns a uid so that doc_id masking in the loss also applies to it.
        """
        if not self._doc_pool:
            return None
        related = set(ex.positive_ids)
        for _ in range(16):
            src_row, raw, doc_keys = self._doc_pool[rng.randrange(len(self._doc_pool))]
            if src_row == row_idx:
                continue
            if self.avoid_related_negatives and related.intersection(doc_keys):
                continue
            rec = self._pool_record(raw, rng)
            if self.auto_ids:
                rec.uid = self._relations.canonical_record_id(rec)
            return rec
        # If nothing suitable can be found (tiny dataset / dense relations), skip this slot
        # and rely on doc_id masking in the loss; never insert a known-related document.
        return None

    def _sample_pool_negative_group(
        self, ex: Example, row_idx: int, rng: random.Random
    ) -> Optional[List[Record]]:
        """Sample a random row from the global pool and return all its views as one group.

        Used when multiview_negatives=true.
        """
        if not self._doc_pool:
            return None
        related = set(ex.positive_ids)
        for _ in range(16):
            src_row, raw, doc_keys = self._doc_pool[rng.randrange(len(self._doc_pool))]
            if src_row == row_idx:
                continue
            if self.avoid_related_negatives and related.intersection(doc_keys):
                continue
            # Return all views of this row (same truncation / false-negative rules as hard negatives)
            recs = self._row_negative_group(src_row, related, 0, rng)
            if recs:
                return recs
        return None

    # ---------------- lexical hard-negative mining ----------------
    def _doc_text_at(self, row: Dict[str, Any], slot: int) -> str:
        """Plain text of the `slot`-th positive view of a row.

        In a 1x3 layout, slot=0 is by convention the text-only document. This only holds
        when `view_strategy` is first / chunk (shuffle changes the order);
        `validate_text_slot()` spot-checks it at startup and warns.
        """
        raw = row.get("positive", row.get("pos"))
        if raw is None:
            return ""
        views = raw if isinstance(raw, (list, tuple)) else [raw]
        if slot >= len(views):
            return ""
        rec = parse_record(views[slot])
        return rec.text or ""

    def validate_text_slot(self, sample: int = 200) -> List[str]:
        """Spot-check that the `slot`-th positive view is text-only; return a list of warnings."""
        slot = int(self._hn_cfg.get("text_view_index", 0))
        warns: List[str] = []
        if self.view_strategy in ("shuffle", "shuffle_chunk"):
            warns.append(
                f"data.hard_negatives mines with a fixed slot (positive view #{slot} = text-only), "
                f"but view_strategy={self.view_strategy} shuffles view order every epoch, "
                "so mined similarities would not match the views actually fed. Use first or chunk."
            )
        n_missing = n_has_image = 0
        for row in self.rows[:sample]:
            raw = row.get("positive", row.get("pos"))
            views = raw if isinstance(raw, (list, tuple)) else ([raw] if raw else [])
            if slot >= len(views):
                n_missing += 1
                continue
            rec = parse_record(views[slot])
            if not rec.text:
                n_missing += 1
            if rec.images:
                n_has_image += 1
        checked = min(sample, len(self.rows))
        if n_missing:
            warns.append(
                f"Checked {checked} rows: {n_missing} rows have no text in positive view #{slot}; "
                "they will get no mined hard negatives (falling back to the original negative sources)"
            )
        if n_has_image:
            warns.append(
                f"Checked {checked} rows: {n_has_image} rows have an image in positive view #{slot}, "
                f"so slot={slot} may not be the text-only document slot; "
                "please check data.hard_negatives.text_view_index"
            )
        return warns

    def _build_hard_negatives(self) -> None:
        cfg = self._hn_cfg
        slot = int(cfg.get("text_view_index", 0))
        texts = [self._doc_text_at(r, slot) for r in self.rows]

        for w in self.validate_text_slot():
            logger.warning("%s", w)

        # Exclusion set: rows known to be related to this query can never be negatives.
        # Rows are looked up via doc_keys, the same criterion as avoid_related_negatives.
        key_to_rows: Dict[str, List[int]] = {}
        for i, r in enumerate(self.rows):
            for k in self._relations.doc_keys(r):
                key_to_rows.setdefault(k, []).append(i)
        exclude: List[set] = []
        for i, r in enumerate(self.rows):
            bad = {i}
            for k in self._relations.doc_keys(r):
                bad.update(key_to_rows.get(k, ()))
            qk = self._relations.row_query_keys[i] if self.auto_positive_ids else ""
            for k in (self._relations.positives_for(qk) if qk else ()):
                bad.update(key_to_rows.get(k, ()))
            exclude.append(bad)

        self._hn_index = LexicalNegativeIndex.load_or_build(
            texts,
            cache_dir=cfg.get("cache_dir"),
            exclude=exclude,
            top_k=int(cfg.get("top_k", 50)),
            min_similarity=float(cfg.get("min_similarity", 0.05)),
            max_similarity=float(cfg.get("max_similarity", 0.9)),
            max_df_ratio=float(cfg.get("max_df_ratio", 0.3)),
            similarity=str(cfg.get("similarity", "f1")),
            block_size=int(cfg.get("block_size", 0)),
            block_assign=str(cfg.get("block_assign", "signature")),
            block_rounds=int(cfg.get("block_rounds", 1)),
            query_chunk=int(cfg.get("query_chunk", 2048)),
            min_df=int(cfg.get("min_df", 2)),
            backend=str(cfg.get("backend", "auto")),
            seed=int(cfg.get("seed", 0)),
        )
        self._hn_schedule = DifficultySchedule(
            window=int(cfg.get("window", 10)),
            schedule=cfg.get("schedule", "linear"),
            warmup_ratio=float(cfg.get("warmup_ratio", 0.0)),
            mix_init=float(cfg.get("mix_init", 0.3)),
            mix_final=float(cfg.get("mix_final", 1.0)),
        )
        logger.info("%s", self._hn_index.summary())
        if cfg.get("preview", True):
            # Print the hardest pairs by default so you can check for missed true positives before training
            logger.info("%s", self._hn_index.describe_pairs(k=int(cfg.get("preview_pairs", 5))))
        SharedProgress.check_start_method()

    def _row_negative_group(
        self, src_row: int, related: set, group: int, rng: random.Random
    ) -> Optional[List[Record]]:
        """Take all positive views of row `src_row` as one negative group (k views).

        Two key points:

        1. **Replace the whole group, not just the text view.** Lexical mining only uses
           view #`text_view_index` (the text-only document in a 1x3 layout) to *find*
           similar rows; the negative attached is that row's complete set of k views.
           Otherwise a negative would have 1 view while positives have k, breaking the
           k x k structure of the block-diagonal InfoNCE.
        2. **Views must be picked exactly like the positives** (same `_pick_views` and
           same `view_group`), so that view j of the negative group corresponds to
           view j of the positive group.

        If **any** view of the row hits the related set (known relevant), the whole
        row is dropped. Removing only the matching view would shift the remaining
        views and misalign the slots.
        """
        raw = self.rows[src_row].get("positive", self.rows[src_row].get("pos"))
        if raw is None:
            return None
        views = list(raw) if isinstance(raw, (list, tuple)) else [raw]
        recs: List[Record] = []
        for v in views:
            rec = parse_record(v)
            if self.auto_ids:
                rec.uid = self._relations.canonical_record_id(rec)
            if rec.uid and rec.uid in related:
                return None  # likely a true positive: drop the whole row
            recs.append(rec)
        if not recs:
            return None
        # Same truncation rules as positives (first / chunk / cycle padding ...)
        picked = self._pick_views(recs, self.num_positives, group, rng)
        return picked or None

    def _mined_negatives(
        self, ex: Example, row_idx: int, n: int, rng: random.Random, group: int = 0
    ) -> List[List[Record]]:
        """Take n hard-negative rows from the mined candidates according to current progress (may return fewer).

        With `multiview_negatives=true` (enabled automatically when k > 1) each hard negative
        returns the full k views of its row, matching the positive group; otherwise a single
        view is returned (k=1).
        """
        if self._hn_index is None or self._hn_schedule is None or n <= 0:
            return []
        cands = self._hn_index.neighbors[row_idx]
        if not cands:
            return []
        p = self.progress.get() if self.progress is not None else 0.0
        start, end = self._hn_schedule.band(len(cands), p)
        pool = cands[start:end] or cands[:1]
        # Over-fetch candidates: rows hitting the related set are dropped, so spares are needed,
        # otherwise hard-negative slots would be left empty (degrading to random negatives).
        n_try = min(len(pool), max(n * 2, n + 2))
        picked = rng.sample(pool, n_try)

        related = set(ex.positive_ids)
        out: List[List[Record]] = []
        seen_rows: set = set()
        for m in picked:
            if len(out) >= n:
                break
            if m.row in seen_rows or m.row == row_idx:
                continue
            seen_rows.add(m.row)

            if self.multiview_negatives:
                recs = self._row_negative_group(m.row, related, group, rng)
                if recs:
                    out.append(recs)
                continue

            # Single-view mode: only the given slot (k=1 path)
            raw = self.rows[m.row].get("positive", self.rows[m.row].get("pos"))
            if raw is None:
                continue
            views = raw if isinstance(raw, (list, tuple)) else [raw]
            slot = int(self._hn_cfg.get("text_view_index", 0))
            if self._hn_cfg.get("negative_view", "slot") == "random":
                rec = parse_record(views[rng.randrange(len(views))])
            else:
                rec = parse_record(views[slot] if slot < len(views) else views[0])
            if self.auto_ids:
                rec.uid = self._relations.canonical_record_id(rec)
            if rec.uid and rec.uid in related:
                continue
            out.append([rec])
        return out[:n]

    def _select_negatives(self, ex: Example, index: int, group: int = 0) -> List[List[Record]]:
        n = self.num_negatives
        if n <= 0:
            return []
        cands = list(ex.negatives)  # List[List[Record]]
        # Sibling view groups use different random seeds, so their hard negatives differ (more signal)
        rng = self._rng_for(index, salt=group * 101)

        # Mined hard negatives take a share of the slots (ratio grows from mix_init to mix_final with the
        # curriculum); the remaining slots use the original sources (in-data negatives / random pool).
        if self._hn_index is not None and self._hn_schedule is not None:
            p = self.progress.get() if self.progress is not None else 0.0
            n_mined = min(n, int(round(n * self._hn_schedule.mix_ratio(p))))
            mined = self._mined_negatives(ex, index, n_mined, rng, group=group)
            if mined:
                rest = n - len(mined)
                if rest <= 0:
                    return mined[:n]
                # Collect all uids in `mined` (each group may contain several)
                mined_uids = {r.uid for group_recs in mined for r in group_recs if r.uid}
                # Drop candidates already in `mined` (compare the first uid of each group)
                cands = [c for c in cands if not any(r.uid in mined_uids for r in c if r.uid)]
                tail = self._select_negatives_plain(ex, index, cands, rest, rng)
                return mined + tail

        if len(cands) >= n:
            if self.negative_strategy == "topk":
                return cands[:n]
            if self.negative_strategy == "random_pool":
                return rng.sample(cands, n)
            rng.shuffle(cands)  # shuffle (default): shuffle then take n, balancing difficulty and diversity
            return cands[:n]

        if not cands and not self._warned_missing_neg:
            logger.warning(
                "Some samples have no hard negatives and will be filled with negative_fill=%s; "
                "set data.num_negatives to 0 to use in-batch negatives only.",
                self.negative_fill,
            )
            self._warned_missing_neg = True

        if self.negative_fill == "drop":
            return cands
        need = n - len(cands)
        if self.negative_fill == "repeat" and cands:
            filler = [cands[i % len(cands)] for i in range(need)]
        else:  # random_pool: sample from the global document pool (i.e. random negatives)
            filler: List[List[Record]] = []
            for _ in range(need):
                if self.multiview_negatives:
                    group = self._sample_pool_negative_group(ex, index, rng)
                    if group is not None:
                        filler.append(group)
                else:
                    rec = self._sample_pool_negative(ex, index, rng)
                    if rec is not None:
                        filler.append([rec])
            if len(cands) + len(filler) < n:
                # The collator requires the same number of negatives per sample, so duplicates are used;
                # duplicated negatives are masked by doc_id in the loss, so no bias is introduced.
                pool = cands + filler
                while len(pool) < n and pool:
                    pool.append(pool[len(pool) % max(len(cands) + len(filler), 1)])
                return pool[:n] if pool else []
        return cands + filler

    def _select_negatives_plain(
        self, ex: Example, index: int, cands: List[List[Record]], n: int, rng: random.Random
    ) -> List[List[Record]]:
        """Default negative selection (in-data negatives + random-pool filling), used when mining is off."""
        if n <= 0:
            return []
        if len(cands) >= n:
            if self.negative_strategy == "topk":
                return cands[:n]
            if self.negative_strategy == "random_pool":
                return rng.sample(cands, n)
            pool = list(cands)
            rng.shuffle(pool)
            return pool[:n]
        if self.negative_fill == "drop":
            return cands
        need = n - len(cands)
        filler: List[List[Record]] = []
        if self.negative_fill == "repeat" and cands:
            filler = [cands[i % len(cands)] for i in range(need)]
        else:
            for _ in range(need):
                if self.multiview_negatives:
                    group = self._sample_pool_negative_group(ex, index, rng)
                    if group is not None:
                        filler.append(group)
                else:
                    rec = self._sample_pool_negative(ex, index, rng)
                    if rec is not None:
                        filler.append([rec])
        return (cands + filler)[:n]

    def _pick_views(
        self, views: List[Record], n: int, group: int, rng: random.Random
    ) -> List[Record]:
        """Select the `group`-th view group (size n) from all views of a sample.

        n <= 0 means "all views, no truncation" (for evaluation; regular shapes not needed).
        """
        if n <= 0:
            return list(views)
        m = len(views)

        if self.view_strategy == "first":
            picked = views[:n]
        elif self.view_strategy == "shuffle":
            order = list(views)
            rng.shuffle(order)
            picked = order[:n]
        else:  # chunk / shuffle_chunk: split M views into ceil(M/n) groups
            order = list(views)
            if self.view_strategy == "shuffle_chunk":
                rng.shuffle(order)
            # Query and positive may have different group counts (e.g. 3 query views vs 6 positive views);
            # the side with fewer groups reuses its groups cyclically so both sides are fully covered.
            n_groups = _n_groups(m, n)
            g = group % n_groups
            picked = order[g * n : g * n + n]

        # Fewer than n views: `cycle` borrows from **all real views** of the sample (valid views,
        # preferring views not yet in this group and repeating only if necessary);
        # `mask` returns them as-is and the collator pads + masks them (excluded from the loss).
        if len(picked) < n and self.view_pad == "cycle" and views:
            picked = list(picked)
            chosen = {id(v) for v in picked}
            spare = [v for v in views if id(v) not in chosen]
            while len(picked) < n:
                picked.append(spare.pop(0) if spare else views[len(picked) % m])
        return list(picked)

    def _select_views(self, ex: Example, index: int, group: int = 0) -> None:
        """Truncate views to the `group`-th view group; padding is **not** done here (the collator pads and masks)."""
        rng = self._rng_for(index, salt=13)
        ex.set_query_views(self._pick_views(ex.query_views, self.num_query_views, group, rng))
        ex.set_positive_views(self._pick_views(ex.positive_views, self.num_positives, group, rng))

        if not self._warned_missing_view and (
            len(ex.query_views) < self.num_query_views
            or len(ex.positive_views) < self.num_positives
        ):
            logger.warning(
                "Some samples have fewer than k views (query=%d/%d, positive=%d/%d); "
                "missing views are duplicated and masked in the loss (no effect on gradients, slight compute overhead). "
                "Set data.view_pad: cycle to count them as valid views.",
                len(ex.query_views),
                self.num_query_views,
                len(ex.positive_views),
                self.num_positives,
            )
            self._warned_missing_view = True

    def _apply_image_root(self, ex: Example) -> None:
        if not self.image_root:
            return
        for rec in ex.all_records():
            rec.images = [
                p if (os.path.isabs(p) or p.startswith(("http://", "https://", "data:")))
                else os.path.join(self.image_root, p)
                for p in rec.images
            ]

    def _instruction_for(self, table: Dict[str, str], role: str, rec: Record) -> Optional[str]:
        """Per-modality instructions are supported:

            task_instructions:
              multiview:
                query: "..."            # fallback
                query_text: "..."       # text-only query views
                query_image: "..."      # image-only query views
                doc: "..."
                doc_text: "..."
                doc_image: "..."

        Implemented in data/instructions.py; **evaluation scripts use the same function**,
        so training and inference prompts cannot silently diverge.
        """
        return resolve_instruction(table, role, rec)

    def _apply_instruction(self, ex: Example) -> None:
        table = self.task_instructions.get(ex.task) or self.task_instructions.get("default") or {}
        if not table:
            return
        for rec in ex.query_views:
            if rec.instruction is None:
                rec.instruction = self._instruction_for(table, "query", rec)
        for rec in ex.positive_views:
            if rec.instruction is None:
                rec.instruction = self._instruction_for(table, "doc", rec)
        # Negatives are List[List[Record]] (each negative is a view group)
        for neg_group in ex.negatives:
            for rec in neg_group:
                if rec.instruction is None:
                    rec.instruction = self._instruction_for(table, "doc", rec)


@DATASETS.register("jsonl_eval")
class RetrievalEvalDataset(ContrastiveJsonlDataset):
    """Evaluation set: no hard negatives, but negatives are added to the candidate pool (see eval/retrieval.py).

    By default `num_query_views = num_positives = -1` (all views): evaluation does not
    gather across GPUs, so there is no need to truncate views. Each query view is a
    separate query whose gold set is all positive views.
    """

    def __init__(self, path, **kwargs):
        kwargs.setdefault("num_negatives", 0)
        kwargs["negative_fill"] = "drop"
        kwargs.setdefault("multiview_k", -1)
        super().__init__(path, **kwargs)

    def __getitem__(self, index: int) -> Example:
        row_idx, group = self._index[index]
        ex = parse_example(self.rows[row_idx], line_id=str(row_idx))
        ex.view_group = group
        if not ex.example_uid:
            ex.example_uid = str(row_idx)
        # Keep original negatives as distractors
        self._select_views(ex, row_idx, group)
        self._apply_image_root(ex)
        self._apply_instruction(ex)
        return ex
