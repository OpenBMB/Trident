"""Identity fields (doc_id / query_id / group_id) and the global relation index.

This layer is **the foundation of false-negative masking**: whether the loss
can recognize that "another row's positive in the batch is also my positive"
depends entirely on the ids provided here.

Three kinds of ids with strictly separated roles:

    doc_id    Document-level identity. **The same document must have the same id
              in every row.** This is the most important one: it lets the loss
              mask every cell (other rows' positives / hard negatives) that equals
              one of my positives.
    query_id  Query-level identity. When a query has several positives and is
              split into several rows, these rows must share one query_id
              (used by the symmetric doc->query direction).
    group_id  Exclusive cluster id. Any query and any doc in the same cluster are
              never negatives of each other. Useful for "all assets of the same
              entity"-style grouping.

Missing ids fall back to a **content hash** (`auto_ids`): identical text + image
paths -> identical id. Even data without any ids therefore avoids false
negatives caused by the same document appearing twice in a batch.

`RelationIndex` adds one more layer: a full-file scan builds a
query_id -> {doc_id} relation table, so "the 2nd positive of the same query
appears in another row" is also recognized. This relation is **non-transitive**
(no union-find over queries), so the dataset is never merged into one giant
cluster that would mask all negatives.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from .schema import Record, parse_records

# Field aliases within a row (kept consistent with schema.parse_example)
_QUERY_KEYS = ("query_views", "queries", "query", "anchor", "q")
_POS_KEYS = ("positive_views", "positives", "positive", "pos", "positive_document")


def _pick(obj: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if obj.get(k) is not None:
            return obj[k]
    return None


def content_key(text: Optional[str], images: Sequence[str]) -> str:
    """Content hash: identical text + image paths (in order) -> identical id.

    Instructions are deliberately **excluded**: they are injected at training time
    and may differ across tasks for the same document.
    """
    raw = "\x01".join([(text or "").strip(), *[str(i) for i in images]])
    return "h:" + hashlib.blake2b(raw.encode("utf-8"), digest_size=10).hexdigest()


def record_key(rec: Record) -> str:
    return content_key(rec.text, rec.images)


def assign_record_uids(records: Iterable[Record]) -> None:
    """Assign a content-hash uid to Records without one (in place)."""
    for rec in records:
        if not rec.uid:
            rec.uid = record_key(rec)


def _views_key(views: List[Record]) -> str:
    """Content hash of a whole group of views (used as a fallback query_id).

    **All views** are used so that different view groups of the same query get
    the same key; otherwise the relation table would fragment.
    """
    return "h:" + hashlib.blake2b(
        "\x02".join(record_key(v) for v in views).encode("utf-8"), digest_size=10
    ).hexdigest()


def row_query_key(obj: Dict[str, Any]) -> str:
    """query_id of a row (explicit > content hash)."""
    qid = obj.get("query_id") or obj.get("qid")
    if qid:
        return str(qid)
    raw = _pick(obj, *_QUERY_KEYS)
    if raw is None:
        return ""
    return _views_key(parse_records(raw))


def row_doc_keys(obj: Dict[str, Any]) -> List[str]:
    """doc_id of each positive view in a row (per-view id > doc_ids[i] > doc_id > content hash).

    Must stay **exactly aligned** with the uid resolution in `schema.parse_example`.
    """
    raw = _pick(obj, *_POS_KEYS)
    if raw is None:
        return []
    doc_ids = obj.get("doc_ids") or ([obj["doc_id"]] if obj.get("doc_id") else None)
    views = parse_records(raw, uids=doc_ids, default_uid=obj.get("doc_id"))
    return [v.uid or record_key(v) for v in views]


def row_declared_positive_ids(obj: Dict[str, Any]) -> List[str]:
    """Explicitly declared doc_ids known to be relevant to this query."""
    raw = obj.get("positive_ids") or obj.get("relevant_doc_ids") or []
    if isinstance(raw, str):
        raw = [raw]
    return [str(x) for x in raw if x]


def row_group_id(obj: Dict[str, Any]) -> Optional[str]:
    gid = obj.get("group_id") or obj.get("cluster_id")
    return str(gid) if gid else None


class DocIdCanonicalizer:
    """Merge multiple identities of the same document into one canonical id.

    Real data contains both of the following cases, and handling only one is not enough:

        same content, different ids   a document has doc_id=d1 in row A and no id
                                      (or d5) in row B -> must be merged, otherwise the
                                      loss cannot tell they are the same document
        same id, different content    several modality views of one document share a
                                      doc_id -> must also be merged so views recognize
                                      each other

    A union-find structure treats "declared ids" and "content hashes" as two kinds of
    nodes in one graph; each Record adds an edge between its id and its content hash,
    and the representative of each connected component becomes the canonical id.

    The union-find is **transitive only along document identity** (content <-> id),
    unlike connected components over query relations, which could merge the whole
    dataset into one giant cluster.
    """

    def __init__(self) -> None:
        self._parent: Dict[str, str] = {}
        self._declared: Set[str] = set()

    def _find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:  # path compression
            self._parent[x], x = root, self._parent[x]
        return root

    def _union(self, a: str, b: str) -> None:
        ra, rb = self._find(a), self._find(b)
        if ra == rb:
            return
        # Prefer human-written ids as representatives (more readable output)
        if rb in self._declared and ra not in self._declared:
            ra, rb = rb, ra
        self._parent[rb] = ra

    def add(self, declared_id: Optional[str], content: str) -> None:
        self._find(content)
        if declared_id:
            self._declared.add(declared_id)
            self._union(declared_id, content)

    def canonical(self, declared_id: Optional[str], content: str) -> str:
        return self._find(declared_id) if declared_id else self._find(content)

    @classmethod
    def from_rows(cls, rows: Sequence[Dict[str, Any]]) -> "DocIdCanonicalizer":
        """Scan positives and hard negatives of all rows to build document-identity components."""
        c = cls()
        for obj in rows:
            for declared, rec in _row_doc_records(obj):
                c.add(declared, record_key(rec))
            for declared, rec in _row_neg_records(obj):
                c.add(declared, record_key(rec))
        return c

    def stats(self) -> Dict[str, float]:
        roots = {self._find(k) for k in self._parent}
        return {"n_nodes": float(len(self._parent)), "n_documents": float(len(roots))}


def _row_doc_records(obj: Dict[str, Any]):
    """[(declared doc_id or None, Record), ...], in the same order as the positive views."""
    raw = _pick(obj, *_POS_KEYS)
    if raw is None:
        return []
    doc_ids = obj.get("doc_ids") or ([obj["doc_id"]] if obj.get("doc_id") else None)
    views = parse_records(raw, uids=doc_ids, default_uid=obj.get("doc_id"))
    return [(v.uid, v) for v in views]


def _row_neg_records(obj: Dict[str, Any]):
    raw = obj.get("negatives") or obj.get("negative") or obj.get("negs") or []
    if not isinstance(raw, list):
        raw = [raw]
    neg_ids = obj.get("negative_ids") or []
    recs = parse_records(raw)
    out = []
    for i, rec in enumerate(recs):
        uid = rec.uid or (str(neg_ids[i]) if i < len(neg_ids) else None)
        out.append((uid, rec))
    return out


class RelationIndex:
    """Global relation table: query_id -> set of doc_ids known to be relevant.

    Only **one-hop** relations are stored (a query links to the positives it has in
    any row). No transitive closure is computed: a popular document could chain
    thousands of queries into one giant cluster, masking all in-batch negatives
    and collapsing training.
    """

    def __init__(
        self,
        rows: Sequence[Dict[str, Any]],
        enabled: bool = True,
        canonicalizer: Optional[DocIdCanonicalizer] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.canon = canonicalizer
        self.q2d: Dict[str, Set[str]] = {}
        # Cache query_key per row so __getitem__ does not recompute content hashes
        self.row_query_keys: List[str] = []
        self.n_rows = len(rows)
        if not self.enabled:
            self.row_query_keys = ["" for _ in rows]
            return
        for obj in rows:
            qk = row_query_key(obj)
            self.row_query_keys.append(qk)
            if not qk:
                continue
            bucket = self.q2d.setdefault(qk, set())
            bucket.update(self.doc_keys(obj))
            bucket.update(row_declared_positive_ids(obj))

    def doc_keys(self, obj: Dict[str, Any]) -> List[str]:
        """Positive doc_ids of a row (canonicalized if a canonicalizer is set)."""
        if self.canon is None:
            return row_doc_keys(obj)
        return [self.canon.canonical(d, record_key(rec)) for d, rec in _row_doc_records(obj)]

    def canonical_record_id(self, rec: Record) -> str:
        """Canonical doc_id of any Record (used for negatives and random-pool documents)."""
        key = record_key(rec)
        if self.canon is None:
            return rec.uid or key
        return self.canon.canonical(rec.uid, key)

    def positives_for(self, query_key: str) -> Set[str]:
        if not self.enabled or not query_key:
            return set()
        return self.q2d.get(query_key, set())

    # ---------------- diagnostics ----------------
    def stats(self) -> Dict[str, float]:
        if not self.enabled or not self.q2d:
            return {}
        sizes = [len(v) for v in self.q2d.values()]
        multi = sum(1 for s in sizes if s > 1)
        return {
            "n_queries": float(len(self.q2d)),
            "n_multi_positive_queries": float(multi),
            "max_positives_per_query": float(max(sizes)),
            "mean_positives_per_query": sum(sizes) / len(sizes),
            "duplicate_query_rows": float(self.n_rows - len(self.q2d)),
        }
