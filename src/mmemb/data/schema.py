"""Unified sample structures.

All data inside the framework is normalized into `Record` (one unit to encode:
text, any number of images, and an optional instruction) and `Example` (one
contrastive sample: query + positive(s) + hard negatives).

**Any dataset that yields `Example` objects can be plugged in** without
changing the model side.

Supported JSONL layouts (all valid; the parser normalizes them):

1) Single view:
    {"query": "a dog on the beach", "positive": {"image": "imgs/dog.jpg"}}
    {"query": {"image": "q.jpg", "text": "Which one is the same product?"},
     "positive": {"text": "white sneakers", "image": "d1.jpg"},
     "negatives": [{"text": "black leather shoes"}, "red high heels"],
     "task": "i2t", "doc_id": "d1", "negative_ids": ["n1", "n2"]}

2) Multi-view / multi-positive: `query` and `positive` may be **lists**, where
   each element is one view. All query views and all positive views of the
   same sample are mutual positives, so the InfoNCE target matrix becomes
   block-diagonal instead of diagonal:

    {"query": [{"text": "a yellow ring"}, {"image": "images/yellow_ring_q.png"}],
     "positive": [{"text": "a yellow ring", "id": "ring_txt"},
                  {"image": "images/yellow_ring_1.png", "id": "ring_img"}],
     "negatives": [{"image": "images/green_ring_1.png", "id": "green_ring_1"}],
     "doc_id": "yellow_ring_1", "task": "multiview"}

   Equivalent explicit field names: `query_views` / `positive_views`
   (or `queries` / `positives`).

3) With identity fields (recommended): add identity fields to any layout above
   so the loss can remove other rows' positives that are also my positives
   from my in-batch negatives:

    {"example_id": "ex_123",          # unique row id (defaults to the line number)
     "query_id":   "q_42",            # must be identical when one query spans several rows
     "doc_id":     "doc_777",         # document-level identity, shared everywhere
     "doc_ids":    ["d1", "d2"],      # per-view ids of the positive views (alternative to doc_id)
     "positive_ids": ["doc_777", "doc_901"],  # all documents known to be relevant to this query
     "group_id":   "grp_ring",        # exclusive cluster: never negatives of each other (optional)
     "negative_ids": ["doc_311"],     # identities of the hard negatives
     "query": ..., "positive": ..., "negatives": [...]}

   All fields are optional: missing ids are filled with a content hash of
   "text + image paths" when `data.auto_ids` is on, and `data.auto_positive_ids`
   scans the whole file to collect all relevant documents per query_id.

   Target matrix (2 query views x 2 positive views):
           d0_txt d0_img | d1_txt d1_img
     q0_txt   1      1   |   0      0
     q0_img   1      1   |   0      0
     q1_txt   0      0   |   1      1
     q1_img   0      0   |   1      1
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

RawRecord = Union[str, Dict[str, Any], None]


@dataclass
class Record:
    """One unit to encode. At least one of `text` / `images` must be non-empty."""

    text: Optional[str] = None
    images: List[str] = field(default_factory=list)
    instruction: Optional[str] = None
    uid: Optional[str] = None

    def is_empty(self) -> bool:
        return not self.text and not self.images

    @property
    def modality(self) -> str:
        """"text" / "image" / "mixed"; used to select per-modality instructions."""
        if self.images and self.text:
            return "mixed"
        if self.images:
            return "image"
        return "text"


@dataclass
class Example:
    """One contrastive sample.

    `query` / `positive` always equal the first view, so single-view code keeps
    working; multi-view information lives in `query_views` / `positive_views`
    (length 1 for single-view samples).
    """

    query: Record
    positive: Record
    negatives: List[List[Record]] = field(default_factory=list)  # multi-view negatives: List[List[Record]]
    task: str = "default"
    # Optional information for the loss (e.g. teacher scores, doc ids for false-negative removal)
    extra: Dict[str, Any] = field(default_factory=dict)
    # Multi-view fields; filled with [query] / [positive] when empty
    query_views: List[Record] = field(default_factory=list)
    positive_views: List[Record] = field(default_factory=list)
    # Which JSONL row this Example comes from. When a row has more views than k
    # and is split into several view groups (several Examples), they share the
    # same example_uid so the loss can mask sibling groups as false negatives.
    example_uid: Optional[str] = None
    view_group: int = 0
    # ---- Identity fields (false-negative masking; see data/identity.py and losses/false_negative.py) ----
    # group_id: exclusive cluster; queries / docs in the same cluster are never negatives of each other.
    group_id: Optional[str] = None
    # query_uid: query-level identity; must be identical when one query spans several rows.
    query_uid: Optional[str] = None
    # positive_ids: **all** doc_ids known to be relevant to this query (including ones not in
    # this row); the loss removes in-batch candidates with these ids from the negatives.
    positive_ids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.query_views:
            self.query_views = [self.query]
        else:
            self.query = self.query_views[0]
        if not self.positive_views:
            self.positive_views = [self.positive]
        else:
            self.positive = self.positive_views[0]

    # ---------------- view setters (keep query / positive in sync) ----------------
    def set_query_views(self, views: Sequence[Record]) -> None:
        views = [v for v in views if v is not None]
        if not views:
            raise ValueError("query_views must not be empty")
        self.query_views = list(views)
        self.query = self.query_views[0]

    def set_positive_views(self, views: Sequence[Record]) -> None:
        views = [v for v in views if v is not None]
        if not views:
            raise ValueError("positive_views must not be empty")
        self.positive_views = list(views)
        self.positive = self.positive_views[0]

    @property
    def num_query_views(self) -> int:
        return len(self.query_views)

    @property
    def num_positive_views(self) -> int:
        return len(self.positive_views)

    @property
    def is_multiview(self) -> bool:
        return self.num_query_views > 1 or self.num_positive_views > 1

    def all_records(self) -> List[Record]:
        """All Records of the sample (each object appears once; safe to modify in place)."""
        negs_flat = [r for group in self.negatives for r in group]
        return [*self.query_views, *self.positive_views, *negs_flat]


_IMAGE_KEYS = ("image", "images", "img", "image_path", "image_paths")
_TEXT_KEYS = ("text", "content", "caption", "sentence")


def parse_record(raw: RawRecord, uid: Optional[str] = None) -> Record:
    if raw is None:
        return Record(uid=uid)
    if isinstance(raw, str):
        return Record(text=raw, uid=uid)
    if not isinstance(raw, dict):
        raise TypeError(f"Cannot parse record: {type(raw)}")

    text = None
    for k in _TEXT_KEYS:
        if raw.get(k):
            text = str(raw[k])
            break

    images: List[str] = []
    for k in _IMAGE_KEYS:
        v = raw.get(k)
        if not v:
            continue
        if isinstance(v, str):
            images.append(v)
        elif isinstance(v, (list, tuple)):
            images.extend([str(x) for x in v if x])
        break

    return Record(
        text=text,
        images=images,
        instruction=raw.get("instruction") or raw.get("instruct"),
        uid=(
            str(raw.get("id") or raw.get("uid") or raw.get("doc_id") or uid)
            if (raw.get("id") or raw.get("uid") or raw.get("doc_id") or uid)
            else None
        ),
    )


def parse_records(
    raw: Any,
    uids: Optional[Sequence[Any]] = None,
    default_uid: Optional[str] = None,
) -> List[Record]:
    """Parse a single record or a list of records into List[Record] (a list of views).

    uid priority: id inside the record > uids[i] > default_uid.
    """
    items = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    out: List[Record] = []
    for i, item in enumerate(items):
        uid = None
        if uids is not None and i < len(uids) and uids[i] is not None:
            uid = str(uids[i])
        elif default_uid is not None:
            uid = str(default_uid)
        out.append(parse_record(item, uid=uid))
    return out


def _pick(obj: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if obj.get(k) is not None:
            return obj[k]
    return None


def parse_example(obj: Dict[str, Any], line_id: str = "") -> Example:
    query_raw = _pick(obj, "query_views", "queries", "query", "anchor", "q")
    pos_raw = _pick(obj, "positive_views", "positives", "positive", "pos", "positive_document")
    if query_raw is None or pos_raw is None:
        raise ValueError(f"Sample is missing the query / positive field: {obj}")

    query_ids = obj.get("query_ids") or ([obj["query_id"]] if obj.get("query_id") else None)
    doc_ids = obj.get("doc_ids") or ([obj["doc_id"]] if obj.get("doc_id") else None)

    query_views = parse_records(
        query_raw, uids=query_ids, default_uid=obj.get("query_id") or (line_id or None)
    )
    positive_views = parse_records(pos_raw, uids=doc_ids, default_uid=obj.get("doc_id"))
    if not query_views or not positive_views:
        raise ValueError(f"Empty query / positive views: {obj}")

    neg_raw = _pick(obj, "negatives", "negative", "negs") or []
    if not isinstance(neg_raw, list):
        neg_raw = [neg_raw]

    neg_ids = obj.get("negative_ids") or []

    # Auto-detect the negative layout:
    # - [[record], [record], ...] -> already multi-view
    # - [record, record, ...]     -> single-view; wrapped as [[record], [record], ...]
    negatives: List[List[Record]] = []
    for i, n in enumerate(neg_raw):
        uid = neg_ids[i] if i < len(neg_ids) else None
        if isinstance(n, (list, tuple)):
            # multi-view: [view1, view2, ...]
            recs = [parse_record(v, uid=uid) for v in n]
        else:
            # single view: record
            recs = [parse_record(n, uid=uid)]
        negatives.append(recs)

    extra: Dict[str, Any] = {}
    if obj.get("teacher_scores") is not None:
        extra["teacher_scores"] = obj["teacher_scores"]
    if len(positive_views) > 1:
        # Backward compatibility: extra positives are also exposed via `extra`
        extra["extra_positives"] = positive_views[1:]

    # example_uid identifies the source JSONL row, used to recognize sibling view groups.
    # It intentionally does **not** fall back to doc_id / query_id, which would treat
    # different queries of the same document as siblings. Cross-row relevance is handled
    # by group_id / positive_ids instead.
    example_uid = obj.get("example_id") or line_id

    # All doc_ids known to be relevant to this query: declared ids + ids of this row's positive views
    declared = obj.get("positive_ids") or obj.get("relevant_doc_ids") or []
    if isinstance(declared, str):
        declared = [declared]
    positive_ids: List[str] = []
    for pid in [*[str(x) for x in declared if x], *[v.uid for v in positive_views if v.uid]]:
        if pid not in positive_ids:
            positive_ids.append(pid)

    group_id = obj.get("group_id") or obj.get("cluster_id")

    return Example(
        query=query_views[0],
        positive=positive_views[0],
        negatives=negatives,
        task=str(obj.get("task", "default")),
        extra=extra,
        query_views=query_views,
        positive_views=positive_views,
        example_uid=str(example_uid) if example_uid else None,
        group_id=str(group_id) if group_id else None,
        query_uid=str(obj["query_id"]) if obj.get("query_id") else None,
        positive_ids=positive_ids,
    )


def count_views(obj: Dict[str, Any], role: str = "query") -> int:
    """Count views without fully parsing (avoids parsing the whole file when building view-group indices)."""
    if role == "query":
        raw = _pick(obj, "query_views", "queries", "query", "anchor", "q")
    else:
        raw = _pick(obj, "positive_views", "positives", "positive", "pos", "positive_document")
    if raw is None:
        return 0
    return len(raw) if isinstance(raw, (list, tuple)) else 1
