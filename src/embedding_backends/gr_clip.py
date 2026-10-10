#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
embedding_backends/gr_clip.py

Post-hoc mean-shift calibration of GR-CLIP ("Closing the Modality Gap for Mixed
Modality Search", Li et al., 2025, Algorithm 1), used as a baseline in our paper.

===============================================================================
Scope: CLIP-based (contrastive dual-tower) models only
===============================================================================
  GR-CLIP's analysis (the modality gap is approximately a constant offset
  orthogonal to the image / text subspaces, so subtracting per-modality means
  removes it) is derived for contrastive dual-tower models such as CLIP /
  SigLIP. VLM-based embedding models encode a text+image record into one
  autoregressively fused vector without separable subspaces, and GR-CLIP does
  not claim the mean shift applies to them.

  Calibration is therefore only enabled for backend classes declaring
  `is_clip_based = True` (the "### Clip-Based" group in
  embedding_backends/__init__.py: clip_openai / siglip2 / altclip /
  jina_clip_v2 / trident_jinaclip). Other backends skip it even when
  GR_CLIP_MEANS is set (with a notice, not an error), so one set of environment
  variables can be used for all models. To force it for a non-CLIP backend:
      export GR_CLIP_ALLOW_NON_CLIP=1
  (or pass gr_clip_allow_non_clip=True in args_dict).

===============================================================================
What it does
===============================================================================
  src/tools/compute_gr_clip_calibration_means.py computes the means and stores
  them in gr_clip_means.npz, bucketed by **(role, modality)**:

      role=query  text   query_mean        (e_bar_q)
      role=query  image  query_image_mean  (e_bar_qI)   [optional]
      role=query  fused  query_fused_mean  (e_bar_qF)   [optional, native-fusion models only]
      role=doc    text   text_mean         (e_bar_T)
      role=doc    image  image_mean        (e_bar_I)
      role=doc    fused  fused_mean        (e_bar_F)    [optional, native-fusion models only]

  This module loads the npz and, following Algorithm 1, subtracts the matching
  mean from every backend output before it is written to disk:

      query text          : e = f_T(q)   - query_mean
      query image         : e = f_I(q)   - query_image_mean
      query fused (late-fusion backend)
                          : e = α·(f_I(q_I) - query_image_mean)
                                + (1-α)·(f_T(q_T) - query_mean)
      doc text            : e = f_T(d)   - text_mean
      doc image           : e = f_I(d)   - image_mean
      doc fused (late-fusion backend)
                          : e = α·(f_I(d_I) - image_mean)
                                + (1-α)·(f_T(d_T) - text_mean)
      fused (native-fusion backend, supports_fused_text_image=True)
                          : subtract the role's *_fused_mean directly (no separable
                            components); falls back to the interpolation above if
                            no fused mean is available.

  All embeddings are L2-normalized again afterwards, keeping the convention that
  stored embeddings are unit vectors (cosine ranking is unaffected).

  Note on α: here α is the **image** weight (fused = α·image + (1-α)·text),
  consistent with common.clip_style_fuse. This is just a different
  parameterization of the same interpolation; fusion and means use the same α.

  Queries are bucketed by modality as well, because query images / fused
  queries (e.g. OVEN, Nights) follow a different distribution than document
  images, just as query text differs from document text. query_image_mean /
  query_fused_mean are optional; when missing (e.g. text-only query sets), the
  doc-side means are reused with a one-time [gr-clip] notice.

===============================================================================
How to enable (priority: args_dict > environment variable)
===============================================================================
  1) Environment variable (recommended):
         export GR_CLIP_MEANS=/path/to/gr_clip_means.npz
     All scripts using the backends (e.g. src/eval/embed_jsonl_unified_multigpu.py)
     then enable calibration automatically; spawned worker processes inherit it.
     To disable temporarily: export GR_CLIP_DISABLE=1 (highest priority).

  2) Or pass a key in args_dict:
         args_dict["gr_clip_means"] = "/path/to/gr_clip_means.npz"

Important: calibration must be disabled while computing the means
(compute_gr_clip_calibration_means.py reuses the same backends and forces
GR_CLIP_DISABLE=1 for this reason).
"""

import os
import threading
from typing import Optional

import numpy as np

# Environment variable names (see module docstring)
ENV_MEANS_PATH = "GR_CLIP_MEANS"
ENV_DISABLE = "GR_CLIP_DISABLE"
ENV_ALLOW_NON_CLIP = "GR_CLIP_ALLOW_NON_CLIP"

# args_dict keys
ARG_MEANS_PATH = "gr_clip_means"
ARG_ENABLE = "gr_clip_enable"
ARG_ALLOW_NON_CLIP = "gr_clip_allow_non_clip"

# Keys in the npz file: three required, three optional.
KEY_QUERY_TEXT = "query_mean"          # e_bar_q
KEY_QUERY_IMAGE = "query_image_mean"   # e_bar_qI  (optional)
KEY_QUERY_FUSED = "query_fused_mean"   # e_bar_qF  (optional)
KEY_DOC_TEXT = "text_mean"             # e_bar_T
KEY_DOC_IMAGE = "image_mean"           # e_bar_I
KEY_DOC_FUSED = "fused_mean"           # e_bar_F   (optional)

REQUIRED_KEYS = (KEY_QUERY_TEXT, KEY_DOC_TEXT, KEY_DOC_IMAGE)
OPTIONAL_KEYS = (KEY_QUERY_IMAGE, KEY_QUERY_FUSED, KEY_DOC_FUSED)

_CACHE: dict[str, "GRClipCalibrator"] = {}
_CACHE_LOCK = threading.Lock()


def _as_str(value) -> str:
    """Strings in npz files are 0-d numpy arrays; convert them to Python str."""
    try:
        return str(np.asarray(value).item())
    except Exception:  # noqa: BLE001
        return str(value)


def _env_flag(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in ("1", "true", "yes")


def _is_query(role: str) -> bool:
    return role == "query"


class GRClipCalibrator:
    """
    Holds the mean vectors bucketed by (role, modality) and provides lookups by
    role + modality.

    Each npz is loaded once per process (see the cache in load_calibrator); the
    vectors are read-only, so sharing one object across backends is safe.
    """

    def __init__(self, path: str, data: dict):
        self.path = path

        def _vec(key: str) -> np.ndarray:
            vec = np.asarray(data[key], dtype=np.float32)
            if vec.ndim != 1:
                raise ValueError(
                    f"{path}: '{key}' should be a 1-d vector, got shape={vec.shape}"
                )
            return vec

        # ---- required ----
        self.query_text_mean: np.ndarray = _vec(KEY_QUERY_TEXT)
        self.doc_text_mean: np.ndarray = _vec(KEY_DOC_TEXT)
        self.doc_image_mean: np.ndarray = _vec(KEY_DOC_IMAGE)

        self.dim: int = self.query_text_mean.shape[0]
        for name, vec in (
            (KEY_DOC_TEXT, self.doc_text_mean),
            (KEY_DOC_IMAGE, self.doc_image_mean),
        ):
            if vec.shape[0] != self.dim:
                raise ValueError(
                    f"{path}: dimension of '{name}' ({vec.shape[0]}) does not match "
                    f"'{KEY_QUERY_TEXT}' ({self.dim})"
                )

        # ---- optional ----
        # query_image_mean / query_fused_mean: unavailable for text-only query sets;
        #   the doc-side mean of the same modality is used instead.
        # fused_mean: only computed for backends with supports_fused_text_image=True.
        def _optional_vec(key: str) -> Optional[np.ndarray]:
            if key not in data:
                return None
            vec = _vec(key)
            if vec.shape[0] != self.dim:
                raise ValueError(
                    f"{path}: dimension of '{key}' ({vec.shape[0]}) does not match "
                    f"the other means ({self.dim})."
                )
            return vec

        self.query_image_mean: Optional[np.ndarray] = _optional_vec(KEY_QUERY_IMAGE)
        self.query_fused_mean: Optional[np.ndarray] = _optional_vec(KEY_QUERY_FUSED)
        self.doc_fused_mean: Optional[np.ndarray] = _optional_vec(KEY_DOC_FUSED)

        self.model_type: str = _as_str(data["model_type"]) if "model_type" in data else ""
        self.checkpoint: str = _as_str(data["checkpoint"]) if "checkpoint" in data else ""
        self.created_at: str = _as_str(data["created_at"]) if "created_at" in data else ""

        # Fallback notices are printed only once each.
        self._warned: set[str] = set()

    # ---- read-only aliases for the doc-side / query-side means ----

    @property
    def query_mean(self) -> np.ndarray:
        return self.query_text_mean

    @property
    def text_mean(self) -> np.ndarray:
        return self.doc_text_mean

    @property
    def image_mean(self) -> np.ndarray:
        return self.doc_image_mean

    @property
    def fused_mean(self) -> Optional[np.ndarray]:
        return self.doc_fused_mean

    # ---- internal: print a notice once ---------------------------------

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        print(f"[gr-clip] {message}", flush=True)

    # ---- mean lookup by (role, modality) -------------------------------

    def text_mean_for_role(self, role: str) -> np.ndarray:
        """e_bar_q for role='query', e_bar_T otherwise (doc)."""
        return self.query_text_mean if _is_query(role) else self.doc_text_mean

    def image_mean_for_role(self, role: str) -> np.ndarray:
        """
        For role='query', use query_image_mean (e_bar_qI); if missing (text-only
        queries during calibration), fall back to the doc-side image_mean with a
        one-time notice.
        """
        if not _is_query(role):
            return self.doc_image_mean

        if self.query_image_mean is not None:
            return self.query_image_mean

        self._warn_once(
            "query_image_fallback",
            "Image found on the query side, but the means file has no query_image_mean "
            "(e_bar_qI); reusing the doc-side image_mean. To remove this bias, "
            "re-run compute_gr_clip_calibration_means.py on data with query images.",
        )
        return self.doc_image_mean

    def has_native_fused_mean(self, role: str) -> bool:
        """Whether a native fused mean is available for the role (its own, or the doc-side fallback)."""
        if _is_query(role):
            return self.query_fused_mean is not None or self.doc_fused_mean is not None
        return self.doc_fused_mean is not None

    def native_fused_mean_for_role(self, role: str) -> np.ndarray:
        """
        Native fused mean, computed by compute_gr_clip_calibration_means.py with
        native fused forwards of backends with supports_fused_text_image=True.
        Check has_native_fused_mean(role) first.

        For role='query', query_fused_mean (e_bar_qF) is preferred; otherwise the
        doc-side fused_mean is used with a one-time notice.
        """
        if _is_query(role):
            if self.query_fused_mean is not None:
                return self.query_fused_mean
            self._warn_once(
                "query_fused_fallback",
                "Native fused record found on the query side, but the means file has no "
                "query_fused_mean (e_bar_qF); reusing the doc-side fused_mean.",
            )
            assert self.doc_fused_mean is not None
            return self.doc_fused_mean

        assert self.doc_fused_mean is not None
        return self.doc_fused_mean

    def interpolated_fused_mean(self, role: str, alpha: float) -> np.ndarray:
        """
        Interpolated mean for fused records: α·image_mean(role) + (1-α)·text_mean(role).

        Used (1) by late-fusion backends (separable components; gr_fuse follows
        Algorithm 1 exactly) and (2) as an approximation for native-fusion backends
        when no fused mean is available.

        α has the same meaning as in common.clip_style_fuse (image weight).
        """
        alpha = float(alpha)
        return (
            alpha * self.image_mean_for_role(role)
            + (1.0 - alpha) * self.text_mean_for_role(role)
        ).astype(np.float32)

    # ---- checks --------------------------------------------------------

    def check_dim(self, dim: int, backend_name: str = "") -> None:
        if dim != self.dim:
            raise ValueError(
                f"GR-CLIP mean dimension ({self.dim}) does not match the embedding "
                f"dimension ({dim}) of backend{' ' + backend_name if backend_name else ''}. "
                f"Means file: {self.path} "
                f"(model_type={self.model_type!r}, checkpoint={self.checkpoint!r}). "
                f"Means must be recomputed with the same model and the same --dim."
            )

    def check_model(self, model_type: str, checkpoint: Optional[str] = None) -> None:
        """A model mismatch only warns (one may deliberately try another checkpoint of the same architecture)."""
        if self.model_type and model_type and self.model_type != model_type:
            print(
                f"[gr-clip][warn] the means file was computed with model_type={self.model_type!r}, "
                f"current is {model_type!r}; the modality gap is model-specific, so reusing means across models is likely invalid.",
                flush=True,
            )
        if checkpoint and self.checkpoint and self.checkpoint != checkpoint:
            print(
                f"[gr-clip][warn] the means file's checkpoint={self.checkpoint!r} "
                f"differs from the current checkpoint={checkpoint!r}.",
                flush=True,
            )

    def describe(self) -> str:
        available = [KEY_QUERY_TEXT, KEY_DOC_TEXT, KEY_DOC_IMAGE]
        if self.query_image_mean is not None:
            available.append(KEY_QUERY_IMAGE)
        if self.query_fused_mean is not None:
            available.append(KEY_QUERY_FUSED)
        if self.doc_fused_mean is not None:
            available.append(KEY_DOC_FUSED)
        return (
            f"path={self.path}, dim={self.dim}, model_type={self.model_type!r}, "
            f"created_at={self.created_at!r}, means={available}"
        )


def load_calibrator(path: str) -> GRClipCalibrator:
    """Load (and cache by path) a gr_clip_means.npz file."""
    key = os.path.abspath(path)

    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached

    if not os.path.isfile(key):
        raise FileNotFoundError(
            f"GR-CLIP means file not found: {key}. "
            f"Generate gr_clip_means.npz with src/tools/compute_gr_clip_calibration_means.py first."
        )

    with np.load(key, allow_pickle=False) as data:
        missing = [k for k in REQUIRED_KEYS if k not in data]
        if missing:
            raise KeyError(
                f"{key}: missing required keys {missing}; "
                f"found {list(data.keys())}. "
                f"This file should be generated by compute_gr_clip_calibration_means.py."
            )
        payload = {k: data[k] for k in data.files}

    calibrator = GRClipCalibrator(key, payload)

    with _CACHE_LOCK:
        _CACHE.setdefault(key, calibrator)
        return _CACHE[key]


def resolve_calibrator(
    args_dict: dict,
    backend_cls: Optional[type] = None,
) -> Optional[GRClipCalibrator]:
    """
    Decide whether GR-CLIP calibration is enabled, from args_dict / environment
    variables / backend type.

    Priority (high to low):
      1. GR_CLIP_DISABLE=1 / args_dict["gr_clip_enable"] is False -> disabled
      2. backend is not CLIP-based (is_clip_based != True) and not explicitly allowed -> disabled
      3. args_dict["gr_clip_means"]
      4. environment variable GR_CLIP_MEANS

    backend_cls=None skips the CLIP-based check (e.g. offline tools reading the means file).

    Returns None when calibration is disabled.
    """
    if _env_flag(ENV_DISABLE):
        return None

    if args_dict.get(ARG_ENABLE) is False:
        return None

    path = args_dict.get(ARG_MEANS_PATH) or os.environ.get(ENV_MEANS_PATH)
    if not path:
        return None

    path = str(path).strip()
    if not path:
        return None

    # ---- CLIP-based gate (see "Scope" in the module docstring) ----
    if backend_cls is not None and not getattr(backend_cls, "is_clip_based", False):
        allow = bool(args_dict.get(ARG_ALLOW_NON_CLIP)) or _env_flag(ENV_ALLOW_NON_CLIP)
        if not allow:
            if args_dict.get("_gr_non_clip_logged") is None:
                args_dict["_gr_non_clip_logged"] = True
                print(
                    f"[gr-clip] backend "
                    f"{args_dict.get('model_type', backend_cls.__name__)!r} "
                    f"is not CLIP-based (is_clip_based=False); skipping GR-CLIP mean calibration, "
                    f"which is only defined for contrastive dual-tower models. "
                    f"Set {ENV_ALLOW_NON_CLIP}=1 to force it.",
                    flush=True,
                )
            return None

    calibrator = load_calibrator(path)
    calibrator.check_model(
        model_type=args_dict.get("model_type", ""),
        checkpoint=args_dict.get("checkpoint"),
    )
    return calibrator
