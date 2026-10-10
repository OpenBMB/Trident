#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
embedding_backends/base.py

Common abstract interface of all model backends. The driver script
(worker_main / prepare_batch / run_batch_on_gpu) only depends on these methods
and is agnostic to the concrete model, so adding a model only requires adding
a backend file.

Two rules for subclasses:
  1. build_inputs_cpu must never touch CUDA, so it can safely run in background
     threads (the driver prefetches batches with a thread pool).
  2. compute_from_inputs is the only method allowed to touch CUDA and is only
     called from the main thread.

GR-CLIP mean-shift calibration (see gr_clip.py): the base class automatically
wraps every subclass's compute_from_inputs and subtracts the modality mean of
each record from its output. Subclasses need **no** changes for this, except
backends using CLIP-style late fusion: they should call
`self.gr_fuse(t, i, role, index=i)` instead of
`clip_style_fuse(t, i, alpha=self.clip_fusion_alpha)`, because fused documents
must be mean-shifted per component *before* interpolation (see
common.clip_style_fuse). Without calibration gr_fuse behaves exactly like
clip_style_fuse.

Note (multi-job mode): the "role" key of self.args_dict is updated in place by
worker_main before each job (a backend is constructed once per process and
reused across jobs). Backends that need the current role should use the `role`
argument passed to compute_from_inputs / process instead of caching
self.args_dict["role"] on the instance.
"""

import abc
import functools
import threading
from typing import Any, Optional

import numpy as np

from .common import clip_style_fuse, l2_normalize
from .gr_clip import GRClipCalibrator, resolve_calibrator


class EmbeddingBackend(abc.ABC):
    supports_fused_text_image: bool = False
    max_images_per_record: int = 1
    clip_fusion_alpha: float = 0.5

    # Whether this backend is a CLIP-based (contrastive dual-tower) model.
    #
    # GR-CLIP mean-shift calibration is only defined for this family, so the
    # default is False and only the "### Clip-Based" backends in
    # embedding_backends/__init__.py (clip_openai / siglip2 / altclip /
    # jina_clip_v2 / trident_jinaclip) set it to True. Other backends skip
    # calibration even if GR_CLIP_MEANS is set (see gr_clip.resolve_calibrator;
    # GR_CLIP_ALLOW_NON_CLIP=1 overrides this).
    #
    # This is orthogonal to supports_fused_text_image:
    #   - is_clip_based: the architecture / training objective is a contrastive dual tower;
    #   - supports_fused_text_image: a text+image record is encoded into one fused
    #     embedding in a single forward.
    # trident_jinaclip has is_clip_based=True and supports_fused_text_image=True.
    is_clip_based: bool = False

    # Whether --matryoshka_dims is supported (several Matryoshka truncation
    # dimensions produced by a single forward).
    #
    # Currently True for trident_jinaclip / trident_qwen3vl / qwen3vl_official.
    # Supporting backends must:
    #   1. call self.setup_matryoshka(full_dim) at the end of load();
    #   2. in compute_from_inputs, return self.matryoshka_pack(raw) instead of a
    #      single-dimension embedding when self.matryoshka_pack_dims is not None.
    # See matryoshka_pack for the packing format; unpacking happens in merge_shards
    # of the driver script.
    supports_matryoshka: bool = False

    def __init__(self, args_dict: dict, device: str):
        self.args_dict = args_dict
        self.device = device
        self.embedding_dim: Optional[int] = None

        # With Matryoshka enabled: dimensions of each packed segment of the
        # compute_from_inputs output (ascending; the last one is the full dim). None = disabled.
        self.matryoshka_pack_dims: Optional[list[int]] = None

        # build_inputs_cpu usually calls HuggingFace fast tokenizers (Rust
        # `tokenizers`), which are not thread-safe: concurrent encode /
        # apply_chat_template calls on the same instance crash with
        # `Exception: Already borrowed` (a Rust RefCell borrow failure that kills
        # the worker process).
        #
        # The driver calls build_inputs_cpu from two places, possibly concurrently:
        #   1) prepare_batch, from several background threads of a
        #      ThreadPoolExecutor(max_workers=prefetch_depth);
        #   2) per-item retries after a failed batch in run_batch_on_gpu, from the
        #      main thread via self.process() -> build_inputs_cpu.
        # This lock is shared by prepare_batch and process() to serialize all
        # tokenizer access.
        self.cpu_lock = threading.Lock()

        # GR-CLIP mean calibration. None when not configured, in which case all
        # calibration logic is skipped.
        self.gr_calibrator: Optional[GRClipCalibrator] = resolve_calibrator(
            args_dict, backend_cls=type(self)
        )
        # Indices of the current batch already calibrated inside the backend (registered
        # by gr_fuse); the generic post-processing skips them to avoid subtracting twice.
        # compute_from_inputs runs serially on the main thread, so a plain set suffices.
        self._gr_precalibrated: set[int] = set()

        if self.gr_calibrator is not None and args_dict.get("_gr_logged") is None:
            args_dict["_gr_logged"] = True
            print(
                f"[gr-clip] mean-shift calibration ENABLED "
                f"({self.gr_calibrator.describe()})",
                flush=True,
            )

    def __init_subclass__(cls, **kwargs) -> None:
        """
        Automatically wrap the subclass's own compute_from_inputs with the
        calibration post-processing.

        __init_subclass__ is used instead of a template method because the driver
        (run_batch_on_gpu) calls backend.compute_from_inputs directly rather than
        going through process(); wrapping the method itself at class creation
        guarantees both paths are calibrated, with no runtime overhead.
        """
        super().__init_subclass__(**kwargs)

        raw = cls.__dict__.get("compute_from_inputs")
        if raw is None or getattr(raw, "_gr_clip_wrapped", False):
            # Not overridden (inherited and already wrapped), or already wrapped.
            return

        @functools.wraps(raw)
        def wrapped(self, prepared, items, role):
            self._gr_precalibrated = set()
            array, token_infos = raw(self, prepared, items, role)
            array = self.apply_gr_calibration(array, items, role)
            return array, token_infos

        wrapped._gr_clip_wrapped = True
        cls.compute_from_inputs = wrapped

    # ------------------------------------------------------------------
    # GR-CLIP mean calibration
    # ------------------------------------------------------------------

    def gr_fuse(
        self,
        text_vec: np.ndarray,
        image_vec: np.ndarray,
        role: str,
        index: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """
        Replacement for common.clip_style_fuse in backends using CLIP-style late fusion.

        Without calibration this equals clip_style_fuse(text_vec, image_vec, alpha).
        With calibration (Algorithm 1 of GR-CLIP), the role-specific text / image
        means are subtracted from each component before interpolation, and `index`
        is registered in _gr_precalibrated so the post-processing skips it.

        role='doc'   -> e_bar_T / e_bar_I
        role='query' -> e_bar_q / e_bar_qI (query_image_mean; falls back to e_bar_I
                        if missing, see gr_clip.py)

        Fused text+image queries (e.g. OVEN) therefore use the same code path as
        documents, with no special casing for callers.
        """
        alpha = self.clip_fusion_alpha if alpha is None else alpha
        calibrator = self.gr_calibrator

        if calibrator is None:
            return clip_style_fuse(text_vec, image_vec, alpha=alpha)

        calibrator.check_dim(int(np.asarray(text_vec).shape[-1]), backend_name=self.name)

        if index is not None:
            self._gr_precalibrated.add(int(index))

        return clip_style_fuse(
            text_vec,
            image_vec,
            alpha=alpha,
            text_mean=calibrator.text_mean_for_role(role),
            image_mean=calibrator.image_mean_for_role(role),
        )

    def gr_fused_mean(self, role: str) -> np.ndarray:
        """
        Mean to subtract for fused text+image records, for both queries and documents
        (role selects the set of means). Used by both native-fusion and late-fusion
        backends through apply_gr_calibration:

          - if self.supports_fused_text_image is True (one record is natively encoded
            into one fused vector, e.g. qwen3vl_official / trident_qwen3vl) and the
            means file contains a native fused mean for the role (doc -> fused_mean /
            e_bar_F, query -> query_fused_mean / e_bar_qF, computed by
            compute_gr_clip_calibration_means.py with native fused forwards), use it;
          - otherwise (late-fusion backend, or no fused mean available) fall back to
            the interpolation α·image_mean(role) + (1-α)·text_mean(role).

        self.gr_calibrator must not be None (guaranteed by the caller).
        """
        calibrator = self.gr_calibrator
        assert calibrator is not None
        if self.supports_fused_text_image and calibrator.has_native_fused_mean(role):
            return calibrator.native_fused_mean_for_role(role)
        return calibrator.interpolated_fused_mean(role=role, alpha=self.clip_fusion_alpha)

    # Backward-compatible alias of gr_fused_mean (previously doc-side only).
    def gr_fused_doc_mean(self, role: str) -> np.ndarray:
        return self.gr_fused_mean(role)

    def apply_gr_calibration(
        self,
        array: np.ndarray,
        items: list[dict],
        role: str,
    ) -> np.ndarray:
        """
        Apply mean calibration to a batch: subtract the mean matching each record's
        (role, modality), then re-normalize. Indices already calibrated by gr_fuse
        are skipped.

        `role` selects the query-side or doc-side means. Both sides are symmetric
        over three modalities (text / image / fused), so datasets with image
        queries (e.g. Nights) or fused queries (e.g. OVEN) need no special casing.

        The modality is determined from the input record only (presence of text /
        images), not from the backend's internals, so this also works for backends
        that natively encode text+image into one vector
        (supports_fused_text_image=True). The mean subtracted for fused records is
        chosen by gr_fused_mean(): the native fused mean if available, otherwise the
        interpolation α·image_mean(role) + (1-α)·text_mean(role)
        (α = clip_fusion_alpha, default 0.5).
        """
        calibrator = self.gr_calibrator
        if calibrator is None:
            return array

        array = np.asarray(array, dtype=np.float32)

        if array.ndim != 2 or array.shape[0] != len(items):
            # Should never happen (the driver maps results by offset). Do not skip
            # silently: a misalignment would subtract the wrong mean from records.
            raise RuntimeError(
                f"backend '{self.name}': GR-CLIP calibration got embeddings with "
                f"shape={array.shape}, which does not match the {len(items)} records."
            )

        calibrator.check_dim(array.shape[1], backend_name=self.name)

        out = array.copy()
        for i, item in enumerate(items):
            if i in self._gr_precalibrated:
                continue

            has_text = bool(item.get("text"))
            has_image = bool(item.get("images"))

            if has_text and has_image:
                mean = self.gr_fused_mean(role)
            elif has_image:
                mean = calibrator.image_mean_for_role(role)
            elif has_text:
                mean = calibrator.text_mean_for_role(role)
            else:
                # Records with neither text nor images should never get here; leave them untouched.
                continue

            out[i] = out[i] - mean

        # Mean subtraction changes the norm; re-normalize so all stored embeddings are
        # unit vectors (rows that were not modified are already unit vectors: no-op).
        return l2_normalize(out)

    # ------------------------------------------------------------------
    # Matryoshka multi-dimension evaluation
    # ------------------------------------------------------------------

    def setup_matryoshka(self, full_dim: int) -> None:
        """
        Determine the packed output dimensions from args_dict["matryoshka_dims"];
        call at the end of load() once the full dimension is known.

        - Values larger than the full dimension are dropped with a notice (the same
          list can be used for jina-clip (1024) and qwen3-vl (2048)).
        - The full dimension is always packed as the last segment (even if not
          listed); the driver writes it to the top-level embeddings.npy so the
          regular outputs / evaluation pipeline stay unchanged.
        """
        requested = self.args_dict.get("matryoshka_dims")
        if not requested:
            self.matryoshka_pack_dims = None
            return

        if not self.supports_matryoshka:
            raise ValueError(
                f"backend '{self.name}' does not support --matryoshka_dims"
            )

        if self.args_dict.get("dim"):
            raise ValueError("--dim and --matryoshka_dims cannot be used together")

        if self.gr_calibrator is not None:
            # GR-CLIP means are computed for a single dimension (checked by check_dim);
            # each truncation dimension would need its own means, which is not supported.
            raise ValueError(
                "--matryoshka_dims cannot be combined with GR-CLIP mean calibration "
                "(GR_CLIP_MEANS): the means correspond to a single dimension. Unset "
                "GR_CLIP_MEANS, or run each dimension separately with --dim."
            )

        full_dim = int(full_dim)
        dims = sorted({int(d) for d in requested})
        kept = [d for d in dims if 0 < d <= full_dim]
        dropped = [d for d in dims if d > full_dim]

        if dropped:
            print(
                f"[matryoshka] backend '{self.name}' full dim is {full_dim}; "
                f"ignoring larger dims {dropped}",
                flush=True,
            )

        if not kept:
            raise ValueError(
                f"--matryoshka_dims={dims} all exceed the model's full dim {full_dim}"
            )

        self.matryoshka_pack_dims = kept if kept[-1] == full_dim else kept + [full_dim]
        print(
            f"[matryoshka] output dims: {kept} (full dim {full_dim})",
            flush=True,
        )

    def matryoshka_pack(self, full: np.ndarray) -> np.ndarray:
        """
        Pack full-dimension embeddings into one row [d_1 | d_2 | ... | d_full]:
        each segment = first d_i dims of the raw embedding, L2-normalized separately.

        This is element-wise identical to running with --dim d_i separately (which
        also truncates then normalizes) but needs a single forward; truncation is
        done in float32 to avoid float16 precision loss.
        """
        pack_dims = self.matryoshka_pack_dims
        assert pack_dims is not None

        full = np.asarray(full, dtype=np.float32)
        if full.ndim != 2 or full.shape[1] != pack_dims[-1]:
            raise RuntimeError(
                f"backend '{self.name}': matryoshka expects full dim {pack_dims[-1]}, "
                f"got output shape={full.shape} (encoder.embedding_dim inconsistent with the output?)"
            )

        return np.concatenate(
            [l2_normalize(full[:, :d]) for d in pack_dims], axis=1,
        )

    @abc.abstractmethod
    def load(self) -> None:
        """Load the model / processor onto self.device. Model-specific imports go here."""
        raise NotImplementedError

    def validate_item(
        self,
        text: Optional[str],
        images: list[str],
        role: str,
    ) -> Optional[str]:
        num_images = len(images)

        if num_images > self.max_images_per_record:
            return (
                f"backend '{self.name}' supports at most "
                f"{self.max_images_per_record} image(s) per record, "
                f"got {num_images}."
            )

        return None

    @property
    def name(self) -> str:
        return self.args_dict.get("model_type", self.__class__.__name__)

    def default_instruction(self, role: str) -> str:
        return ""

    @abc.abstractmethod
    def build_inputs_cpu(self, items: list[dict]) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def compute_from_inputs(
        self,
        prepared: Any,
        items: list[dict],
        role: str,
    ) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
        raise NotImplementedError

    def process(
        self,
        items: list[dict],
        role: str,
    ) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
        """build_inputs_cpu + compute_from_inputs in one call (used for per-item retries)."""
        # Per-item retries run on the main thread and may race with background prefetch
        # threads on the tokenizer, so the same cpu_lock must be used (see __init__).
        with self.cpu_lock:
            prepared = self.build_inputs_cpu(items)
        return self.compute_from_inputs(prepared, items, role)

    def render_prompt(
        self,
        text: Optional[str],
        images: list[str],
        instruction: Optional[str],
        role: str,
    ) -> str:
        return text or "<no text>"
