#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
embedding_backends/common.py

Low-level helpers shared by all backends: image loading (including parallel
loading), torch dtype parsing, vector normalization, CLIP-style late fusion,
and a safe "model output -> numpy" conversion.

Design principle: this file only imports light dependencies (torch / numpy).
Heavy model libraries (transformers / peft / ...) are imported inside each
backend's load() method, so loading one backend never pulls in the
dependencies of the others.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import numpy as np
import torch


def is_remote_path(path: str) -> bool:
    return path.startswith(("http://", "https://", "data:"))


def load_pil_image(path: str):
    """
    CPU only: open a path (local file or http(s) URL) as a PIL.Image.

    Note: `.convert("RGB")` returns a new PIL.Image without the original
    `.filename` attribute. Some downstream debugging code reads
    `image.filename` to report which image failed; restoring it here ensures the
    original exception is shown instead of an unrelated AttributeError.
    """
    from PIL import Image

    if is_remote_path(path):
        import io
        import urllib.request

        with urllib.request.urlopen(path, timeout=30) as response:
            data = response.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
        image.filename = path
        return image

    image = Image.open(path).convert("RGB")
    image.filename = path
    return image


def load_images_parallel(
    paths: list[Optional[str]],
    max_workers: int = 8,
) -> list[Optional[Any]]:
    """
    Load the (at most one) image of each record in a batch in parallel.

    paths[i] is None / empty if record i has no image; the corresponding result
    is None. PIL decoding and network I/O release the GIL, so a thread pool
    reduces the wall-clock time from O(batch_size) to O(batch_size / workers).

    If an image fails to load, future.result() raises and the exception
    propagates to prepare_batch, which retries the batch item by item.
    """
    results: list[Optional[Any]] = [None] * len(paths)
    indices_to_load = [i for i, p in enumerate(paths) if p]

    if not indices_to_load:
        return results

    workers = max(1, min(max_workers, len(indices_to_load)))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(load_pil_image, paths[i]): i
            for i in indices_to_load
        }
        for future, idx in future_to_idx.items():
            results[idx] = future.result()

    return results


def ensure_transformers_special_tokens_compat() -> None:
    """
    Compatibility patch: some checkpoints (and some recently exported Qwen
    tokenizer_config.json files) store "extra_special_tokens" in the list format
    introduced in transformers v5, e.g.

        "extra_special_tokens": ["<|image_pad|>", "<|video_pad|>", ...]

    With transformers v4.x, `_set_model_specific_special_tokens` only accepts the
    dict format (token name -> token string) and fails with

        AttributeError: 'list' object has no attribute 'keys'

    This is a known transformers version issue (huggingface/transformers #45376,
    #44781), not a broken checkpoint. The patch converts the list into a dict
    (each token mapped to itself) before calling the original method.
    It must be called once in every worker subprocess, before
    AutoProcessor.from_pretrained / AutoTokenizer.from_pretrained.

    Older transformers versions (e.g. 4.40.2, required by visrag_ret) do not have
    `_set_model_specific_special_tokens` and never hit this issue, so the patch is
    skipped when the method does not exist.
    """
    try:
        from transformers import PreTrainedTokenizerBase
    except ImportError:
        return

    if not hasattr(PreTrainedTokenizerBase, "_set_model_specific_special_tokens"):
        # Older transformers (e.g. 4.40.2) lack this method and are unaffected; skip.
        return

    original = PreTrainedTokenizerBase._set_model_specific_special_tokens

    if getattr(original, "_extra_special_tokens_list_patch", False):
        # Already patched (e.g. several backends in one process called this function).
        return

    def patched(self, special_tokens, *args, **kwargs):
        if isinstance(special_tokens, list):
            special_tokens = {token: token for token in special_tokens}
        return original(self, special_tokens, *args, **kwargs)

    patched._extra_special_tokens_list_patch = True
    PreTrainedTokenizerBase._set_model_specific_special_tokens = patched


def ensure_tokenizers_thread_safety() -> None:
    """
    Compatibility patch: a HuggingFace fast tokenizer (Rust `tokenizers`) instance
    called concurrently from several threads crashes with

        Exception: Already borrowed

    This is a Rust borrow-check failure, not a Python exception, so
    `except Exception` cannot catch it and the worker process dies.

    The driver's prefetch threads already serialize build_inputs_cpu calls with
    backend.cpu_lock, but some backend libraries may spawn their own internal
    threads that share one tokenizer. As a last line of defense, the leaf methods
    of the Rust core `tokenizers.Tokenizer` (encode / encode_batch /
    encode_batch_fast) are wrapped **at class level** with a process-wide lock,
    serializing every access regardless of which thread makes it.

    This lock only wraps leaf calls and is independent of backend.cpu_lock; the
    outer lock never waits on the inner one in the same thread, so no deadlock
    cycle is possible.
    """
    try:
        from tokenizers import Tokenizer as RustTokenizer
    except ImportError:
        return

    lock = threading.Lock()

    def make_patched(original):
        def patched(self, *args, **kwargs):
            with lock:
                return original(self, *args, **kwargs)

        patched._thread_safety_patch = True
        return patched

    for method_name in ("encode", "encode_batch", "encode_batch_fast"):
        original = getattr(RustTokenizer, method_name, None)

        if original is None or getattr(original, "_thread_safety_patch", False):
            continue

        try:
            setattr(RustTokenizer, method_name, make_patched(original))
        except (AttributeError, TypeError):
            # Some environments may not allow patching this extension type; skip this method.
            continue


def ensure_mistral_regex_disabled() -> None:
    """
    Compatibility patch: around transformers 4.57, fast tokenizers gained an
    automatic "suspected Mistral regex" check (huggingface/transformers#42591,
    #44736, #44031). For tokenizers with vocab_size > 100000 whose local
    config.json either
        a) has no "transformers_version" field, or
        b) has an old "transformers_version" and no "model_type",
    a warning suggests passing `fix_mistral_regex=True`, and passing True replaces
    the pre_tokenizer with the Mistral tekken regex.

    The check misfires on non-Mistral models: e.g. jina-clip-v2 (whose text tower
    is XLM-RoBERTa based) ships a config.json without "transformers_version".
    By default this only produces a noisy warning, but for jina_clip_v2 /
    trident_jinaclip the tokenizer is loaded inside trust_remote_code model code,
    where we cannot pass fix_mistral_regex=False. If that code ever passed
    fix_mistral_regex=True, tokenization would diverge from training and
    embedding quality would drop.

    This function short-circuits the `_patch_mistral_regex` classmethod into
    "return the tokenizer unchanged" before the model library is imported, so the
    check / warning / patch never applies. Verified for both the transformers 4.x
    layout (method on PreTrainedTokenizerBase) and the 5.x layout (method on
    tokenization_utils_tokenizers.TokenizersBackend).

    Only called by the jina_clip_v2 / trident_jinaclip backends.
    """
    try:
        import transformers
    except ImportError:
        return

    target_cls = None

    # transformers 5.x: method on TokenizersBackend.
    try:
        from transformers.tokenization_utils_tokenizers import TokenizersBackend

        if hasattr(TokenizersBackend, "_patch_mistral_regex"):
            target_cls = TokenizersBackend
    except ImportError:
        pass

    # transformers 4.x (including 4.57.x): method on PreTrainedTokenizerBase.
    if target_cls is None:
        from transformers import PreTrainedTokenizerBase

        if hasattr(PreTrainedTokenizerBase, "_patch_mistral_regex"):
            target_cls = PreTrainedTokenizerBase

    if target_cls is None:
        # This version has no such check; nothing to patch.
        return

    if getattr(target_cls._patch_mistral_regex, "_noop_mistral_regex_patch", False):
        # Already patched (several backends in one process called this function).
        return

    def _noop(cls, tokenizer, *args, **kwargs):
        return tokenizer

    _noop._noop_mistral_regex_patch = True
    target_cls._patch_mistral_regex = classmethod(_noop)


def parse_torch_dtype(dtype: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }

    dtype = dtype.lower()

    if dtype not in mapping:
        raise ValueError(
            f"Unsupported dtype: {dtype}. "
            f"Choose from {sorted(mapping.keys())}."
        )

    return mapping[dtype]


def l2_normalize(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return (array / norms).astype(np.float32)


def clip_style_fuse(
    text_vec: np.ndarray,
    image_vec: np.ndarray,
    alpha: float = 0.5,
    text_mean: Optional[np.ndarray] = None,
    image_mean: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    CLIP-style late fusion for models whose official API cannot produce one
    fused embedding per record: text and image are encoded separately,
    L2-normalized, then combined as
    e_fused = normalize(alpha * e_image + (1 - alpha) * e_text).

    text_mean / image_mean are the GR-CLIP modality means (e_bar_T / e_bar_I).
    If given, they are subtracted from each normalized component **before** fusion:

        fused = normalize( α·(u_I - e_bar_I) + (1-α)·(u_T - e_bar_T) )

    This must happen before fusion: subtracting from the fused, re-normalized
    result would subtract a vector scaled by 1/||α·u_I + (1-α)·u_T||, which is not
    equivalent to "mean-shift each component, then interpolate" (Algorithm 1 of
    GR-CLIP) except at α=0 or α=1. This is why late-fusion backends call
    EmbeddingBackend.gr_fuse() instead of relying on the generic post-processing.

    Without means (default) this is plain late fusion.
    """
    text_vec = np.asarray(text_vec, dtype=np.float32)
    image_vec = np.asarray(image_vec, dtype=np.float32)

    text_norm = np.linalg.norm(text_vec)
    image_norm = np.linalg.norm(image_vec)

    text_unit = text_vec / text_norm if text_norm > 0 else text_vec
    image_unit = image_vec / image_norm if image_norm > 0 else image_vec

    if text_mean is not None:
        text_unit = text_unit - np.asarray(text_mean, dtype=np.float32)
    if image_mean is not None:
        image_unit = image_unit - np.asarray(image_mean, dtype=np.float32)

    fused = alpha * image_unit + (1.0 - alpha) * text_unit

    fused_norm = np.linalg.norm(fused)
    if fused_norm > 0:
        fused = fused / fused_norm

    return fused.astype(np.float32)


def to_numpy_f32(x: Any) -> np.ndarray:
    """
    Convert model outputs safely into a float32 numpy array on CPU.

    Handles three return types:
      * a list / tuple of tensors (e.g. some models return one GPU tensor per
        record): each element is converted and the results are stacked;
      * a single torch.Tensor (possibly on GPU): detached and moved to CPU;
      * anything else: np.asarray(x, dtype=float32).

    Calling np.asarray directly on CUDA tensors (or lists of them) raises
    "can't convert cuda:0 device type tensor to numpy", which would make every
    batch fall back to slow per-item retries.
    """
    if isinstance(x, (list, tuple)):
        arrays = [to_numpy_f32(item) for item in x]
        return np.stack(arrays, axis=0).astype(np.float32)

    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float32).cpu().numpy()

    return np.asarray(x, dtype=np.float32)