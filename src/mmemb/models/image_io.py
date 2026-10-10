"""Image loading: normalize the various image specifications in JSONL into PIL.Image.

Supported: local paths / file:// / http(s) / data:base64 / bytes / PIL objects /
dict wrappers such as {"image": ...}.

Kept in a separate module so all encoders share the same resolution rules
(e.g. for image_root and the file:// prefix). Depends only on Pillow, not on
transformers.
"""
from __future__ import annotations

import base64
import io
import os
from typing import Any, Optional


def load_pil(src: Any):
    """Load any supported specification into an RGB PIL.Image."""
    from PIL import Image

    if src is None:
        return None
    if hasattr(src, "convert"):                      # already a PIL.Image
        return src.convert("RGB")
    if isinstance(src, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(src))).convert("RGB")
    if isinstance(src, dict):
        return load_pil(
            src.get("image") or src.get("path") or src.get("url") or src.get("bytes")
        )
    if not isinstance(src, str):
        raise TypeError(f"Unsupported image type: {type(src)}")

    if src.startswith("data:"):
        return Image.open(io.BytesIO(base64.b64decode(src.split(",", 1)[1]))).convert("RGB")
    if src.startswith(("http://", "https://")):
        import urllib.request

        with urllib.request.urlopen(src, timeout=20) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")
    if src.startswith("file://"):
        src = src[len("file://"):]
    if not os.path.isfile(src):
        raise FileNotFoundError(f"Image not found: {src}")
    return Image.open(src).convert("RGB")


def resolve_path(path: str, image_root: Optional[str] = "") -> str:
    """Relative path + data.image_root -> loadable path."""
    if not isinstance(path, str):
        return path
    if os.path.isabs(path) or path.startswith(
        ("http://", "https://", "data:", "oss://", "file://")
    ):
        return path
    if image_root and not os.path.exists(path):
        return os.path.join(image_root, path)
    return path
