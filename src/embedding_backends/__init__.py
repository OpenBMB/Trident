#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
embedding_backends/__init__.py

Collects all model backends into BACKEND_REGISTRY.

To add a new model:
  1. create a file under embedding_backends/ with a class inheriting from
     EmbeddingBackend (from .base) that implements at least load /
     build_inputs_cpu / compute_from_inputs (see any existing backend);
  2. import it here and register it in BACKEND_REGISTRY below;
  3. no changes are needed in embed_jsonl_unified_multigpu.py, which only uses
     BACKEND_REGISTRY and the --model_type string key.
"""

### Clip-Based
from .altclip import AltCLIPBackend
from .clip_openai import OpenAICLIPBackend
from .siglip2 import Siglip2Backend
from .jina_clip_v2 import JinaClipV2Backend
from .trident_jinaclip import TridentJinaClipBackend

### LLM-Based
from .jina_v5_omni import JinaV5OmniBackend
from .qwen3vl_official import Qwen3VLOfficialBackend
from .unime_phi35v import UniMEPhi35VBackend
from .visrag_ret import VisragRetBackend
from .trident_qwen3vl import TridentQwen3VLBackend

BACKEND_REGISTRY: dict[str, type] = {
   "trident_qwen3vl": TridentQwen3VLBackend,
   "trident_jinaclip": TridentJinaClipBackend,
   "jina_v5_omni": JinaV5OmniBackend,
   "jina_clip_v2": JinaClipV2Backend,
   "qwen3vl_official": Qwen3VLOfficialBackend,
   "unime_phi35v": UniMEPhi35VBackend,
   "visrag_ret": VisragRetBackend,
   "clip_vit_l14": OpenAICLIPBackend,
   "siglip2": Siglip2Backend,
   "altclip": AltCLIPBackend,
}

__all__ = ["BACKEND_REGISTRY"]