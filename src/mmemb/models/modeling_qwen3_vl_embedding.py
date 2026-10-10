# coding=utf-8
"""Qwen3-VL Embedding **model definition**.

Adapted from the official Qwen3-VL-Embedding implementation
(`qwen3_vl_embedding.py`, Qwen Team, Apache License 2.0).

This file serves two purposes:

1. during training, `mmemb/models/qwen3_vl_embedding.py` imports it as the backbone;
2. `save_pretrained` **copies this file verbatim into the checkpoint directory** and
   writes ``auto_map`` into config.json, so trained checkpoints can be loaded with
   plain transformers:

       from transformers import AutoModel, AutoProcessor
       model = AutoModel.from_pretrained(ckpt, trust_remote_code=True)

   Therefore this file **may only depend on torch + transformers**; it must not
   import framework modules or `qwen_vl_utils` (a data-side utility that
   downstream users may not have installed).

Differences from the official implementation:
  * the `Qwen3VLEmbedder` data-preprocessing class is removed; that logic lives in
    the framework encoder, which integrates with the Record / instruction system;
  * two inference helpers, `pool()` and `embed()`, are added for downstream use;
  * everything else (class names, `_checkpoint_conversion_mapping`, forward
    signature, output structure) matches the official code, so weight keys are
    interchangeable with official checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLModel,
    Qwen3VLPreTrainedModel,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

__all__ = ["Qwen3VLForEmbeddingOutput", "Qwen3VLForEmbedding", "last_token_pool"]


@dataclass
class Qwen3VLForEmbeddingOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.Tensor] = None


def last_token_pool(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Hidden state of the last valid token of each sequence.

    Same implementation as the official one (`attention_mask.flip(1).argmax(1)`);
    correct for both left and right padding.
    """
    flipped = attention_mask.flip(dims=[1])
    last_one_positions = flipped.argmax(dim=1)
    col = attention_mask.shape[1] - last_one_positions - 1
    row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
    return hidden_state[row, col]


class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
    """Qwen3-VL embedding backbone without lm_head.

    Weight keys match the `model.*` prefix of `Qwen3VLForConditionalGeneration`,
    so `from_pretrained("Qwen/Qwen3-VL-8B-Instruct")` works directly (lm_head is
    ignored), and official `Qwen3-VL-Embedding` weights can be loaded as well.
    """

    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False
    # Note: this file uses `from __future__ import annotations` (PEP 563), which turns
    # annotations such as `config: Qwen3VLConfig` into the string "Qwen3VLConfig" at
    # runtime. transformers 5.x derives `config_class` from the subclass's `config`
    # annotation; a string there makes `config_class.from_pretrained(...)` fail with
    # `'str' object has no attribute 'from_pretrained'`. Hence config_class is
    # assigned explicitly here; the annotation below is kept only for type hints.
    config_class = Qwen3VLConfig
    config: Qwen3VLConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    # ---------------------------------------------------------------- submodule passthrough
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLForEmbeddingOutput]:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        return Qwen3VLForEmbeddingOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )

    # ---------------------------------------------------------------- inference helpers
    @torch.no_grad()
    def embed(self, normalize: bool = True, **inputs) -> torch.Tensor:
        """One step: processor outputs -> [B, D] embeddings.

        Usage:
            inputs = processor(text=..., images=..., return_tensors="pt").to(model.device)
            emb = model.embed(**inputs)
        """
        attention_mask = inputs.get("attention_mask")
        out = self(**inputs, use_cache=False)
        pooled = last_token_pool(out.last_hidden_state, attention_mask)
        pooled = pooled.float()
        if normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled

    @staticmethod
    def pool(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return last_token_pool(hidden_state, attention_mask)