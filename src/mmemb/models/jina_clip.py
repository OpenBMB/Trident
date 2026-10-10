"""jina-clip-v2 encoder (dual tower); default checkpoint: `jinaai/jina-clip-v2`.

Fundamental difference from the Qwen backbone (read this before editing)
---------------------------------------------------------------------------
Qwen3-VL is a single tower: text and image are packed into one sequence, go
through one LLM, and last-token pooling is applied. jina-clip-v2 has two towers:

    text  : jina-XLM-RoBERTa (RoPE, 8k context) -> mean pooling -> 1024-d
    vision: EVA-02 ViT (512x512, patch 14)      -> CLS token    -> 1024-d

Both towers output into the same semantic space, with **no cross-modal attention**:

  * a Record with text only -> text tower only;
  * image only -> vision tower only;
  * text + image (fused view) -> each tower produces an embedding, combined
    according to `fusion`. jina-clip has no native fusion; the fusion weights are
    added by this framework (the default 0.5/0.5 normalized sum is the most
    stable starting point; Table 2 of the paper).

Key properties of jina-clip-v2:

  * the text limit is **not 64 tokens**: RoPE supports up to 8192 tokens
    (official inference truncates to 512 by default);
  * the text tower is itself a strong multilingual retrieval model;
  * MRL: embeddings can be truncated to any of [32, 64, 128, 256, 512, 768, 1024].

Implementation details
----------------------
1. **Do not pass `attn_implementation`.** JinaCLIPModel does not use the
   transformers attention interface and raises
   `ValueError: JinaCLIPModel does not support Flash Attention 2.0 yet`.
   flash-attn (text tower) / xformers (vision tower) are **used automatically
   when installed**, controlled by use_text_flash_attn / use_vision_xformers in the config.

2. **Pass `torch_dtype=`, not `dtype=`.** JinaCLIPPreTrainedModel overrides
   from_pretrained with `if 'torch_dtype' not in kwargs: kwargs['torch_dtype'] = 'auto'`.
   On transformers 4.57, passing the new name `dtype=` would add a conflicting
   `torch_dtype='auto'`, and the requested bf16 would be silently ignored. Hence the
   framework-wide load_hf_pretrained() is not used here.

3. **The text tower computes its own attention mask** as `(input_ids != pad_token_id)`.
   The tokenizer's pad_token_id must therefore match text_config.pad_token_id,
   otherwise mean pooling averages over padding (**silent quality loss**). This is
   validated at startup, and attention_mask is intentionally **not** passed.

4. **Gradient checkpointing is only enabled for the text tower.** EVA-02's
   rel_pos_bias handling is incompatible with checkpoint recomputation, so vision
   checkpointing is disabled (see `gradient_checkpointing_enable`). Use
   `image.patch_dropout` or a smaller batch to save memory on the vision side.

5. **Image size must equal vision_config.image_size** (PatchEmbed asserts it).
   A mismatch between the processor size and the model is reported at startup.

DDP considerations (handled in this file)
-----------------------------------------
In text-only batches no vision parameter is used, and DDP raises "parameters
that were not used in producing loss". With `keep_towers_alive` (on by default)
a tiny dummy image is forwarded in such batches and its output is added with
weight 0: the parameters join the graph with zero gradients, so values are
unaffected and `ddp_find_unused_parameters` (slow) is not needed.

The text tower **always** participates in the forward: every Record is
tokenized (an empty string for records without text) and a `has_text` mask
removes empty-string outputs from the fusion.

`logit_scale` is never used (only get_text_features / get_image_features are
called), so it is frozen; the temperature is controlled by loss.temperature.

Environment
-----------
transformers 4.57.x + timm + einops (the jina-clip remote code depends on timm).
Requires `trust_remote_code=true`.

See configs/trident_jinaclip.yaml for an example configuration.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from ..data.schema import Record
from ..registry import MODELS
from ..utils.env import require_model_env
from ..utils.misc import get_logger, resolve_dtype
from .base import BaseEmbedder
from .image_io import load_pil, resolve_path

logger = get_logger(__name__)

# Default truncation length of the official encode_text. The model (RoPE) supports 8192,
# but training and inference must use the same length, so the default is stated explicitly.
DEFAULT_MAX_LENGTH = 512

# MRL dimensions of jina-clip-v2 (config.matryoshka_dimensions).
# Truncating to other dimensions does not fail, but quality is not guaranteed by MRL.
MATRYOSHKA_DIMS = (32, 64, 128, 256, 512, 768, 1024)

# Default LoRA targets (Linear names differ between the two towers; this is the union):
#   text tower jina-XLM-RoBERTa (flash-attn BERT style): Wqkv / out_proj / fc1 / fc2
#   vision tower EVA-02                                 : qkv or q/k/v_proj, proj,
#                                                         w1/w2/w3 (SwiGLU) or fc1/fc2
# Note: Qwen-style o_proj / gate_proj names match nothing here.
DEFAULT_LORA_TARGETS = [
    # text tower
    "Wqkv", "out_proj",
    # vision tower
    "qkv", "q_proj", "k_proj", "v_proj", "proj", "w1", "w2", "w3",
    # may exist in both towers
    "fc1", "fc2",
]


@MODELS.register("jina_clip")
@MODELS.register("jina_clip_v2")
@MODELS.register("jina_clip_v1")   # v1 uses the same remote code, so the implementation is shared
class JinaCLIPEncoder(BaseEmbedder):
    """jina-clip-v2 dual-tower encoder (text / image / fused text-image)."""

    # ------------------------------------------------------------------ init
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__(cfg)
        require_model_env(str(cfg.get("type", "jina_clip")))
        from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoTokenizer

        path = cfg["pretrained_model_name_or_path"]
        # The jina-clip model definition is Hub remote code; there is no alternative
        trust = bool(cfg.get("trust_remote_code", True))
        if not trust:
            raise ValueError(
                "The jina-clip model definition is Hub remote code; "
                "model.trust_remote_code=true is required to load it."
            )
        # To pin the remote-code version (recommended for reproducibility):
        #   model.code_revision: <commit sha of jina-clip-implementation>
        code_revision = cfg.get("code_revision")
        revision = cfg.get("revision")
        hub_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if revision:
            hub_kwargs["revision"] = revision
        if code_revision:
            hub_kwargs["code_revision"] = code_revision

        # ---------------- text ----------------
        self.max_length = int(cfg.get("max_length", DEFAULT_MAX_LENGTH))
        # The text tower computes attention_mask from pad_token_id, so `longest` padding is
        # sufficient (no fixed-length padding needed).
        self.padding = str(cfg.get("padding", "longest"))
        # jina-clip-v2 is not instruction-tuned, so instructions are **not** prepended by default
        # (Table 2 of the paper: Instruction = None).
        self.use_instruction = bool(cfg.get("use_instruction", False))
        self.empty_text = str(cfg.get("empty_text", ""))
        self._warned_truncate = False

        # ---------------- image ----------------
        image_cfg = cfg.get("image") or {}
        self.max_images_per_record = int(image_cfg.get("max_images_per_record", 1))
        self.image_root = cfg.get("image_root") or ""

        # ---------------- fusion ----------------
        # sum   : weighted sum of the two L2-normalized tower outputs (default, most stable)
        # mean  : equivalent to sum with text_weight=0.5
        # text_only / image_only : use a single tower (for single-modality ablations)
        self.fusion = str(cfg.get("fusion", "sum")).lower()
        if self.fusion not in ("sum", "mean", "text_only", "image_only"):
            raise ValueError(f"Unknown model.fusion: {self.fusion}")
        self.text_weight = float(cfg.get("fusion_text_weight", 0.5))
        if self.fusion == "mean":
            self.text_weight = 0.5
        self.keep_towers_alive = bool(cfg.get("keep_towers_alive", True))

        # ---------------- tokenizer / image processor ----------------
        # AutoProcessor is intentionally **not** used: JinaCLIPProcessor inherits from
        # CLIPProcessor, whose __call__ / validation logic changed several times in 4.5x
        # and can fail during processor initialization. The model's own get_tokenizer /
        # get_preprocess also load the two parts separately.
        proc_path = cfg.get("processor_name_or_path") or path
        self.tokenizer = AutoTokenizer.from_pretrained(
            proc_path, **hub_kwargs
        )
        self.image_processor = AutoImageProcessor.from_pretrained(
            proc_path, use_fast=True, **hub_kwargs
        )

        # ---------------- backbone ----------------
        hf_cfg = AutoConfig.from_pretrained(path, **hub_kwargs)
        embed_dim = int(getattr(hf_cfg, "projection_dim", 0)) or int(
            hf_cfg.text_config.embed_dim
        )
        text_dim = int(hf_cfg.text_config.embed_dim)
        vis_dim = int(hf_cfg.vision_config.embed_dim)
        if text_dim != vis_dim:
            raise RuntimeError(
                f"Text tower ({text_dim}) and vision tower ({vis_dim}) output dimensions differ "
                "and cannot be fused in one space. Please check that the weights are standard jina-clip."
            )

        # Optional MRL truncation, applied to **each tower's output** before normalization and
        # fusion (MRL is nested, so the first k dimensions are a usable embedding).
        self.truncate_dim = cfg.get("truncate_dim")
        self.truncate_dim = int(self.truncate_dim) if self.truncate_dim else None
        if self.truncate_dim:
            if self.truncate_dim > embed_dim:
                raise ValueError(
                    f"truncate_dim={self.truncate_dim} exceeds the model output dimension {embed_dim}"
                )
            if self.truncate_dim not in MATRYOSHKA_DIMS:
                logger.warning(
                    "truncate_dim=%d is not one of the official MRL dimensions %s; quality is not guaranteed",
                    self.truncate_dim,
                    list(MATRYOSHKA_DIMS),
                )
            embed_dim = self.truncate_dim
        logger.info(
            "jina-clip embedding dim=%d%s",
            embed_dim,
            f" (MRL-truncated from {text_dim})" if self.truncate_dim else " (text=vision)",
        )

        if cfg.get("attn_implementation"):
            # Not silently ignored: tell the user explicitly that this field has no effect
            # for this backbone, so they do not assume flash-attn is enabled.
            logger.warning(
                "model.attn_implementation=%s has no effect for jina-clip and is ignored. "
                "flash-attn (text tower) and xformers (vision tower) are enabled automatically when installed, "
                "controlled by use_text_flash_attn / use_vision_xformers in the config.",
                cfg.get("attn_implementation"),
            )

        dtype = resolve_dtype(cfg.get("dtype", "bfloat16"))
        # See note 2 in the module docstring: torch_dtype= is required; load_hf_pretrained() is not used
        self.backbone = AutoModel.from_pretrained(
            path, torch_dtype=dtype, **hub_kwargs
        )

        self._check_pad_token(hf_cfg)
        self._check_image_size(hf_cfg)
        self._init_projection(embed_dim)

        # ---------------- logit_scale ----------------
        # JinaCLIPModel.logit_scale is only used by the CLIP loss in model.forward().
        # This framework only calls get_text_features / get_image_features, so it never
        # enters the graph -> DDP would report "parameters that were not used".
        scale = getattr(self._core(), "logit_scale", None)
        if isinstance(scale, torch.nn.Parameter):
            scale.requires_grad_(False)
            logger.info("logit_scale frozen (temperature is controlled by loss.temperature)")

        # ---------------- freezing ----------------
        if bool(cfg.get("freeze_vision", False)):
            for p in self._vision_module().parameters():
                p.requires_grad_(False)
            logger.info("Vision tower frozen")
        if bool(cfg.get("freeze_text", False)):
            for p in self._text_module().parameters():
                p.requires_grad_(False)
            logger.info("Text tower frozen")

        # ---------------- patch dropout (memory / speed knob) ----------------
        # Randomly drops a fraction of patch tokens during training (recommended by the
        # jina authors to trade for batch size). Only set when > 0: with prob==0 in training
        # mode PatchDropout returns a tensor instead of (tensor, indices).
        patch_dropout = float(image_cfg.get("patch_dropout", 0.0) or 0.0)
        if patch_dropout > 0:
            self._set_patch_dropout(patch_dropout)

        # Gradient checkpointing is enabled by the Trainer via gradient_checkpointing_enable().
        if cfg.get("gradient_checkpointing", False):
            logger.info(
                "gradient_checkpointing=true detected; "
                "the Trainer will call gradient_checkpointing_enable() at startup"
            )

        lora_cfg = cfg.get("lora") or {}
        self.merge_lora_on_save = bool(lora_cfg.get("merge_on_save", True))
        if lora_cfg.get("enable"):
            self._apply_lora(lora_cfg)

        # Dummy image cache (for keep_towers_alive), built lazily on first use
        self._dummy_pixel_values: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ startup checks
    def _check_pad_token(self, hf_cfg: Any) -> None:
        """The tokenizer's pad_token_id must match the text tower's (see note 3).

        Note that the **inner** config is used: HFTextEncoder sets
        `self.config = self.transformer.config` (the jina-XLM-RoBERTa config), not the
        outer JinaCLIPTextConfig, whose pad_token_id is null in the released weights.
        """
        model_pad = getattr(self._text_module().config, "pad_token_id", None)
        if model_pad is None:
            model_pad = getattr(hf_cfg.text_config, "pad_token_id", None)
        if model_pad is None:
            return
        tok_pad = self.tokenizer.pad_token_id
        if tok_pad is None:
            raise RuntimeError(
                "The tokenizer has no pad_token_id, so the text tower cannot derive attention_mask. "
                "Please make sure the tokenizer shipped with jina-clip is used."
            )
        if int(tok_pad) != int(model_pad):
            raise RuntimeError(
                f"tokenizer.pad_token_id={tok_pad} does not match the model's "
                f"text_config.pad_token_id={model_pad}.\n"
                "  The jina-clip text tower computes attention_mask as (input_ids != pad_token_id); "
                "a mismatch averages padding into mean pooling (no error, silent quality loss).\n"
                "  Usually processor_name_or_path points to a different model."
            )

    def _check_image_size(self, hf_cfg: Any) -> None:
        """The processor size must equal the vision tower's image_size (PatchEmbed asserts it)."""
        model_size = getattr(hf_cfg.vision_config, "image_size", None)
        proc_size = getattr(self.image_processor, "size", None)
        if model_size is None or proc_size is None:
            return
        if isinstance(proc_size, dict):  # standard HF image processor layout
            proc_size = proc_size.get("shortest_edge") or proc_size.get("height")
        if isinstance(proc_size, (list, tuple)):
            proc_size = proc_size[0]
        if proc_size is not None and int(proc_size) != int(model_size):
            raise RuntimeError(
                f"The image processor outputs {proc_size}x{proc_size}, but the vision tower requires "
                f"{model_size}x{model_size}. EVA's PatchEmbed asserts this, "
                "so the first forward would crash.\n"
                "  Common cause: processor_name_or_path points to jina-clip-v1 (224) "
                "while the weights are v2 (512)."
            )

    # ------------------------------------------------------------------ helpers
    def _core(self):
        """Return the underlying JinaCLIPModel (unwrapping LoRA if needed)."""
        model = self.backbone
        return getattr(model, "base_model", model) if hasattr(model, "peft_config") else model

    def _vision_module(self):
        core = self._core()
        if hasattr(core, "vision_model"):
            return core.vision_model
        raise RuntimeError("vision_model not found on the backbone")

    def _text_module(self):
        core = self._core()
        if hasattr(core, "text_model"):
            return core.text_model
        raise RuntimeError("text_model not found on the backbone")

    def _param_dtype(self) -> torch.dtype:
        return next(self.backbone.parameters()).dtype

    def _set_patch_dropout(self, prob: float) -> None:
        vision = self._vision_module()
        module = getattr(vision, "patch_dropout", None)
        if module is not None and hasattr(module, "prob"):
            module.prob = prob
            logger.info("Vision tower patch_dropout=%.2f", prob)
            return
        # With patch_dropout=0 in the weights the layer is nn.Identity and must be recreated.
        # PatchDropout is defined in the remote-code eva_model module; take it from there.
        try:
            import sys

            eva_mod = sys.modules[type(vision).__module__]
            vision.patch_dropout = eva_mod.PatchDropout(prob)
            logger.info("Vision tower patch_dropout=%.2f (new PatchDropout)", prob)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to set patch_dropout (%s); leaving it unchanged", e)

    def gradient_checkpointing_enable(self) -> None:
        """Handle gradient_checkpointing_enable calls from the Trainer or external code.

        Enabled for the text tower only; disabled for the vision tower because EVA-02's
        rel_pos_bias computation is incompatible with checkpoint recomputation. Memory on
        the vision side can be saved via patch dropout or a smaller batch instead.
        """
        # ---- text tower: enable ----
        text = self._text_module()
        inner = getattr(text, "transformer", None)
        if inner is not None and hasattr(inner, "gradient_checkpointing_enable"):
            try:
                try:
                    inner.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                except TypeError:
                    inner.gradient_checkpointing_enable()
                if hasattr(inner, "enable_input_require_grads"):
                    inner.enable_input_require_grads()
                logger.info("Text tower gradient checkpointing enabled")
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to enable text tower gradient checkpointing: %s", e)

        # ---- vision tower: disable ----
        vision = self._vision_module()
        if hasattr(vision, "grad_checkpointing"):
            vision.grad_checkpointing = False
        logger.warning(
            "Vision tower gradient checkpointing disabled "
            "(EVA-02 is incompatible with checkpointing; use patch_dropout or a smaller batch to save memory)"
        )

    def gradient_checkpointing_disable(self) -> None:
        """Disable all gradient checkpointing."""
        vision = self._vision_module()
        if hasattr(vision, "grad_checkpointing"):
            vision.grad_checkpointing = False

        text = self._text_module()
        inner = getattr(text, "transformer", None)
        if inner is not None and hasattr(inner, "gradient_checkpointing_disable"):
            try:
                inner.gradient_checkpointing_disable()
            except Exception:  # noqa: BLE001
                pass

    def _apply_lora(self, lora_cfg: Dict[str, Any]) -> None:
        import re

        from peft import LoraConfig, get_peft_model

        targets = list(lora_cfg.get("target_modules") or DEFAULT_LORA_TARGETS)
        exclude = lora_cfg.get("exclude_regex")
        pattern = re.compile(exclude) if exclude else None
        # Attach to selected towers only: lora.towers: [text] / [vision] / [text, vision]
        towers = [str(t).lower() for t in (lora_cfg.get("towers") or ["text", "vision"])]

        full_names: List[str] = []
        seen_leaf: Dict[str, int] = {}
        for name, module in self.backbone.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            in_text = ".text_model." in f".{name}."
            in_vision = ".vision_model." in f".{name}."
            if in_text and "text" not in towers:
                continue
            if in_vision and "vision" not in towers:
                continue
            leaf = name.split(".")[-1]
            seen_leaf[leaf] = seen_leaf.get(leaf, 0) + 1
            if leaf not in targets:
                continue
            if pattern is not None and pattern.search(name):
                continue
            full_names.append(name)

        if not full_names:
            raise RuntimeError(
                f"LoRA target_modules={targets} matched nothing.\n"
                f"  Linear leaf names in the model (name: count): {seen_leaf}\n"
                "  Note that jina-clip towers use different names: text tower Wqkv/out_proj/fc1/fc2, "
                "vision tower qkv (or q_proj/k_proj/v_proj)/proj/w1/w2/w3."
            )

        self.backbone = get_peft_model(
            self.backbone,
            LoraConfig(
                r=int(lora_cfg.get("r", 16)),
                lora_alpha=int(lora_cfg.get("alpha", 32)),
                lora_dropout=float(lora_cfg.get("dropout", 0.05)),
                bias="none",
                task_type=None,
                target_modules=full_names,
                modules_to_save=lora_cfg.get("modules_to_save"),
            ),
        )
        logger.info("LoRA attached to %d Linear layers (towers=%s)", len(full_names), towers)
        if hasattr(self.backbone, "print_trainable_parameters"):
            self.backbone.print_trainable_parameters()

    # ------------------------------------------------------------------ input construction
    def _text_of(self, rec: Record) -> str:
        text = (rec.text or "").strip()
        if self.use_instruction and rec.instruction:
            text = f"{rec.instruction.strip()} {text}".strip()
        return text

    def build_inputs(self, records: Sequence[Record], role: str = "query") -> Dict[str, Any]:
        texts: List[str] = []
        has_text: List[bool] = []
        images: List[Any] = []
        owners: List[int] = []          # image j belongs to Record owners[j]

        for i, rec in enumerate(records):
            text = self._text_of(rec)
            has_text.append(bool(text))
            # Always reserve a text slot: the text tower must always run, otherwise DDP reports unused parameters
            texts.append(text or self.empty_text)

            for p in (rec.images or [])[: self.max_images_per_record]:
                images.append(load_pil(resolve_path(p, self.image_root)))
                owners.append(i)

        tok = self.tokenizer(
            texts,
            padding=self.padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        self._warn_if_truncated(texts)

        # Only input_ids are passed: the text tower derives the mask from pad_token_id (see note 3);
        # an attention_mask would be silently swallowed by **kwargs anyway.
        batch: Dict[str, Any] = {
            "input_ids": tok["input_ids"],
            "has_text": torch.tensor(has_text, dtype=torch.bool),
        }

        if images:
            proc = self.image_processor(images=images)
            batch["pixel_values"] = proc["pixel_values"]
            batch["image_owner"] = torch.tensor(owners, dtype=torch.long)

        return batch

    def _warn_if_truncated(self, texts: Sequence[str]) -> None:
        if self._warned_truncate:
            return
        for t in texts:
            if not t:
                continue
            n = len(self.tokenizer(t, add_special_tokens=True)["input_ids"])
            if n > self.max_length:
                logger.warning(
                    "Text truncated to %d tokens (original length %d). The jina-clip-v2 text tower uses RoPE "
                    "and supports up to 8192 tokens, so model.max_length can be increased -- "
                    "but training and inference must use the same value.",
                    self.max_length,
                    n,
                )
                self._warned_truncate = True
                return

    # ------------------------------------------------------------------ forward
    def _text_features(self, inputs: Dict[str, Any]) -> torch.Tensor:
        return self._core().get_text_features(input_ids=inputs["input_ids"])

    def _image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self._core().get_image_features(
            pixel_values=pixel_values.to(self._param_dtype())
        )

    def _dummy_pixels(self, device: torch.device) -> torch.Tensor:
        """An all-black image used to keep the vision tower "used" in text-only batches.

        Generated through the image_processor, so its size matches the model (PatchEmbed asserts it).
        """
        if self._dummy_pixel_values is None:
            from PIL import Image

            proc = self.image_processor(
                images=[Image.new("RGB", (64, 64), (0, 0, 0))]
            )
            self._dummy_pixel_values = proc["pixel_values"]
        return self._dummy_pixel_values.to(device)

    def _maybe_truncate(self, feats: torch.Tensor) -> torch.Tensor:
        if self.truncate_dim:
            return feats[:, : self.truncate_dim]
        return feats

    def encode_features(self, **inputs) -> torch.Tensor:
        input_ids = inputs["input_ids"]
        bsz = input_ids.size(0)
        device = input_ids.device

        # ---- text tower: always runs ----
        txt = self._maybe_truncate(self._text_features(inputs)).float()
        txt = F.normalize(txt, p=2, dim=-1)
        has_text = inputs.get("has_text")
        if has_text is None:
            has_text = torch.ones(bsz, dtype=torch.bool, device=device)
        has_text = has_text.to(device).view(-1).bool()

        dim = txt.size(-1)
        img = torch.zeros(bsz, dim, device=device, dtype=txt.dtype)
        has_image = torch.zeros(bsz, dtype=torch.bool, device=device)
        alive_term: Optional[torch.Tensor] = None

        owner = inputs.get("image_owner")
        pixel_values = inputs.get("pixel_values")
        if owner is not None and owner.numel() > 0 and pixel_values is not None:
            feats = self._maybe_truncate(self._image_features(pixel_values)).float()
            feats = F.normalize(feats, p=2, dim=-1)
            owner = owner.to(device).view(-1)
            counts = torch.zeros(bsz, device=device, dtype=feats.dtype)
            counts.index_add_(0, owner, torch.ones_like(owner, dtype=feats.dtype))
            img = img.index_add(0, owner, feats)
            img = img / counts.clamp(min=1.0).unsqueeze(-1)
            img = F.normalize(img, p=2, dim=-1)          # re-normalize after averaging multiple images
            has_image = counts > 0
        elif self.keep_towers_alive and self.training:
            # Text-only batch: run a dummy image to keep the vision tower in the graph (zero gradient)
            dummy = self._image_features(self._dummy_pixels(device))
            alive_term = dummy.float().sum() * 0.0

        pooled = self._fuse(txt, img, has_text, has_image)
        if alive_term is not None:
            pooled = pooled + alive_term
        return self.post_pool(pooled)

    def _fuse(
        self,
        txt: torch.Tensor,
        img: torch.Tensor,
        has_text: torch.Tensor,
        has_image: torch.Tensor,
    ) -> torch.Tensor:
        """Combine the two tower embeddings into one.

        Rules (applied to normalized embeddings):
          text + image -> text_weight * txt + (1 - text_weight) * img
          one side     -> use that side (no shrinking for a missing modality; re-normalized later)
          neither      -> fall back to the text tower's output for the empty string (never an
                          all-zero vector, which would produce NaN in InfoNCE)
        """
        ht = has_text.float().unsqueeze(-1)
        hi = has_image.float().unsqueeze(-1)

        if self.fusion == "text_only":
            # The vision tower was still run (so DDP sees its parameters used), but with weight 0
            return txt + img * 0.0
        if self.fusion == "image_only":
            # Records without images (e.g. text-only queries) fall back to the text tower; otherwise all zeros
            return img * hi + txt * (1.0 - hi)

        both = (ht * hi)
        w_txt = self.text_weight * both + ht * (1.0 - both)
        w_img = (1.0 - self.text_weight) * both + hi * (1.0 - both)
        fused = w_txt * txt + w_img * img
        # If both are empty (should not happen), fall back to the text tower output to avoid zeros
        empty = ((ht + hi) == 0).float()
        return fused + empty * txt

    # ------------------------------------------------------------------ saving
    def _save_backbone(self, save_dir: str) -> None:
        backbone = self.backbone
        is_peft = hasattr(backbone, "peft_config")

        if is_peft and self.merge_lora_on_save:
            merged = backbone.merge_and_unload()
            merged.save_pretrained(save_dir, safe_serialization=True)
            logger.info("LoRA merged into the backbone before saving (lora.merge_on_save=true)")
        else:
            backbone.save_pretrained(save_dir)
            if is_peft:
                logger.info("Only the LoRA adapter was saved (lora.merge_on_save=false)")

        self.tokenizer.save_pretrained(save_dir)
        self.image_processor.save_pretrained(save_dir)
        logger.info(
            "The checkpoint can be loaded with `AutoModel.from_pretrained('%s', trust_remote_code=True)` "
            "(standard jina-clip format; auto_map in the config still points to the Hub remote code)",
            os.path.basename(save_dir.rstrip("/")) or save_dir,
        )