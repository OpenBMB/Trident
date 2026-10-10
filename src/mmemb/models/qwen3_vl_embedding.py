"""Qwen3-VL-Embedding encoder (reuses the official Qwen model definition).

The backbone is `Qwen3VLForEmbedding` (see modeling_qwen3_vl_embedding.py):
  * weight keys are identical to official Qwen3-VL-Embedding checkpoints;
  * trained checkpoints can be loaded with `AutoModel.from_pretrained(..., trust_remote_code=True)`;
  * the input format matches the official `Qwen3VLEmbedder`.

Use it with `model.type: qwen3_vl_embedding`.

Input format (**identical to the official `Qwen3VLEmbedder.format_model_input`**):

    [{"role": "system", "content": [{"type": "text", "text": <instruction>}]},
     {"role": "user",   "content": [ {"type": "video", ...},      # only if present
                                     {"type": "image", ...},      # only if present
                                     {"type": "text", "text": ...}]}]

    apply_chat_template(..., add_generation_prompt=True)  ->  ends with
    `<|im_start|>assistant\\n`, which is the position used by last-token pooling.

Instructions (two are enough for retrieval, see Figure 13 of the paper):
    priority 1  Record.instruction (injected by the dataset from data.task_instructions)
    priority 2  model.prompt.query_instruction / doc_instruction
    priority 3  DEFAULT_QUERY_INSTRUCTION / DEFAULT_DOC_INSTRUCTION
    As in the official code, a `.` is appended if the instruction does not end with punctuation.

Memory and official constants:
    IMAGE_FACTOR = 32, min_pixels = 4*32*32, max_pixels = 1800*32*32.
    If GPU memory is insufficient, reduce `model.image.max_pixels` first (the biggest lever).

See configs/trident_qwen3vl.yaml for a full configuration.
"""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import unicodedata
from typing import Any, Dict, List, Optional, Sequence

import torch

from ..data.schema import Record
from ..registry import MODELS
from ..utils.misc import get_logger, load_hf_pretrained, resolve_dtype
from .base import BaseEmbedder
from .modeling_qwen3_vl_embedding import Qwen3VLForEmbedding

logger = get_logger(__name__)

# ------------------------------------------------------------------ official constants
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2          # 32
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR          # 4096
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR       # 1843200
FPS = 1.0
MAX_FRAMES = 64
FRAME_MAX_PIXELS = 768 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_TOTAL_PIXELS = 10 * FRAME_MAX_PIXELS

# Retrieval only needs two instructions: one for queries, one for documents
DEFAULT_QUERY_INSTRUCTION = "Represent the query for retrieving relevant content."
DEFAULT_DOC_INSTRUCTION = "Represent the candidate content for retrieval."

# Only these keys are passed to backbone.forward; extra processor outputs such as
# second_per_grid_ts / do_sample_frames are dropped.
_MODEL_INPUT_KEYS = {
    "input_ids",
    "attention_mask",
    "position_ids",
    "inputs_embeds",
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
    "cache_position",
    # Recent transformers (Qwen3VLModel.compute_3d_position_ids) require mm_token_type_ids
    # for M-RoPE whenever image_grid_thw / video_grid_thw are passed; otherwise they raise
    # "Multimodal data was passed ... but `mm_token_type_ids` is missing".
    "mm_token_type_ids",
}

# Default LoRA targets: language model only (the vision tower is frozen by default)
DEFAULT_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
DEFAULT_LORA_EXCLUDE_REGEX = r"visual|vision_tower"

_MODELING_FILE = "modeling_qwen3_vl_embedding.py"
_AUTO_MAP = {
    "AutoModel": f"{_MODELING_FILE[:-3]}.Qwen3VLForEmbedding",
    "AutoModelForCausalLM": f"{_MODELING_FILE[:-3]}.Qwen3VLForEmbedding",
}


# ------------------------------------------------------------------ media loading
def _load_pil(src: Any):
    from PIL import Image

    if src is None:
        return None
    if hasattr(src, "convert"):          # already a PIL.Image
        return src.convert("RGB")
    if isinstance(src, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(src))).convert("RGB")
    if isinstance(src, dict):
        return _load_pil(src.get("image") or src.get("path") or src.get("url") or src.get("bytes"))
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


def _as_uri(path: str) -> str:
    """qwen_vl_utils expects the `file://` prefix (as in the official format_model_input)."""
    if path.startswith(("http://", "https://", "oss://", "data:", "file://")):
        return path
    return "file://" + os.path.abspath(path)


@MODELS.register("qwen3_vl_embedding")
class Qwen3VLEmbeddingEncoder(BaseEmbedder):
    """Qwen3-VL-Embedding encoder (text / image / video)."""

    # ------------------------------------------------------------------ init
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__(cfg)
        from transformers import AutoConfig, AutoProcessor

        path = cfg["pretrained_model_name_or_path"]
        trust = bool(cfg.get("trust_remote_code", False))

        # ---------------- text / template ----------------
        self.pooling_mode = str(cfg.get("pooling", "last_token"))
        self.max_length = int(cfg.get("max_length", 8192))
        self.max_text_tokens = cfg.get("max_text_tokens")   # truncate text first when images are present, so image tokens are never cut
        self.image_root = cfg.get("image_root") or ""
        self.add_generation_prompt = bool(cfg.get("add_generation_prompt", True))

        prompt_cfg = cfg.get("prompt") or {}
        self.query_instruction = prompt_cfg.get("query_instruction", DEFAULT_QUERY_INSTRUCTION)
        self.doc_instruction = prompt_cfg.get("doc_instruction", DEFAULT_DOC_INSTRUCTION)
        # Retrieval uses only default.query / default.doc, but the task dimension is kept for extensibility
        data_cfg = cfg.get("data") or {}
        self.task_instructions = data_cfg.get("task_instructions") or {}

        # ---------------- vision / video ----------------
        image_cfg = cfg.get("image") or {}
        self.min_pixels = int(image_cfg.get("min_pixels", MIN_PIXELS))
        self.max_pixels = int(image_cfg.get("max_pixels", MAX_PIXELS))
        video_cfg = cfg.get("video") or {}
        self.fps = float(video_cfg.get("fps", FPS))
        self.max_frames = int(video_cfg.get("max_frames", MAX_FRAMES))
        self.total_pixels = int(video_cfg.get("total_pixels", MAX_TOTAL_PIXELS))

        # ---------------- processor ----------------
        proc_path = cfg.get("processor_name_or_path") or path
        self.processor = AutoProcessor.from_pretrained(
            proc_path,
            trust_remote_code=trust,
            # The official Qwen3VLEmbedder uses right padding; last-token pooling works with both,
            # but right padding is the default to match official inference.
            padding_side=str(cfg.get("padding_side", "right")),
        )
        if not hasattr(self.processor, "image_processor"):
            raise RuntimeError(
                "The loaded processor has no image_processor (probably only a tokenizer was loaded). "
                "Qwen3-VL requires the full AutoProcessor, otherwise images are silently dropped."
            )
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = str(cfg.get("padding_side", "right"))

        # With qwen_vl_utils, follow the official path (smart_resize outside, no resize in the processor);
        # otherwise let the image_processor resize. Both train fine; the former matches official inference pixel-exactly.
        self._vision_fn = self._resolve_vision_fn(cfg)
        if self._vision_fn is None:
            self._configure_image_processor()

        # ---------------- backbone ----------------
        hf_cfg = AutoConfig.from_pretrained(path, trust_remote_code=trust)
        hidden_size = self._infer_hidden_size(hf_cfg)
        logger.info("Qwen3-VL text hidden_size=%d", hidden_size)

        self.backbone = load_hf_pretrained(
            Qwen3VLForEmbedding.from_pretrained,
            path,
            dtype=resolve_dtype(cfg.get("dtype", "bfloat16")),
            attn_implementation=cfg.get("attn_implementation", "sdpa"),
        )
        # Qwen3VLForEmbedding has no lm_head, so drop_lm_head is not needed

        self._init_projection(hidden_size)
        if self.projection is not None:
            logger.warning(
                "model.embed_dim projection head enabled: the head is not part of the HF weights, "
                "so AutoModel-loaded checkpoints will return the raw %d-dim embeddings. "
                "Set embed_dim to null for checkpoints that work out of the box.",
                hidden_size,
            )

        # ---------------- freeze the vision tower ----------------
        # Frozen by default for two reasons: (1) in text-only batches the vision tower is unused,
        # and full fine-tuning under DDP raises "parameters that were not used in producing loss";
        # (2) contrastive fine-tuning of the vision tower is prone to collapse.
        if bool(cfg.get("freeze_vision", True)):
            for p in self.backbone.model.visual.parameters():
                p.requires_grad_(False)
            logger.info("Vision tower frozen")

        if cfg.get("gradient_checkpointing", False):
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(self.backbone, "enable_input_require_grads"):
                self.backbone.enable_input_require_grads()

        lora_cfg = cfg.get("lora") or {}
        self.merge_lora_on_save = bool(lora_cfg.get("merge_on_save", True))
        if lora_cfg.get("enable"):
            self._apply_lora(lora_cfg)

    # ------------------------------------------------------------------ build helpers
    @staticmethod
    def _infer_hidden_size(hf_cfg) -> int:
        for attr in ("text_config", "llm_config"):
            sub = getattr(hf_cfg, attr, None)
            if sub is not None and getattr(sub, "hidden_size", None):
                return int(sub.hidden_size)
        if getattr(hf_cfg, "hidden_size", None):
            return int(hf_cfg.hidden_size)
        raise RuntimeError(
            "Cannot infer hidden_size from config (expected config.text_config.hidden_size for Qwen3-VL)"
        )

    def _resolve_vision_fn(self, cfg: Dict[str, Any]):
        """Return qwen_vl_utils.process_vision_info, or None to use the fallback path."""
        if not bool(cfg.get("use_qwen_vl_utils", True)):
            logger.info("use_qwen_vl_utils=false; vision preprocessing uses the processor's built-in resize")
            return None
        try:
            from qwen_vl_utils.vision_process import process_vision_info

            logger.info("Vision preprocessing uses qwen_vl_utils (pixel-identical to official inference)")
            return process_vision_info
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Could not import qwen_vl_utils (%s). Falling back to the processor's built-in resize: "
                "training works, but image resizing will differ slightly from inference with the official "
                "qwen_vl_utils. Recommended: `pip install qwen-vl-utils`.",
                e,
            )
            return None

    def _configure_image_processor(self) -> None:
        """Fallback path: write min/max_pixels into the image_processor (the main memory lever)."""
        ip = self.processor.image_processor
        setattr(ip, "min_pixels", int(self.min_pixels))
        setattr(ip, "max_pixels", int(self.max_pixels))
        size = getattr(ip, "size", None)
        if isinstance(size, dict):
            try:
                size["shortest_edge"] = int(self.min_pixels)
                size["longest_edge"] = int(self.max_pixels)
                ip.size = size
            except (TypeError, KeyError):
                logger.warning("Unrecognized image_processor.size structure; only min/max_pixels attributes were set")
        logger.info("Image pixel limits: min_pixels=%d max_pixels=%d", self.min_pixels, self.max_pixels)

    def _apply_lora(self, lora_cfg: Dict[str, Any]) -> None:
        import re

        from peft import LoraConfig, get_peft_model

        targets = list(lora_cfg.get("target_modules") or DEFAULT_LORA_TARGETS)
        exclude = lora_cfg.get("exclude_regex", DEFAULT_LORA_EXCLUDE_REGEX)
        pattern = re.compile(exclude) if exclude else None

        # PEFT target_modules uses suffix matching and would also hit same-named Linear layers
        # in the vision tower, so targets are expanded into full names (also used for logging).
        full_names: List[str] = []
        for name, module in self.backbone.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            if name.split(".")[-1] not in targets:
                continue
            if pattern is not None and pattern.search(name):
                continue
            full_names.append(name)

        if not full_names:
            raise RuntimeError(
                f"LoRA target_modules={targets} matched nothing; please check named_modules"
            )

        peft_conf = LoraConfig(
            r=int(lora_cfg.get("r", 32)),
            lora_alpha=int(lora_cfg.get("alpha", 64)),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
            bias="none",
            task_type=None,
            target_modules=full_names,
            modules_to_save=lora_cfg.get("modules_to_save"),
        )
        self.backbone = get_peft_model(self.backbone, peft_conf)
        n_layers = len({n.split(".layers.")[1].split(".")[0] for n in full_names if ".layers." in n})
        logger.info("LoRA attached to %d Linear layers covering %d decoder layers", len(full_names), n_layers)
        if hasattr(self.backbone, "print_trainable_parameters"):
            self.backbone.print_trainable_parameters()

    # ------------------------------------------------------------------ instruction
    @staticmethod
    def _normalize_instruction(instruction: Optional[str]) -> Optional[str]:
        """Official behavior: append `.` if the instruction does not end with punctuation."""
        if not instruction:
            return None
        instruction = instruction.strip()
        if not instruction:
            return None
        if not unicodedata.category(instruction[-1]).startswith("P"):
            instruction += "."
        return instruction

    def _instruction_for(self, rec: Record, role: str) -> Optional[str]:
        # 1) injected into the Record by the dataset from data.task_instructions (the usual path)
        if getattr(rec, "instruction", None):
            return self._normalize_instruction(rec.instruction)
        # 2) model-side fallback: task_instructions.default.{query,doc}
        table = self.task_instructions.get("default") or {}
        if isinstance(table, dict) and table.get(role):
            return self._normalize_instruction(table[role])
        # 3) global prompt config
        fallback = self.query_instruction if role == "query" else self.doc_instruction
        return self._normalize_instruction(fallback)

    # ------------------------------------------------------------------ conversation assembly
    def _truncate_text(self, text: Optional[str]) -> Optional[str]:
        if not text or not self.max_text_tokens:
            return text
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= int(self.max_text_tokens):
            return text
        return self.tokenizer.decode(ids[: int(self.max_text_tokens)], skip_special_tokens=True)

    def _abs_path(self, p: str) -> str:
        if os.path.isabs(p) or p.startswith(("http://", "https://", "data:", "oss://", "file://")):
            return p
        if self.image_root and not os.path.exists(p):
            return os.path.join(self.image_root, p)
        return p

    def _videos_of(self, rec: Record) -> List[Any]:
        """Records have no `videos` field by default; pass it through if present (path / frame list)."""
        raw = getattr(rec, "videos", None) or getattr(rec, "video", None) or []
        if isinstance(raw, str):
            raw = [raw]
        return list(raw)

    def _format_conversation(self, rec: Record, role: str) -> List[Dict[str, Any]]:
        """Matches the official `Qwen3VLEmbedder.format_model_input`: instruction in system, content in user.

        Content order also matches the official code: video -> image -> text.
        """
        instruction = self._instruction_for(rec, role)
        content: List[Dict[str, Any]] = []

        for v in self._videos_of(rec):
            item: Dict[str, Any] = {"type": "video"}
            if isinstance(v, list):     # list of frames
                item["video"] = [_as_uri(self._abs_path(x)) if isinstance(x, str) else x for x in v]
                item["total_pixels"] = self.total_pixels
            else:
                item["video"] = _as_uri(self._abs_path(str(v)))
                item["fps"] = self.fps
                item["max_frames"] = self.max_frames
            content.append(item)

        for img in (rec.images or []):
            content.append(
                {
                    "type": "image",
                    "image": _as_uri(self._abs_path(img)) if isinstance(img, str) else img,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                }
            )

        text = self._truncate_text(rec.text)
        if text:
            content.append({"type": "text", "text": text})

        # Official fallback: insert "NULL" when empty so the sequence is non-empty (pooling needs a position)
        if not content:
            content.append({"type": "text", "text": "NULL"})

        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": instruction or DEFAULT_QUERY_INSTRUCTION}],
            },
            {"role": "user", "content": content},
        ]

    # ------------------------------------------------------------------ build_inputs
    def build_inputs(self, records: Sequence[Record], role: str = "query") -> Dict[str, Any]:
        conversations = [self._format_conversation(rec, role) for rec in records]
        texts = self.processor.apply_chat_template(
            conversations, add_generation_prompt=self.add_generation_prompt, tokenize=False
        )
        if isinstance(texts, str):     # some versions return a str for a single item
            texts = [texts]

        if self._vision_fn is not None:
            images, videos, video_metadata, video_kwargs = self._vision_via_utils(conversations)
            do_resize = False          # already smart_resized by qwen_vl_utils
        else:
            images = self._vision_fallback(conversations)
            videos, video_metadata, video_kwargs = None, None, {}
            do_resize = True

        kwargs: Dict[str, Any] = {
            "text": texts,
            "padding": True,
            "return_tensors": "pt",
            "do_resize": do_resize,
        }
        if images:
            kwargs["images"] = images
        if videos:
            kwargs["videos"] = videos
            if video_metadata is not None:
                kwargs["video_metadata"] = video_metadata
            kwargs.update(video_kwargs)

        # **Never** truncate when images/videos are present: cutting <|image_pad|> raises
        # "Image features and image tokens do not match". Text length has already been
        # truncated by tokens in _truncate_text.
        if not images and not videos:
            kwargs["truncation"] = True
            kwargs["max_length"] = self.max_length

        batch = self.processor(**kwargs)
        return dict(batch)

    def _vision_via_utils(self, conversations):
        """Official path: qwen_vl_utils.process_vision_info."""
        try:
            images, video_inputs, video_kwargs = self._vision_fn(
                conversations,
                image_patch_size=IMAGE_BASE_FACTOR,
                return_video_metadata=True,
                return_video_kwargs=True,
            )
        except TypeError:
            # older qwen_vl_utils versions lack image_patch_size / return_video_metadata
            images, video_inputs = self._vision_fn(conversations)
            video_kwargs = {}
            return images, video_inputs, None, video_kwargs

        videos, video_metadata = None, None
        if video_inputs:
            videos, video_metadata = zip(*video_inputs)
            videos, video_metadata = list(videos), list(video_metadata)
        return images, videos, video_metadata, (video_kwargs or {})

    def _vision_fallback(self, conversations):
        """Without qwen_vl_utils: load images as PIL and let the processor resize."""
        images = []
        for conv in conversations:
            for msg in conv:
                for item in msg["content"]:
                    if item.get("type") == "video":
                        raise RuntimeError(
                            "Video inputs require qwen_vl_utils; please `pip install qwen-vl-utils`"
                        )
                    if item.get("type") == "image":
                        images.append(_load_pil(item["image"]))
        return images

    # ------------------------------------------------------------------ forward
    def _model_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        param_dtype = next(self.backbone.parameters()).dtype
        out: Dict[str, Any] = {}
        for k, v in inputs.items():
            if k not in _MODEL_INPUT_KEYS:
                continue
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                v = v.to(param_dtype)      # pixel_values are fp32 and must match bf16
            out[k] = v
        return out

    def _pool(self, hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attention_mask is None:
            return hidden[:, -1] if self.pooling_mode != "mean" else hidden.mean(dim=1)
        if self.pooling_mode == "last_token":
            # Equivalent to the official `_pooling_last`; correct for left and right padding
            idx = attention_mask.long().cumsum(dim=-1).argmax(dim=-1)
            return hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
        if self.pooling_mode == "mean":
            m = attention_mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)
        return self.pooler(hidden, attention_mask)

    def encode_features(self, **inputs) -> torch.Tensor:
        attention_mask = inputs.get("attention_mask")
        out = self.backbone(**self._model_inputs(inputs), use_cache=False)
        hidden = out.last_hidden_state
        pooled = self._pool(hidden, attention_mask)
        return self.post_pool(pooled)

    # ------------------------------------------------------------------ saving
    def _export_remote_code(self, save_dir: str) -> None:
        """Copy the model definition into the checkpoint dir and write auto_map, so plain transformers can load it."""
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), _MODELING_FILE)
        dst = os.path.join(save_dir, _MODELING_FILE)
        try:
            shutil.copyfile(src, dst)
        except OSError as e:  # noqa: BLE001
            logger.warning("Failed to copy %s: %s", _MODELING_FILE, e)
            return

        cfg_path = os.path.join(save_dir, "config.json")
        if not os.path.isfile(cfg_path):
            logger.warning("%s does not exist; skipping auto_map (expected when only a LoRA adapter is saved)", cfg_path)
            return
        with open(cfg_path, "r", encoding="utf-8") as f:
            conf = json.load(f)
        conf["auto_map"] = dict(_AUTO_MAP)
        conf["architectures"] = ["Qwen3VLForEmbedding"]
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(conf, f, ensure_ascii=False, indent=2)
        logger.info("auto_map written; %s can be loaded with AutoModel.from_pretrained(..., trust_remote_code=True)", save_dir)

    def _save_backbone(self, save_dir: str) -> None:
        backbone = self.backbone
        is_peft = hasattr(backbone, "peft_config")

        if is_peft and self.merge_lora_on_save:
            # Merge into full weights; the checkpoint is a standard HF model directory (recommended)
            merged = backbone.merge_and_unload()
            merged.save_pretrained(save_dir, safe_serialization=True)
            logger.info("LoRA merged into the backbone before saving (lora.merge_on_save=true)")
        else:
            backbone.save_pretrained(save_dir)
            if is_peft:
                logger.info("Only the LoRA adapter was saved (lora.merge_on_save=false)")

        self.processor.save_pretrained(save_dir)
        if not (is_peft and not self.merge_lora_on_save):
            self._export_remote_code(save_dir)