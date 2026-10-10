#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
embedding_backends/trident_qwen3vl.py

Backend for Trident-Qwen3VL checkpoints trained with this repository (src/train.py).
"""

import json
import os
from typing import Optional

import torch

from .base import EmbeddingBackend
from .common import l2_normalize


class TridentQwen3VLBackend(EmbeddingBackend):
    supports_fused_text_image = True
    # Supports --matryoshka_dims: several truncation dims from one forward (see base.py).
    supports_matryoshka = True
    max_images_per_record = 64

    DEFAULT_INSTRUCTIONS = {
        "query": "Represent the query for retrieving relevant content.",
        "doc": "Represent the candidate content for retrieval.",
    }

    def default_instruction(self, role: str) -> str:
        return self.DEFAULT_INSTRUCTIONS[role]

    def load(self) -> None:
        from mmemb.models import build_encoder

        checkpoint = self.args_dict["checkpoint"]
        base_model = self.args_dict.get("base_model")
        dtype = self.args_dict.get("dtype", "bfloat16")
        max_pixels = self.args_dict.get("max_pixels")
        merge_lora = self.args_dict.get("merge_lora", True)
        self.dim = self.args_dict.get("dim")

        config_path = os.path.join(checkpoint, "mmemb_model.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"{config_path} does not exist. The checkpoint must be a directory "
                f"written by save_pretrained of this repository (containing mmemb_model.json)"
            )

        with open(config_path, "r", encoding="utf-8") as file:
            cfg = json.load(file)

        model_type = cfg.get("type")
        if model_type and model_type != "qwen3_vl_embedding":
            print(
                f"[warn] mmemb_model.json of the checkpoint has type={model_type!r}, "
                f"not qwen3_vl_embedding; please check the checkpoint path.",
                flush=True,
            )

        is_adapter = os.path.isfile(
            os.path.join(checkpoint, "adapter_config.json")
        )

        cfg["lora"] = {"enable": False}
        cfg["gradient_checkpointing"] = False
        cfg["dtype"] = dtype

        if is_adapter:
            if base_model:
                cfg["pretrained_model_name_or_path"] = base_model
            base_path = cfg["pretrained_model_name_or_path"]
            # Hugging Face repo ids (e.g. Qwen/Qwen3-VL-2B-Instruct) are allowed; only
            # explicit local paths are checked for existence.
            is_local = os.path.isabs(base_path) or base_path.startswith((".", "~"))
            if is_local and not os.path.exists(base_path):
                raise FileNotFoundError(
                    f"The checkpoint is a LoRA adapter and needs base weights, "
                    f"but {base_path} does not exist. Specify it with --base_model (local path or HF repo id)"
                )
        else:
            cfg["pretrained_model_name_or_path"] = base_model or checkpoint

        if os.path.isfile(os.path.join(checkpoint, "preprocessor_config.json")):
            cfg["processor_name_or_path"] = checkpoint

        if max_pixels is not None:
            image_cfg = dict(cfg.get("image") or {})
            image_cfg["max_pixels"] = int(max_pixels)
            cfg["image"] = image_cfg

        print(
            f"[worker] loading encoder: checkpoint={checkpoint}, "
            f"adapter={is_adapter}, dtype={dtype}",
            flush=True,
        )

        self.encoder = build_encoder(cfg)

        if is_adapter:
            from peft import PeftModel

            self.encoder.backbone = PeftModel.from_pretrained(
                self.encoder.backbone,
                checkpoint,
            )
            if merge_lora:
                self.encoder.backbone = self.encoder.backbone.merge_and_unload()
                print("[worker] LoRA merged into the base model", flush=True)

        self.encoder.load_extra(checkpoint)
        self.encoder.eval()
        self.encoder.to(self.device)

        self.tokenizer = self.encoder.tokenizer
        self.embedding_dim = self.encoder.embedding_dim
        self.image_token_ids = self._collect_image_token_ids()
        self.setup_matryoshka(self.embedding_dim)

        print(
            f"[worker] encoder ready, embedding_dim={self.embedding_dim}",
            flush=True,
        )

    def _collect_image_token_ids(self) -> set[int]:
        ids: set[int] = set()
        for token in ("<|image_pad|>", "<|video_pad|>"):
            try:
                token_id = self.tokenizer.convert_tokens_to_ids(token)
            except Exception:
                continue
            if isinstance(token_id, int) and token_id >= 0:
                ids.add(token_id)

        config = getattr(self.encoder.backbone, "config", None)
        for attr in ("image_token_id", "video_token_id"):
            value = getattr(config, attr, None)
            if isinstance(value, int) and value >= 0:
                ids.add(value)

        return ids

    @staticmethod
    def _make_record(
        text: Optional[str],
        images: list[str],
        instruction: Optional[str] = None,
    ):
        from mmemb.data.schema import Record

        try:
            return Record(text=text or "", images=images, instruction=instruction)
        except TypeError:
            pass

        for key in ("image", "image_paths", "imgs"):
            try:
                record = Record(text=text or "", **{key: images})
                break
            except TypeError:
                continue
        else:
            record = Record(text=text or "")
            try:
                object.__setattr__(record, "images", images)
            except Exception:
                pass

        try:
            object.__setattr__(record, "instruction", instruction)
        except Exception:
            pass

        return record

    def _count_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> list[tuple[int, int, int]]:
        out: list[tuple[int, int, int]] = []

        for row_idx in range(input_ids.size(0)):
            row = input_ids[row_idx]
            if attention_mask is not None:
                row = row[attention_mask[row_idx].bool()]

            total = int(row.numel())

            if self.image_token_ids:
                mask = torch.zeros_like(row, dtype=torch.bool)
                for token_id in self.image_token_ids:
                    mask |= row == token_id
                image_tokens = int(mask.sum().item())
            else:
                image_tokens = 0

            out.append((total, total - image_tokens, image_tokens))

        return out

    def render_prompt(self, text, images, instruction, role) -> str:
        record = self._make_record(text, images, instruction)

        format_conv = getattr(self.encoder, "_format_conversation", None)
        if format_conv is not None:
            try:
                conversation = format_conv(record, role)
                rendered = self.encoder.processor.apply_chat_template(
                    [conversation],
                    add_generation_prompt=getattr(
                        self.encoder, "add_generation_prompt", False
                    ),
                    tokenize=False,
                )
                if isinstance(rendered, list):
                    rendered = rendered[0]
                return rendered
            except Exception as exc:  # noqa: BLE001
                return f"<render_prompt via _format_conversation failed: {exc}>"

        compose = getattr(self.encoder, "_compose_text", None)
        if compose is not None:
            return compose(record, role)

        return record.text or ""

    def build_inputs_cpu(self, items: list[dict]):
        records = [
            self._make_record(
                item.get("text"),
                list(item.get("images") or []),
                item.get("instruction"),
            )
            for item in items
        ]
        # In multi-job mode the role is updated in place by worker_main for each job,
        # so this always reads the role of the job currently being processed.
        role = self.args_dict["role"]
        return self.encoder.build_inputs(records, role=role)

    @torch.inference_mode()
    def compute_from_inputs(self, prepared, items, role):
        model_inputs = {
            key: (
                value.to(self.device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in prepared.items()
        }

        token_infos = self._count_tokens(
            model_inputs["input_ids"],
            model_inputs.get("attention_mask"),
        )

        embeddings = self.encoder.encode_features(**model_inputs)
        embeddings = embeddings.detach().float()

        if self.matryoshka_pack_dims:
            # One forward, all Matryoshka dims packed; the driver unpacks them when merging.
            array = self.matryoshka_pack(embeddings.cpu().numpy())
        else:
            if self.dim and self.dim < embeddings.size(-1):
                embeddings = embeddings[:, : self.dim]
            array = l2_normalize(embeddings.cpu().numpy())

        if array.shape[0] != len(items):
            raise ValueError(
                f"Model returned {array.shape[0]} embeddings "
                f"for {len(items)} inputs."
            )

        return array, token_infos
