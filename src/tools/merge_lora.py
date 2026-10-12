#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tools/merge_lora.py

Merge a Trident-Qwen3VL LoRA-adapter checkpoint (written by src/train.py, i.e. a directory
with mmemb_model.json + adapter_config.json) into the base model and save a standalone
checkpoint that can be used WITHOUT --base_model:

    python src/tools/merge_lora.py \
        --adapter outputs/trident_qwen3vl_2b/final \
        --output  outputs/trident_qwen3vl_2b/merged \
        --base_model Qwen/Qwen3-VL-2B-Instruct

The model is loaded exactly like embedding_backends/trident_qwen3vl.py does, then
`merge_and_unload()` is applied and the following is written to --output:

  * full backbone weights (safetensors) + config.json
  * the processor / tokenizer files (preprocessor_config.json, tokenizer*, chat template, ...)
  * mmemb_model.json (with LoRA disabled) and every extra file of the adapter checkpoint that
    `encoder.load_extra()` may need (adapter_* and trainer files are skipped)
  * merge_info.json (provenance)

By default the script verifies the result: embeddings of a few probe inputs must match
  (1) before vs. after merging (merge is numerically sound), and
  (2) in-memory merged model vs. the model reloaded from --output (saving / loading is correct).

Optionally the checkpoint can be uploaded to the Hugging Face Hub (--push_to_hub).

IMPORTANT: when evaluating the merged checkpoint do NOT pass --base_model to eval.sh. For a
non-adapter checkpoint the backend would then load the base model's weights instead of the
merged ones.
"""

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

import torch

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Same instructions as embedding_backends/trident_qwen3vl.py
DEFAULT_INSTRUCTIONS = {
    "query": "Represent the query for retrieving relevant content.",
    "doc": "Represent the candidate content for retrieval.",
}

# Files of the adapter checkpoint that must not be copied into the merged checkpoint.
SKIP_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
    "README.md",
    "optimizer.pt",
    "scheduler.pt",
    "scaler.pt",
    "trainer_state.json",
    "training_args.bin",
}
SKIP_PREFIXES = ("rng_state", "checkpoint-", "global_step", "runs")

PROBE_QUERIES = [
    "How many stores did Saint Laurent operate in Western Europe in 2020?",
    "Which year had the highest revenue according to the bar chart?",
    "What is the main topic of the presentation slide?",
]
PROBE_DOCS = [
    "A bar chart showing the number of stores of a luxury fashion brand by region and year.",
    "Annual revenue of a company, summarized in a short table with three columns.",
    "A slide with the title 'Quarterly review' and three bullet points about sales.",
]


# --------------------------------------------------------------------------- helpers
def log(message: str) -> None:
    print(f"[merge] {message}", flush=True)


def make_record(text: str, images: list, instruction: Optional[str]):
    """Build an mmemb Record (mirrors TridentQwen3VLBackend._make_record)."""
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
        object.__setattr__(record, "instruction", instruction)
    except Exception:  # noqa: BLE001
        pass
    return record


def load_encoder(
    checkpoint: Path,
    base_model: Optional[str],
    dtype: str,
    merge_lora: bool,
):
    """Load an encoder from a checkpoint the same way TridentQwen3VLBackend.load() does."""
    from mmemb.models import build_encoder

    cfg_path = checkpoint / "mmemb_model.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"{cfg_path} does not exist. The checkpoint must be a directory written by "
            f"this repository (containing mmemb_model.json)."
        )
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    is_adapter = (checkpoint / "adapter_config.json").is_file()

    cfg["lora"] = {"enable": False}
    cfg["gradient_checkpointing"] = False
    cfg["dtype"] = dtype

    if is_adapter:
        if base_model:
            cfg["pretrained_model_name_or_path"] = base_model
    else:
        # A merged checkpoint carries its own weights. --base_model is intentionally ignored.
        cfg["pretrained_model_name_or_path"] = str(checkpoint)

    if (checkpoint / "preprocessor_config.json").is_file():
        cfg["processor_name_or_path"] = str(checkpoint)

    encoder = build_encoder(cfg)

    if is_adapter:
        from peft import PeftModel

        encoder.backbone = PeftModel.from_pretrained(encoder.backbone, str(checkpoint))
        if merge_lora:
            encoder.backbone = encoder.backbone.merge_and_unload()

    encoder.load_extra(str(checkpoint))
    encoder.eval()
    return encoder, cfg


@torch.inference_mode()
def embed(encoder, device: str, probes: list) -> torch.Tensor:
    """Embed probe inputs [(role, text, images), ...] -> L2-normalised [N, D] float tensor (CPU)."""
    chunks = []
    for role in ("query", "doc"):
        items = [p for p in probes if p[0] == role]
        if not items:
            continue
        records = [make_record(t, imgs, DEFAULT_INSTRUCTIONS[role]) for _, t, imgs in items]
        inputs = encoder.build_inputs(records, role=role)
        inputs = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()
        }
        emb = encoder.encode_features(**inputs).detach().float()
        chunks.append(torch.nn.functional.normalize(emb, dim=-1).cpu())
    return torch.cat(chunks, dim=0)


def min_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Embedding shapes differ: {tuple(a.shape)} vs {tuple(b.shape)}")
    return float((a * b).sum(dim=-1).min().item())


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_probes(image_path: Optional[str]) -> list:
    probes = [("query", t, []) for t in PROBE_QUERIES] + [("doc", t, []) for t in PROBE_DOCS]
    if image_path:
        probes.append(("doc", "", [image_path]))  # image-only document
        probes.append(("doc", PROBE_DOCS[0], [image_path]))  # fused document (text + image)
    return probes


# --------------------------------------------------------------------------- saving
def save_backbone(backbone, out_dir: Path, max_shard_size: str) -> None:
    if hasattr(backbone, "save_pretrained"):
        backbone.save_pretrained(
            str(out_dir), safe_serialization=True, max_shard_size=max_shard_size
        )
        return
    # Fallback for plain nn.Modules
    from safetensors.torch import save_file

    state = {k: v.detach().cpu().contiguous() for k, v in backbone.state_dict().items()}
    save_file(state, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    config = getattr(backbone, "config", None)
    if config is not None and hasattr(config, "save_pretrained"):
        config.save_pretrained(str(out_dir))


def should_skip(name: str) -> bool:
    return name in SKIP_FILES or name.startswith(SKIP_PREFIXES)


def save_merged(
    encoder,
    adapter_dir: Path,
    out_dir: Path,
    cfg_used: dict,
    base_model: str,
    dtype: str,
    max_shard_size: str,
    model_card: Optional[Path],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    log("saving backbone weights ...")
    save_backbone(encoder.backbone, out_dir, max_shard_size)

    processor = getattr(encoder, "processor", None)
    if processor is not None and hasattr(processor, "save_pretrained"):
        log("saving processor ...")
        processor.save_pretrained(str(out_dir))
    tokenizer = getattr(encoder, "tokenizer", None)
    if tokenizer is not None and not (out_dir / "tokenizer_config.json").is_file():
        log("saving tokenizer ...")
        tokenizer.save_pretrained(str(out_dir))

    # mmemb_model.json: original config, LoRA disabled
    cfg = json.loads((adapter_dir / "mmemb_model.json").read_text(encoding="utf-8"))
    cfg["lora"] = {"enable": False}
    (out_dir / "mmemb_model.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Extra files needed by encoder.load_extra() and anything else the adapter dir contains.
    copied = []
    for item in sorted(adapter_dir.iterdir()):
        if should_skip(item.name) or item.name == "mmemb_model.json":
            continue
        target = out_dir / item.name
        if target.exists():
            continue  # already written above (config / processor files)
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
        copied.append(item.name)
    if copied:
        log(f"copied extra files from the adapter checkpoint: {', '.join(copied)}")

    if model_card is not None:
        shutil.copy2(model_card, out_dir / "README.md")
        log(f"model card copied from {model_card}")

    try:
        import peft
        import transformers

        versions = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        }
    except Exception:  # noqa: BLE001
        versions = {"torch": torch.__version__}
    info = {
        "base_model": base_model,
        "adapter_checkpoint": str(adapter_dir),
        "dtype": dtype,
        "method": "peft merge_and_unload",
        "versions": versions,
    }
    (out_dir / "merge_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def push_to_hub(folder: Path, repo_id: str, private: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload merged Trident-Qwen3VL checkpoint",
    )
    log(f"uploaded to https://huggingface.co/{repo_id}")


# --------------------------------------------------------------------------- main
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge a Trident-Qwen3VL LoRA adapter into its base model and save a standalone checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--adapter", required=True, help="LoRA-adapter checkpoint dir (e.g. <output_dir>/final).")
    p.add_argument("--output", required=True, help="Directory for the merged checkpoint.")
    p.add_argument(
        "--base_model",
        default=None,
        help="Base model (local path or HF repo id). Default: the one recorded in mmemb_model.json.",
    )
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available).")
    p.add_argument("--max_shard_size", default="5GB")
    p.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty --output.")
    p.add_argument("--model_card", default=None, help="Optional README.md to copy into --output.")
    p.add_argument("--no_verify", action="store_true", help="Skip the numerical verification.")
    p.add_argument(
        "--verify_image",
        default=None,
        help="Optional image path; adds image-only and fused probes to the verification.",
    )
    p.add_argument("--merge_tol", type=float, default=0.995,
                   help="Min cosine similarity between embeddings before / after merging.")
    p.add_argument("--reload_tol", type=float, default=0.9999,
                   help="Min cosine similarity between the in-memory merged model and the reloaded checkpoint.")
    p.add_argument("--push_to_hub", default=None, metavar="REPO_ID", help="Upload --output to this HF model repo.")
    p.add_argument("--private", action="store_true", help="Create the Hub repo as private.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    adapter_dir = Path(args.adapter).expanduser().resolve()
    out_dir = Path(args.output).expanduser().resolve()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_card = Path(args.model_card).expanduser().resolve() if args.model_card else None

    if not (adapter_dir / "adapter_config.json").is_file():
        sys.exit(
            f"[merge] ERROR: {adapter_dir} has no adapter_config.json - it is not a LoRA-adapter "
            f"checkpoint (already merged?)."
        )
    if model_card is not None and not model_card.is_file():
        sys.exit(f"[merge] ERROR: model card not found: {model_card}")
    if out_dir == adapter_dir:
        sys.exit("[merge] ERROR: --output must differ from --adapter.")
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        sys.exit(f"[merge] ERROR: {out_dir} is not empty (use --overwrite).")

    probes = build_probes(args.verify_image)

    # 1) load adapter (unmerged) -------------------------------------------------------
    log(f"loading adapter checkpoint {adapter_dir} (device={device}, dtype={args.dtype})")
    encoder, cfg_used = load_encoder(adapter_dir, args.base_model, args.dtype, merge_lora=False)
    encoder.to(device)
    base_model = cfg_used.get("pretrained_model_name_or_path", "")

    ref_unmerged = None
    if not args.no_verify:
        log("embedding probe inputs with the unmerged adapter ...")
        ref_unmerged = embed(encoder, device, probes)

    # 2) merge ---------------------------------------------------------------------------
    log("merging LoRA weights (merge_and_unload) ...")
    encoder.backbone = encoder.backbone.merge_and_unload()
    encoder.eval()

    ref_merged = None
    if not args.no_verify:
        ref_merged = embed(encoder, device, probes)
        cos = min_cosine(ref_unmerged, ref_merged)
        log(f"verify (before vs. after merge): min cosine = {cos:.6f} (tol {args.merge_tol})")
        if cos < args.merge_tol:
            sys.exit("[merge] ERROR: merged model differs from the adapter model; nothing was saved.")

    # 3) save ---------------------------------------------------------------------------
    save_merged(
        encoder, adapter_dir, out_dir, cfg_used, base_model, args.dtype, args.max_shard_size, model_card
    )
    del encoder
    free_memory()

    # 4) reload from disk exactly like the evaluation backend and compare -----------------
    if not args.no_verify:
        log(f"reloading {out_dir} as a standalone checkpoint ...")
        reloaded, _ = load_encoder(out_dir, None, args.dtype, merge_lora=False)
        reloaded.to(device)
        emb = embed(reloaded, device, probes)
        cos = min_cosine(ref_merged, emb)
        log(f"verify (in-memory vs. reloaded): min cosine = {cos:.6f} (tol {args.reload_tol})")
        del reloaded
        free_memory()
        if cos < args.reload_tol:
            sys.exit(
                "[merge] ERROR: the reloaded checkpoint does not reproduce the merged model. "
                "Do not release it; check the saved files."
            )

    log(f"done. Merged checkpoint: {out_dir}")
    log(
        "evaluate it WITHOUT --base_model, e.g.\n"
        f"  bash scripts/eval.sh trident_qwen3vl {out_dir} vdoc"
    )

    if args.push_to_hub:
        push_to_hub(out_dir, args.push_to_hub, args.private)


if __name__ == "__main__":
    main()
