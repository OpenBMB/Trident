#!/usr/bin/env bash
# Train Trident (or an ablation) with torchrun.
#
# Usage:
#   bash scripts/train.sh                                   # Trident-Qwen3VL-2B (default)
#   bash scripts/train.sh configs/trident_jinaclip.yaml     # Trident-JinaCLIP
#   bash scripts/train.sh configs/ablations/qwen3vl_wo_mp_infonce.yaml
#   bash scripts/train.sh configs/trident_qwen3vl.yaml --set train.output_dir=outputs/my_run
#
# Environment variables:
#   NUM_GPUS   number of GPUs on this node (default 8; the paper uses 8 x A100-80GB)
#   MASTER_PORT torchrun rendezvous port (default 29500)
#
# The effective global batch size is NUM_GPUS x per_device_train_batch_size x
# gradient_accumulation_steps (8 x 8 x 2 = 128 in the paper). If you use fewer GPUs,
# increase train.gradient_accumulation_steps accordingly to keep it unchanged.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

CONFIG="${1:-configs/trident_qwen3vl.yaml}"
[[ $# -gt 0 ]] && shift

NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"

EXTRA_ARGS=()
if [[ $# -gt 0 && "$1" == "--set" ]]; then
    EXTRA_ARGS=("$@")
elif [[ $# -gt 0 ]]; then
    EXTRA_ARGS=(--set "$@")
fi

torchrun --nproc_per_node "$NUM_GPUS" --master_port "$MASTER_PORT" \
    src/train.py \
    --config "$CONFIG" \
    "${EXTRA_ARGS[@]}"
