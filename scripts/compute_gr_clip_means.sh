#!/usr/bin/env bash
# Compute GR-CLIP mean-shift calibration vectors for a CLIP-based model
# (clip_vit_l14 / siglip2 / altclip / jina_clip_v2 / trident_jinaclip).
#
# Usage:
#   bash scripts/compute_gr_clip_means.sh <model_type> <checkpoint> [data_jsonl] [image_root]
#
# Example:
#   bash scripts/compute_gr_clip_means.sh jina_clip_v2 jinaai/jina-clip-v2
#   export GR_CLIP_MEANS=outputs/gr_clip_calib/jina_clip_v2/gr_clip_means.npz
#   bash scripts/eval.sh jina_clip_v2 jinaai/jina-clip-v2 vdoc
#
# The calibration data uses the training-data format (query + three positive views,
# see README). By default the Trident training set is used.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    sed -n '2,15p' "$0"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

MODEL_TYPE="$1"
CHECKPOINT="$2"
DATA="${3:-data/train/train.jsonl}"
IMAGE_ROOT="${4:-data/images}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
OUT_DIR="${OUT_DIR:-outputs/gr_clip_calib/${MODEL_TYPE}}"

python src/tools/compute_gr_clip_calibration_means.py \
    --data "$DATA" \
    --image_root "$IMAGE_ROOT" \
    --model_type "$MODEL_TYPE" \
    --checkpoint "$CHECKPOINT" \
    --gpus "$GPUS" \
    --batch_size 32 \
    --out_dir "$OUT_DIR"

echo "[done] export GR_CLIP_MEANS=$OUT_DIR/gr_clip_means.npz"
