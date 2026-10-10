#!/usr/bin/env bash
# Evaluate a retriever under the Origin and Mix settings (Table 1 / Table 4).
#
# Usage:
#   bash scripts/eval.sh <model_type> <checkpoint> [vdoc|natural] [extra args for run_all_datasets.sh]
#
# Examples:
#   # Trident (checkpoints trained with this repository)
#   bash scripts/eval.sh trident_qwen3vl  outputs/trident_qwen3vl_2b/final vdoc \
#        --base_model Qwen/Qwen3-VL-2B-Instruct
#   bash scripts/eval.sh trident_jinaclip outputs/trident_jinaclip/final vdoc
#
#   # Baselines
#   bash scripts/eval.sh jina_clip_v2     jinaai/jina-clip-v2                     vdoc
#   bash scripts/eval.sh siglip2          google/siglip2-large-patch16-384        vdoc
#   bash scripts/eval.sh clip_vit_l14     openai/clip-vit-large-patch14           vdoc
#   bash scripts/eval.sh altclip          BAAI/AltCLIP                            vdoc
#   bash scripts/eval.sh qwen3vl_official Qwen/Qwen3-VL-Embedding-2B              vdoc
#   bash scripts/eval.sh jina_v5_omni     <jina-embeddings-v5-omni-small path>    vdoc
#   bash scripts/eval.sh unime_phi35v     <UniME-Phi3.5-V-4.2B path>              vdoc
#   bash scripts/eval.sh visrag_ret       openbmb/VisRAG-Ret                      vdoc
#
#   # Matryoshka evaluation (Appendix I)
#   bash scripts/eval.sh trident_qwen3vl outputs/trident_qwen3vl_2b_matryoshka/final vdoc \
#        --matryoshka_dims 128,256,512,1024,2048
#
# GR-CLIP baseline ("+ GR-CLIP" rows, CLIP-based models only):
#   1. compute the calibration means once per model (see scripts/compute_gr_clip_means.sh);
#   2. export GR_CLIP_MEANS=<out_dir>/gr_clip_means.npz and run this script again.
#   export GR_CLIP_DISABLE=1 disables calibration even if GR_CLIP_MEANS is set.
#
# Results are written to outputs/eval/<model_type>/ (see scripts/run_all_datasets.sh).

set -euo pipefail

if [[ $# -lt 2 ]]; then
    sed -n '2,32p' "$0"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_TYPE="$1"
CHECKPOINT="$2"
BENCHMARK="${3:-vdoc}"
shift 2
[[ $# -gt 0 ]] && shift

RUN_NAME="${RUN_NAME:-${MODEL_TYPE}_${BENCHMARK}}"
if [[ -n "${GR_CLIP_MEANS:-}" && -z "${GR_CLIP_DISABLE:-}" ]]; then
    RUN_NAME="${RUN_NAME}_grclip"
fi

bash "$SCRIPT_DIR/run_all_datasets.sh" \
    --model_type "$MODEL_TYPE" \
    --checkpoint "$CHECKPOINT" \
    --benchmark "$BENCHMARK" \
    --embed_root "outputs/embedding/${RUN_NAME}" \
    --eval_root "outputs/eval/${RUN_NAME}" \
    "$@"
