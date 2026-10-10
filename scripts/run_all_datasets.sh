#!/usr/bin/env bash
# run_all_datasets.sh
#
# Loads the model once and runs:
#   1) query / origin_corpus / mix_corpus embeddings for every dataset;
#   2) per-dataset evaluation on the original single-modality corpus ("Origin",
#      corpus = origin_corpus.jsonl);
#   3) mixed-modality evaluation ("Mix"): the mix_corpus of all selected datasets
#      is merged into one large corpus, and every dataset is evaluated on it.
#
# Usage
# -----
#   bash scripts/run_all_datasets.sh --model_type trident_qwen3vl --checkpoint <ckpt_dir>
#   bash scripts/run_all_datasets.sh --gpus 0,1,2,3
#   bash scripts/run_all_datasets.sh --datasets ChartQA,SlideVQA
#   bash scripts/run_all_datasets.sh --benchmark natural   # Google_WIT,MSCOCO,VisualNews,OVEN
#   bash scripts/run_all_datasets.sh --skip_existing
#   bash scripts/run_all_datasets.sh --no_mix          # Origin evaluation only
#   bash scripts/run_all_datasets.sh --only_mix        # Mix evaluation only
#
#   # Matryoshka evaluation (trident_jinaclip / trident_qwen3vl / qwen3vl_official only):
#   # one forward produces all dims; every dim is evaluated (Origin + Mix) and a
#   # summary table is written. Dims larger than the model's full dim (e.g. 2048 for
#   # jina-clip) are ignored automatically.
#   bash scripts/run_all_datasets.sh --matryoshka_dims 128,256,512,1024,2048
#
#   # LoRA checkpoints that were saved as adapters need their base model:
#   bash scripts/run_all_datasets.sh --base_model Qwen/Qwen3-VL-2B-Instruct
#
# All settings in the "Configuration" block can also be overridden with
# environment variables of the same name, e.g.
#   MODEL_TYPE=jina_clip_v2 CHECKPOINT=jinaai/jina-clip-v2 bash scripts/run_all_datasets.sh
#
# Directory layout:
#   data/<DATASET>/queries.jsonl
#   data/<DATASET>/origin_corpus.jsonl
#   data/<DATASET>/mix_corpus.jsonl
#   data/<DATASET>/qrels.jsonl
#   -> <EMBED_ROOT>/<TAG>_queries/
#   -> <EMBED_ROOT>/<TAG>_origin_corpus/
#   -> <EMBED_ROOT>/<TAG>_mix_corpus/
#   -> <EVAL_ROOT>/<TAG>_result.json          Origin (per dataset)
#   -> <EVAL_ROOT>/mix_result.json            Mix (merged corpus)
#   With --matryoshka_dims additionally:
#   -> <EMBED_ROOT>/<TAG>_*/matryoshka/dim_<d>/
#   -> <EVAL_ROOT>/matryoshka/dim_<d>/<TAG>_result.json
#   -> <EVAL_ROOT>/matryoshka/dim_<d>/mix_result.json
#   -> <EVAL_ROOT>/matryoshka/matryoshka_summary.json
#   After all evaluations, a spreadsheet is exported (see TABLE_* below):
#   -> <EVAL_ROOT>/results_table.xlsx
#      one row (full dim) without Matryoshka; one row per dim with Matryoshka.

set -uo pipefail

# ============================================================================
# Project root
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

# ============================================================================
# Configuration (override with CLI flags or environment variables)
# ============================================================================

# Backend name, see src/embedding_backends/__init__.py:
#   trident_qwen3vl / trident_jinaclip / qwen3vl_official / jina_clip_v2 /
#   jina_v5_omni / unime_phi35v / visrag_ret / clip_vit_l14 / siglip2 / altclip
MODEL_TYPE="${MODEL_TYPE:-trident_qwen3vl}"
# Local directory or Hugging Face repo id of the checkpoint.
CHECKPOINT="${CHECKPOINT:-checkpoints/Trident-Qwen3VL-2B}"
# Base model for LoRA-adapter checkpoints (trident_qwen3vl / trident_jinaclip when
# the adapter was not merged). Leave empty to use the base model recorded in the
# checkpoint. Local directory or Hugging Face repo id, e.g.
#   BASE_MODEL="Qwen/Qwen3-VL-2B-Instruct"
BASE_MODEL="${BASE_MODEL:-}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DTYPE="${DTYPE:-bfloat16}"
SAVE_DTYPE="${SAVE_DTYPE:-float16}"
PREVIEW="${PREVIEW:-5}"
K_VALUES="${K_VALUES:-1,5,10}"
DEVICE="${DEVICE:-cuda}"

# Matryoshka dims, comma separated; empty = disabled. Supported by
# trident_jinaclip / trident_qwen3vl / qwen3vl_official, e.g.
#   MATRYOSHKA_DIMS="128,256,512,1024,2048"
MATRYOSHKA_DIMS="${MATRYOSHKA_DIMS:-}"
SUMMARY_SCRIPT="src/eval/summarize_matryoshka.py"

# Export an xlsx table after evaluation.
#   MODEL_NAME   model name shown in the table (defaults to MODEL_TYPE; or --model_name)
#   TABLE_METRIC metric in the table (x100, two decimals); K_VALUES must contain its k
MODEL_NAME="${MODEL_NAME:-}"
TABLE_METRIC="${TABLE_METRIC:-ndcg@10}"
TABLE_SCRIPT="src/eval/export_results_table.py"

# Parallel image loading threads + CPU prefetch depth
# (see src/eval/embed_jsonl_unified_multigpu.py).
IMAGE_LOAD_WORKERS="${IMAGE_LOAD_WORKERS:-32}"
PREFETCH_DEPTH="${PREFETCH_DEPTH:-3}"

# The mix corpus is large; rows moved to the GPU per chunk (reduce if OOM).
CORPUS_CHUNK_SIZE="${CORPUS_CHUNK_SIZE:-20000}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-1024}"

EMBED_SCRIPT="src/eval/embed_jsonl_unified_multigpu.py"
EVAL_SCRIPT="src/eval/eval_retrieval.py"
MIX_EVAL_SCRIPT="src/eval/eval_mix_retrieval.py"
MANIFEST_SCRIPT="src/eval/build_manifests.py"

DATA_ROOT="${DATA_ROOT:-data}"
EMBED_ROOT="${EMBED_ROOT:-outputs/embedding/${MODEL_TYPE}}"
EVAL_ROOT="${EVAL_ROOT:-outputs/eval/${MODEL_TYPE}}"

# Corpus file names inside each dataset directory.
QUERIES_NAME="queries.jsonl"
ORIGIN_CORPUS_NAME="origin_corpus.jsonl"
MIX_CORPUS_NAME="mix_corpus.jsonl"
QRELS_NAME="qrels.jsonl"

# Dataset directory name -> short tag used for output directories.
# Note: the tag is also the id namespace in the Mix evaluation and must be unique.
declare -A DATASET_TAGS=(
    ["ChartQA"]="cqa"
    ["DocVQA"]="dqa"
    ["InfoVQA"]="iqa"
    ["SlideVQA"]="sqa"
    ["ViDoSeek"]="vdsk"
    ["Dude"]="dude"
    ["Google_WIT"]="gwit"
    ["MSCOCO"]="mscoco"
    ["VisualNews"]="vnew"
    ["OVEN"]="oven"
)

# Benchmarks of the paper:
#   vdoc    : Table 1 (visual document retrieval)
#   natural : Table 4 (natural-image retrieval, Appendix F)
VDOC_DATASETS=(ChartQA DocVQA InfoVQA SlideVQA ViDoSeek Dude)
NATURAL_DATASETS=(Google_WIT MSCOCO VisualNews OVEN)

if [[ -n "${DATASET_LIST:-}" ]]; then
    IFS=',' read -r -a DATASETS <<< "$DATASET_LIST"
else
    DATASETS=("${VDOC_DATASETS[@]}")
fi

# ============================================================================
# Command-line arguments
# ============================================================================

SKIP_EXISTING=0
RUN_SINGLE=1
RUN_MIX=1
DROP_MISSING_GOLD=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_type)
            MODEL_TYPE="$2"; shift 2 ;;
        --checkpoint)
            CHECKPOINT="$2"; shift 2 ;;
        --base_model)
            BASE_MODEL="$2"; shift 2 ;;
        --gpus)
            GPUS="$2"; shift 2 ;;
        --datasets)
            IFS=',' read -r -a DATASETS <<< "$2"; shift 2 ;;
        --benchmark)
            case "$2" in
                vdoc)    DATASETS=("${VDOC_DATASETS[@]}") ;;
                natural) DATASETS=("${NATURAL_DATASETS[@]}") ;;
                *) echo "[error] --benchmark must be vdoc or natural, got: $2"; exit 1 ;;
            esac
            shift 2 ;;
        --data_root)
            DATA_ROOT="$2"; shift 2 ;;
        --embed_root)
            EMBED_ROOT="$2"; shift 2 ;;
        --eval_root)
            EVAL_ROOT="$2"; shift 2 ;;
        --batch_size)
            BATCH_SIZE="$2"; shift 2 ;;
        --k_values)
            K_VALUES="$2"; shift 2 ;;
        --skip_existing)
            SKIP_EXISTING=1; shift ;;
        --image_load_workers)
            IMAGE_LOAD_WORKERS="$2"; shift 2 ;;
        --prefetch_depth)
            PREFETCH_DEPTH="$2"; shift 2 ;;
        --no_mix)
            RUN_MIX=0; shift ;;
        --only_mix)
            RUN_SINGLE=0; shift ;;
        --corpus_chunk_size)
            CORPUS_CHUNK_SIZE="$2"; shift 2 ;;
        --drop_missing_gold)
            DROP_MISSING_GOLD=1; shift ;;
        --matryoshka_dims)
            MATRYOSHKA_DIMS="$2"; shift 2 ;;
        --model_name)
            MODEL_NAME="$2"; shift 2 ;;
        *)
            echo "[warn] unknown argument: $1 (ignored)"; shift ;;
    esac
done

if [[ $RUN_SINGLE -eq 0 && $RUN_MIX -eq 0 ]]; then
    echo "[error] --no_mix and --only_mix cannot be used together"
    exit 1
fi

# BASE_MODEL may be a local directory or a HF / ModelScope repo id. Existence is
# only checked for obvious local paths (starting with /, ./ or ../), since repo
# ids also contain "/".
if [[ -n "$BASE_MODEL" && "$BASE_MODEL" =~ ^(\.{0,2}/) ]]; then
    if [[ ! -e "$BASE_MODEL" ]]; then
        echo "[error] local path given by --base_model does not exist: $BASE_MODEL"
        exit 1
    fi
    echo "[init] using local base_model: $BASE_MODEL"
fi

# Accept "[128, 256, 512]" / "128 256 512" / "128,256,512" and normalize to an array.
MATRYOSHKA_DIM_LIST=()
if [[ -n "$MATRYOSHKA_DIMS" ]]; then
    read -r -a MATRYOSHKA_DIM_LIST <<< "$(echo "$MATRYOSHKA_DIMS" | tr -d '[]' | tr ',' ' ')"
    for D in "${MATRYOSHKA_DIM_LIST[@]}"; do
        if ! [[ "$D" =~ ^[0-9]+$ ]] || [[ "$D" -le 0 ]]; then
            echo "[error] invalid value in --matryoshka_dims: $D"
            exit 1
        fi
    done
    MATRYOSHKA_DIMS="$(IFS=','; echo "${MATRYOSHKA_DIM_LIST[*]}")"
    echo "[init] matryoshka dims: $MATRYOSHKA_DIMS"
fi

dataset_tag() {
    local name="$1"
    if [[ -n "${DATASET_TAGS[$name]+x}" ]]; then
        echo "${DATASET_TAGS[$name]}"
    else
        echo "${name,,}"
    fi
}

mkdir -p "$EMBED_ROOT" "$EVAL_ROOT"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
JOBS_JSON="$EVAL_ROOT/jobs_manifest_${RUN_TS}.json"
EVAL_MANIFEST="$EVAL_ROOT/eval_manifest_${RUN_TS}.json"
SUMMARY_LOG="$EVAL_ROOT/run_all_${RUN_TS}.log"

echo "[init] model_type: $MODEL_TYPE"
echo "[init] checkpoint: $CHECKPOINT"
echo "[init] jobs manifest: $JOBS_JSON"
echo "[init] eval manifest: $EVAL_MANIFEST"
echo "[init] summary log: $SUMMARY_LOG"

# ============================================================================
# Step 1: build the jobs manifest (queries / origin_corpus / mix_corpus) and the
#         eval manifest. Datasets with missing files are skipped.
# ============================================================================

FAILED_DATASETS=()

DATASETS_CSV="$(IFS=','; echo "${DATASETS[*]}")"

TAGS_CSV=""
for DATASET in "${DATASETS[@]}"; do
    TAG=$(dataset_tag "$DATASET")
    if [[ -n "$TAGS_CSV" ]]; then TAGS_CSV+=","; fi
    TAGS_CSV+="${DATASET}=${TAG}"
done

MANIFEST_EXTRA_ARGS=()
[[ $RUN_MIX -eq 0 ]] && MANIFEST_EXTRA_ARGS+=(--no_mix)
[[ $RUN_SINGLE -eq 0 ]] && MANIFEST_EXTRA_ARGS+=(--only_mix)

DATASETS_OK_FILE="$EVAL_ROOT/datasets_ok_${RUN_TS}.txt"

python "$MANIFEST_SCRIPT" \
        --data_root "$DATA_ROOT" \
        --embed_root "$EMBED_ROOT" \
        --datasets "$DATASETS_CSV" \
        --tags "$TAGS_CSV" \
        --jobs_json "$JOBS_JSON" \
        --eval_manifest "$EVAL_MANIFEST" \
        --queries_name "$QUERIES_NAME" \
        --origin_corpus_name "$ORIGIN_CORPUS_NAME" \
        --mix_corpus_name "$MIX_CORPUS_NAME" \
        --qrels_name "$QRELS_NAME" \
        "${MANIFEST_EXTRA_ARGS[@]}" > "$DATASETS_OK_FILE"
MANIFEST_STATUS=$?

mapfile -t EVAL_DATASETS < "$DATASETS_OK_FILE"

if [[ $MANIFEST_STATUS -ne 0 || ${#EVAL_DATASETS[@]} -eq 0 ]]; then
    echo "[error] manifest generation failed or no dataset is available; exiting"
    exit 1
fi

# Record datasets skipped by build_manifests.py
for DATASET in "${DATASETS[@]}"; do
    FOUND=0
    for OK in "${EVAL_DATASETS[@]}"; do
        [[ "$OK" == "$DATASET" ]] && FOUND=1 && break
    done
    [[ $FOUND -eq 0 ]] && FAILED_DATASETS+=("$DATASET (missing input files)")
done

echo "[init] datasets to evaluate: ${EVAL_DATASETS[*]}"

# ============================================================================
# Step 2: run the embedding script once; the model is loaded once for all jobs.
# ============================================================================

EMBED_EXTRA_ARGS=()
[[ "$SKIP_EXISTING" -eq 1 ]] && EMBED_EXTRA_ARGS+=(--skip_existing)
[[ -n "$BASE_MODEL" ]] && EMBED_EXTRA_ARGS+=(--base_model "$BASE_MODEL")
[[ -n "$MATRYOSHKA_DIMS" ]] && EMBED_EXTRA_ARGS+=(--matryoshka_dims "$MATRYOSHKA_DIMS")

echo ""
echo "=============================================================="
echo "[run] embedding (queries / origin_corpus / mix_corpus of all datasets, single model load)"
echo "=============================================================="

TOTAL_START=$(date +%s)

python "$EMBED_SCRIPT" \
    --jobs_json "$JOBS_JSON" \
    --model_type "$MODEL_TYPE" \
    --checkpoint "$CHECKPOINT" \
    --gpus "$GPUS" \
    --batch_size "$BATCH_SIZE" \
    --dtype "$DTYPE" \
    --save_dtype "$SAVE_DTYPE" \
    --preview "$PREVIEW" \
    --image_load_workers "$IMAGE_LOAD_WORKERS" \
    --prefetch_depth "$PREFETCH_DEPTH" \
    "${EMBED_EXTRA_ARGS[@]}"

EMBED_STATUS=$?

if [[ $EMBED_STATUS -ne 0 ]]; then
    echo "[error] embedding script failed (exit=$EMBED_STATUS); evaluations below are skipped where embeddings.npy is missing"
fi

EVAL_COMMON_ARGS=(--k_values "$K_VALUES" --device "$DEVICE"
                  --corpus_chunk_size "$CORPUS_CHUNK_SIZE"
                  --query_batch_size "$QUERY_BATCH_SIZE")
[[ $DROP_MISSING_GOLD -eq 1 ]] && EVAL_COMMON_ARGS+=(--drop_missing_gold)

# ============================================================================
# Steps 3 / 4 are wrapped in functions so the same logic runs both for the full
# dimension (SUBDIR empty, results in $EVAL_ROOT) and for each Matryoshka dim
# (SUBDIR=matryoshka/dim_<d>, results in $EVAL_ROOT/matryoshka/dim_<d>).
#
#   $1 SUBDIR   subdirectory under each embedding directory ("" = top level)
#   $2 OUT_DIR  output directory of the evaluation results
#   $3 LABEL    label used in logs
# ============================================================================

with_subdir() {
    if [[ -n "$2" ]]; then echo "$1/$2"; else echo "$1"; fi
}

# Step 3: per-dataset evaluation (corpus = origin_corpus)
run_single_evals() {
    local SUBDIR="$1" OUT_DIR="$2" LABEL="$3"
    mkdir -p "$OUT_DIR"

    for DATASET in "${EVAL_DATASETS[@]}"; do
        TAG=$(dataset_tag "$DATASET")
        QUERIES_OUT=$(with_subdir "$EMBED_ROOT/${TAG}_queries" "$SUBDIR")
        CORPUS_OUT=$(with_subdir "$EMBED_ROOT/${TAG}_origin_corpus" "$SUBDIR")
        QRELS="$DATA_ROOT/$DATASET/$QRELS_NAME"
        EVAL_OUT="$OUT_DIR/${TAG}_result.json"

        echo ""
        echo "=============================================================="
        echo "[dataset] $DATASET (tag=$TAG) - origin corpus [$LABEL]"
        echo "=============================================================="

        if [[ ! -f "$QUERIES_OUT/embeddings.npy" || ! -f "$CORPUS_OUT/embeddings.npy" ]]; then
            echo "[error] embeddings.npy missing for $DATASET; skipping origin evaluation [$LABEL]"
            FAILED_DATASETS+=("$DATASET (missing embeddings for origin eval, $LABEL)")
            continue
        fi

        DATASET_START=$(date +%s)

        python "$EVAL_SCRIPT" \
            --queries_dir "$QUERIES_OUT" \
            --corpus_dir "$CORPUS_OUT" \
            --qrels "$QRELS" \
            --dataset_name "$DATASET" \
            --output_json "$EVAL_OUT" \
            "${EVAL_COMMON_ARGS[@]}"
        EVAL_RC=$?

        DATASET_END=$(date +%s)
        ELAPSED=$((DATASET_END - DATASET_START))

        if [[ $EVAL_RC -ne 0 ]]; then
            echo "[error] origin evaluation of $DATASET failed (${ELAPSED}s) [$LABEL]"
            FAILED_DATASETS+=("$DATASET (origin eval failed, $LABEL)")
        else
            echo "[dataset] $DATASET done in ${ELAPSED}s, results: $EVAL_OUT"
        fi

        echo "$DATASET [$LABEL]: origin_eval_elapsed=${ELAPSED}s eval_json=${EVAL_OUT}" >> "$SUMMARY_LOG"
    done
}

# Step 4: Mix evaluation (mix_corpus of all datasets merged into one corpus)
run_mix_eval() {
    local SUBDIR="$1" OUT_DIR="$2" LABEL="$3"
    local MIX_READY_DATASETS=()
    mkdir -p "$OUT_DIR"

    for DATASET in "${EVAL_DATASETS[@]}"; do
        TAG=$(dataset_tag "$DATASET")
        QUERIES_OUT=$(with_subdir "$EMBED_ROOT/${TAG}_queries" "$SUBDIR")
        MIX_OUT=$(with_subdir "$EMBED_ROOT/${TAG}_mix_corpus" "$SUBDIR")

        if [[ -f "$QUERIES_OUT/embeddings.npy" && -f "$MIX_OUT/embeddings.npy" ]]; then
            MIX_READY_DATASETS+=("$DATASET")
        else
            echo "[warn] queries / mix_corpus embeddings missing for $DATASET; excluded from Mix evaluation [$LABEL]"
            FAILED_DATASETS+=("$DATASET (missing embeddings for mix eval, $LABEL)")
        fi
    done

    if [[ ${#MIX_READY_DATASETS[@]} -eq 0 ]]; then
        echo "[error] no dataset available for Mix evaluation; skipping [$LABEL]"
        FAILED_DATASETS+=("mix (no dataset ready, $LABEL)")
        return
    fi

    MIX_DATASETS_CSV="$(IFS=','; echo "${MIX_READY_DATASETS[*]}")"
    MIX_EVAL_OUT="$OUT_DIR/mix_result.json"
    MIX_RUN_DIR="$OUT_DIR/mix_runs"

    echo ""
    echo "=============================================================="
    echo "[mix] merged-corpus evaluation over ${#MIX_READY_DATASETS[@]} datasets: $MIX_DATASETS_CSV [$LABEL]"
    echo "=============================================================="

    MIX_START=$(date +%s)

    python "$MIX_EVAL_SCRIPT" \
        --manifest "$EVAL_MANIFEST" \
        --datasets "$MIX_DATASETS_CSV" \
        --embedding_subdir "$SUBDIR" \
        --output_json "$MIX_EVAL_OUT" \
        --dump_run_dir "$MIX_RUN_DIR" \
        "${EVAL_COMMON_ARGS[@]}"
    MIX_RC=$?

    MIX_END=$(date +%s)
    MIX_ELAPSED=$((MIX_END - MIX_START))

    if [[ $MIX_RC -ne 0 ]]; then
        echo "[error] Mix evaluation failed (${MIX_ELAPSED}s) [$LABEL]"
        FAILED_DATASETS+=("mix (eval failed, $LABEL)")
    else
        echo "[mix] done in ${MIX_ELAPSED}s, results: $MIX_EVAL_OUT"
    fi

    echo "MIX [$LABEL]: eval_elapsed=${MIX_ELAPSED}s eval_json=${MIX_EVAL_OUT}" >> "$SUMMARY_LOG"
}

run_evals() {
    local SUBDIR="$1" OUT_DIR="$2" LABEL="$3"
    [[ $RUN_SINGLE -eq 1 ]] && run_single_evals "$SUBDIR" "$OUT_DIR" "$LABEL"
    [[ $RUN_MIX -eq 1 ]] && run_mix_eval "$SUBDIR" "$OUT_DIR" "$LABEL"
}

# Whether a Matryoshka dim was actually produced (dims larger than the model's
# full dim are skipped by the embedding script; this is not a failure).
matryoshka_dim_produced() {
    local D="$1"
    for DATASET in "${EVAL_DATASETS[@]}"; do
        TAG=$(dataset_tag "$DATASET")
        [[ -f "$EMBED_ROOT/${TAG}_queries/matryoshka/dim_${D}/embeddings.npy" ]] && return 0
    done
    return 1
}

# Full dimension
run_evals "" "$EVAL_ROOT" "full"

# Matryoshka: evaluate every dim
MATRYOSHKA_DONE=()
if [[ ${#MATRYOSHKA_DIM_LIST[@]} -gt 0 ]]; then
    for D in "${MATRYOSHKA_DIM_LIST[@]}"; do
        if ! matryoshka_dim_produced "$D"; then
            echo "[matryoshka] dim=$D produced no embeddings (probably larger than the model's full dim); skipping"
            continue
        fi
        echo ""
        echo "##############################################################"
        echo "[matryoshka] dim=$D"
        echo "##############################################################"
        run_evals "matryoshka/dim_${D}" "$EVAL_ROOT/matryoshka/dim_${D}" "dim=${D}"
        MATRYOSHKA_DONE+=("$D")
    done

    if [[ ${#MATRYOSHKA_DONE[@]} -gt 0 ]]; then
        python "$SUMMARY_SCRIPT" \
            --eval_root "$EVAL_ROOT" \
            --dims "$(IFS=','; echo "${MATRYOSHKA_DONE[*]}")" \
            || echo "[warn] Matryoshka summary failed"
    else
        echo "[error] no Matryoshka dim was produced"
        FAILED_DATASETS+=("matryoshka (no dim produced)")
    fi
fi

# ============================================================================
# Export a spreadsheet (xlsx) from all result files above.
#   - without Matryoshka: one row (full dimension);
#   - with Matryoshka: one row per produced dim.
# Failures only warn and do not affect the saved results.
# ============================================================================

TABLE_ARGS=(--eval_root "$EVAL_ROOT" --embed_root "$EMBED_ROOT"
            --model_name "${MODEL_NAME:-$MODEL_TYPE}"
            --metric "$TABLE_METRIC"
            --datasets "$(IFS=','; echo "${EVAL_DATASETS[*]}")")

if [[ ${#MATRYOSHKA_DIM_LIST[@]} -gt 0 ]]; then
    if [[ ${#MATRYOSHKA_DONE[@]} -gt 0 ]]; then
        TABLE_ARGS+=(--dims "$(IFS=','; echo "${MATRYOSHKA_DONE[*]}")")
    else
        TABLE_ARGS=()   # no dim produced, no table
    fi
fi

if [[ ${#TABLE_ARGS[@]} -gt 0 ]]; then
    echo ""
    echo "[table] exporting table -> $EVAL_ROOT/results_table.xlsx"
    python "$TABLE_SCRIPT" "${TABLE_ARGS[@]}" \
        || echo "[warn] table export failed (saved evaluation results are unaffected)"
fi

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo ""
echo "=============================================================="
echo "[all done] total time ${TOTAL_ELAPSED}s (single model load + all embeddings + all evaluations)"
echo "[all done] summary log: $SUMMARY_LOG"

if [[ ${#FAILED_DATASETS[@]} -gt 0 ]]; then
    echo "[all done] the following items failed or were skipped; please check:"
    for item in "${FAILED_DATASETS[@]}"; do
        echo "  - $item"
    done
    exit 1
else
    echo "[all done] all datasets succeeded"
fi
