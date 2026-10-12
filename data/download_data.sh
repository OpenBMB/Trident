#!/usr/bin/env bash
# =============================================================================
# download_data.sh — download & prepare the Trident training / evaluation data
#
#   Train : https://huggingface.co/datasets/OpenBMB/Trident-Train
#   Eval  : https://huggingface.co/datasets/OpenBMB/Trident-Eval
#
# Layout produced (relative to --data-dir, default: the directory of this script):
#
#   Train/train_{all,vdoc,nature}.jsonl
#   Train/images/<...>                      # extracted from images/*.tar.gz
#   <Dataset>/{queries,origin_corpus,mix_corpus,qrels}.jsonl
#   <Dataset>/images/<...>                  # extracted from images.tar.gz
#
# Works with bash >= 3.2 (macOS default) and Linux. Re-running is safe:
# downloads resume, finished extractions are skipped.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAIN_REPO="OpenBMB/Trident-Train"
EVAL_REPO="OpenBMB/Trident-Eval"

TRAIN_ARCHIVES="aokvqa docvqa dude fvqa infovqa mscoco openwikitable render_images visualmrc visualnews wiki"
VDOC_DATASETS="ChartQA DocVQA InfoVQA SlideVQA ViDoSeek Dude"
NATURAL_DATASETS="Google_WIT MSCOCO VisualNews OVEN"
ALL_EVAL_DATASETS="$VDOC_DATASETS $NATURAL_DATASETS"

# ----------------------------- defaults --------------------------------------
TARGET=""
DATA_DIR="$SCRIPT_DIR"
TRAIN_SUBSET="all"        # all | vdoc | natural
TRAIN_IMAGES=""           # comma-separated archive names; empty = all
BENCHMARK=""              # vdoc | natural; empty = all eval datasets
EVAL_DATASETS=""          # comma-separated dataset names; overrides --benchmark
NO_EXTRACT=0
REMOVE_ARCHIVES=0
FORCE=0
JOBS=4

# ----------------------------- logging ---------------------------------------
if [ -t 1 ]; then
  BOLD=$'\033[1m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; RESET=$'\033[0m'
else
  BOLD=""; GREEN=""; YELLOW=""; RED=""; RESET=""
fi
info() { printf '%s[trident]%s %s\n' "$GREEN" "$RESET" "$*"; }
warn() { printf '%s[trident] WARN:%s %s\n' "$YELLOW" "$RESET" "$*" >&2; }
die()  { printf '%s[trident] ERROR:%s %s\n' "$RED" "$RESET" "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Trident data downloader

Usage:
  bash data/download_data.sh <target> [options]

Targets:
  train     Download Trident-Train (jsonl + image archives) into <data-dir>/Train
  eval      Download Trident-Eval  (per-dataset jsonl + images) into <data-dir>/<Dataset>
  all       train + eval

Common options:
  --data-dir DIR        Where to put the data (default: directory of this script)
  --mirror              Use https://hf-mirror.com (sets HF_ENDPOINT unless already set)
  --token TOKEN         Hugging Face token (exported as HF_TOKEN; not needed for public data)
  --jobs N              Parallel extraction jobs (default: 4)
  --no-extract          Download only, do not extract the image archives
  --remove-archives     Delete each .tar.gz after it has been extracted
  --force               Re-extract archives even if they were extracted before
  -h, --help            Show this help

Train options:
  --train-subset S      all (default) | vdoc | natural
                        -> train_all.jsonl / train_vdoc.jsonl / train_nature.jsonl
  --train-images LIST   Comma-separated image archives to fetch (default: all), from:
                        aokvqa,docvqa,dude,fvqa,infovqa,mscoco,openwikitable,
                        render_images,visualmrc,visualnews,wiki

Eval options:
  --benchmark B         vdoc  : ChartQA, DocVQA, InfoVQA, SlideVQA, ViDoSeek, Dude
                        natural: Google_WIT, MSCOCO, VisualNews, OVEN
                        (default: all ten datasets)
  --eval-datasets LIST  Comma-separated dataset names, e.g. ChartQA,DocVQA

Examples:
  bash data/download_data.sh all
  bash data/download_data.sh train --train-subset vdoc --train-images docvqa,dude,infovqa
  bash data/download_data.sh eval  --benchmark vdoc
  bash data/download_data.sh eval  --eval-datasets ChartQA,MSCOCO --mirror

Notes:
  * Image archives are large; check your free disk space first.
  * Re-running resumes interrupted downloads and skips finished extractions.
    Archives removed with --remove-archives are not downloaded again.
  * Requires the Hugging Face CLI:  pip install -U "huggingface_hub[cli]"
EOF
}

# ----------------------------- helpers ---------------------------------------
in_list() {  # in_list "a b c" x  -> 0 if x is in the space separated list
  local item
  for item in $1; do
    if [ "$item" = "$2" ]; then return 0; fi
  done
  return 1
}

csv_to_words() { echo "${1//,/ }"; }

need_arg() {  # need_arg "$@" : option must be followed by a value
  if [ $# -lt 2 ]; then die "Option $1 requires a value (see --help)"; fi
}

HF_CLI=()
find_hf_cli() {
  if command -v hf >/dev/null 2>&1; then
    HF_CLI=(hf)
  elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_CLI=(huggingface-cli)
  else
    die "Hugging Face CLI not found. Install it with:  pip install -U \"huggingface_hub[cli]\""
  fi
}

# hf_download <repo> <local_dir> <file>...   (files are repo-relative paths)
hf_download() {
  local repo="$1" local_dir="$2"
  shift 2
  info "Downloading ${#} file(s) from ${BOLD}${repo}${RESET} -> ${local_dir}"
  "${HF_CLI[@]}" download "$repo" "$@" --repo-type dataset --local-dir "$local_dir"
}

# extract_archive <archive> <dest_dir>
extract_archive() {
  local archive="$1" dest="$2" stamp="${1}.extracted"
  if [ ! -f "$archive" ]; then
    warn "Archive not found, skipping: $archive"
    return 1
  fi
  if [ -f "$stamp" ] && [ "$FORCE" -eq 0 ]; then
    info "Already extracted, skipping: $(basename "$archive")"
  else
    info "Extracting $(basename "$archive") -> $dest"
    if tar -xzf "$archive" -C "$dest"; then
      touch "$stamp"
    else
      warn "Failed to extract $archive"
      return 1
    fi
  fi
  if [ "$REMOVE_ARCHIVES" -eq 1 ]; then rm -f "$archive"; fi
  return 0
}

# extract_parallel "<archive>|<dest>" ...
extract_parallel() {
  local fail=0 item pid
  local pids=()
  for item in "$@"; do
    extract_archive "${item%%|*}" "${item#*|}" &
    pids+=("$!")
    if [ "${#pids[@]}" -ge "$JOBS" ]; then
      for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
      pids=()
    fi
  done
  if [ "${#pids[@]}" -gt 0 ]; then
    for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
  fi
  return "$fail"
}

# already_done <archive> : extracted before and archive removed afterwards
already_done() {
  [ "$FORCE" -eq 0 ] && [ -f "${1}.extracted" ] && [ ! -f "$1" ]
}

count_lines() { wc -l < "$1" | tr -d ' '; }

# ----------------------------- train -----------------------------------------
download_train() {
  local dest="$DATA_DIR/Train" img_dir="$DATA_DIR/Train/images"
  local names name archive
  local files=("$TRAIN_JSONL")
  local extract_list=()
  mkdir -p "$img_dir"

  if [ -n "$TRAIN_IMAGES" ]; then names="$(csv_to_words "$TRAIN_IMAGES")"; else names="$TRAIN_ARCHIVES"; fi
  for name in $names; do
    in_list "$TRAIN_ARCHIVES" "$name" || die "Unknown training image archive '$name'. Valid: ${TRAIN_ARCHIVES// /, }"
    archive="$img_dir/$name.tar.gz"
    if already_done "$archive"; then
      info "images/$name already extracted — skipping download"
      continue
    fi
    files+=("images/$name.tar.gz")
    extract_list+=("$archive|$img_dir")
  done

  hf_download "$TRAIN_REPO" "$dest" "${files[@]}"

  if [ "$NO_EXTRACT" -eq 0 ] && [ "${#extract_list[@]}" -gt 0 ]; then
    extract_parallel "${extract_list[@]}" || die "Some training image archives failed to extract."
  fi
}

verify_train() {
  local f="$DATA_DIR/Train/$TRAIN_JSONL"
  [ -f "$f" ] || die "Missing $f"
  info "Train OK: $TRAIN_JSONL ($(count_lines "$f") lines)"
  if [ "$NO_EXTRACT" -eq 0 ] && [ -z "$(ls -A "$DATA_DIR/Train/images" 2>/dev/null | grep -v '\.tar\.gz' || true)" ]; then
    warn "Train/images contains no extracted files — check the archives."
  fi
}

# ----------------------------- eval ------------------------------------------
download_eval() {
  local ds archive
  local files=() extract_list=()
  for ds in $EVAL_SEL; do
    mkdir -p "$DATA_DIR/$ds"
    files+=("$ds/queries.jsonl" "$ds/origin_corpus.jsonl" "$ds/mix_corpus.jsonl" "$ds/qrels.jsonl")
    archive="$DATA_DIR/$ds/images.tar.gz"
    if already_done "$archive"; then
      info "$ds/images already extracted — skipping download"
      continue
    fi
    files+=("$ds/images.tar.gz")
    extract_list+=("$archive|$DATA_DIR/$ds")
  done

  hf_download "$EVAL_REPO" "$DATA_DIR" "${files[@]}"

  if [ "$NO_EXTRACT" -eq 0 ] && [ "${#extract_list[@]}" -gt 0 ]; then
    extract_parallel "${extract_list[@]}" || die "Some evaluation image archives failed to extract."
  fi
}

verify_eval() {
  local ds f missing=0
  for ds in $EVAL_SEL; do
    for f in queries.jsonl origin_corpus.jsonl mix_corpus.jsonl qrels.jsonl; do
      if [ ! -f "$DATA_DIR/$ds/$f" ]; then warn "Missing $ds/$f"; missing=1; fi
    done
    if [ "$NO_EXTRACT" -eq 0 ] && [ ! -d "$DATA_DIR/$ds/images" ]; then
      warn "Missing $ds/images (not extracted?)"; missing=1
    fi
  done
  if [ "$missing" -eq 1 ]; then die "Eval data incomplete."; fi
  info "Eval OK: $(echo "$EVAL_SEL" | wc -w | tr -d ' ') dataset(s): $EVAL_SEL"
}

# ----------------------------- argument parsing ------------------------------
if [ $# -eq 0 ]; then usage; exit 0; fi
case "$1" in
  train|eval|all) TARGET="$1"; shift ;;
  -h|--help)      usage; exit 0 ;;
  *)              die "Unknown target '$1' (expected: train | eval | all). See --help." ;;
esac

MIRROR=0
while [ $# -gt 0 ]; do
  case "$1" in
    --data-dir)         need_arg "$@"; DATA_DIR="$2"; shift 2 ;;
    --train-subset)     need_arg "$@"; TRAIN_SUBSET="$2"; shift 2 ;;
    --train-images)     need_arg "$@"; TRAIN_IMAGES="$2"; shift 2 ;;
    --benchmark)        need_arg "$@"; BENCHMARK="$2"; shift 2 ;;
    --eval-datasets)    need_arg "$@"; EVAL_DATASETS="$2"; shift 2 ;;
    --jobs)             need_arg "$@"; JOBS="$2"; shift 2 ;;
    --token)            need_arg "$@"; export HF_TOKEN="$2"; shift 2 ;;
    --mirror)           MIRROR=1; shift ;;
    --no-extract)       NO_EXTRACT=1; shift ;;
    --remove-archives)  REMOVE_ARCHIVES=1; shift ;;
    --force)            FORCE=1; shift ;;
    -h|--help)          usage; exit 0 ;;
    *)                  die "Unknown option '$1'. See --help." ;;
  esac
done

case "$JOBS" in ''|*[!0-9]*) die "--jobs must be a positive integer" ;; esac
if [ "$JOBS" -lt 1 ]; then die "--jobs must be >= 1"; fi

case "$TRAIN_SUBSET" in
  all)            TRAIN_JSONL="train_all.jsonl" ;;
  vdoc)           TRAIN_JSONL="train_vdoc.jsonl" ;;
  natural|nature) TRAIN_JSONL="train_nature.jsonl" ;;
  *)              die "--train-subset must be one of: all, vdoc, natural" ;;
esac

if [ -n "$EVAL_DATASETS" ]; then
  EVAL_SEL="$(csv_to_words "$EVAL_DATASETS")"
  for d in $EVAL_SEL; do
    in_list "$ALL_EVAL_DATASETS" "$d" || die "Unknown eval dataset '$d'. Valid: ${ALL_EVAL_DATASETS// /, }"
  done
elif [ -n "$BENCHMARK" ]; then
  case "$BENCHMARK" in
    vdoc)    EVAL_SEL="$VDOC_DATASETS" ;;
    natural) EVAL_SEL="$NATURAL_DATASETS" ;;
    *)       die "--benchmark must be vdoc or natural" ;;
  esac
else
  EVAL_SEL="$ALL_EVAL_DATASETS"
fi

mkdir -p "$DATA_DIR"
DATA_DIR="$(cd "$DATA_DIR" && pwd)"

if [ "$MIRROR" -eq 1 ]; then export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"; fi
if command -v python3 >/dev/null 2>&1 && python3 -c 'import hf_transfer' >/dev/null 2>&1; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
fi
command -v tar >/dev/null 2>&1 || die "'tar' is required."
find_hf_cli

# ----------------------------- run -------------------------------------------
info "Data dir : $DATA_DIR"
info "Endpoint : ${HF_ENDPOINT:-https://huggingface.co}"
if [ "$TARGET" = "train" ] || [ "$TARGET" = "all" ]; then info "Train    : $TRAIN_JSONL + images (${TRAIN_IMAGES:-all archives})"; fi
if [ "$TARGET" = "eval" ]  || [ "$TARGET" = "all" ]; then info "Eval     : $EVAL_SEL"; fi

if [ "$TARGET" = "train" ] || [ "$TARGET" = "all" ]; then
  download_train
  verify_train
fi
if [ "$TARGET" = "eval" ] || [ "$TARGET" = "all" ]; then
  download_eval
  verify_eval
fi

echo
info "${BOLD}Done.${RESET} Next steps:"
if [ "$TARGET" = "train" ] || [ "$TARGET" = "all" ]; then
  echo "  train : bash scripts/train.sh configs/trident_qwen3vl.yaml --set data.train_path=$DATA_DIR/Train/$TRAIN_JSONL"
fi
if [ "$TARGET" = "eval" ] || [ "$TARGET" = "all" ]; then
  echo "  eval  : bash scripts/eval.sh trident_qwen3vl outputs/trident_qwen3vl_2b/final vdoc --base_model Qwen/Qwen3-VL-2B-Instruct"
fi
