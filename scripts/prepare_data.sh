#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: prepare_data.sh [options]

Generate distilled data and curate samples in one pass.
All options can be set via flags or by editing the configuration
variables below. By default, the curated output directory aligns
with the distilled data save path.

Options:
  --model <name>                Teacher/model name for both stages (default: resnet18)
  --generate-batch-size <int>   Batch size used during synthetic data generation (default: 256)
  --test-batch-size <int>       Test batch size for generation sanity checks (default: 512)
  --group-count <int>           Number of distilled data groups to create (default: 4)
  --beta <float>                Beta hyperparameter for generation (default: 0.1)
  --gamma <float>               Gamma hyperparameter for generation (default: 0.5)
  --save-path-head <path>       Root directory for distilled data and default curated output (default: ./data/distilled)
  --dataset-path <path>         Path to the OOD dataset root (ImageFolder layout) [required]
  --output-dir <path>           Directory to store curated shards (default: same as --save-path-head)
  --file-prefix <name>          Prefix for curated shard filenames (default: unified_curated)
  --sample-batch-size <int>     Batch size for teacher scoring during sampling (default: 1024)
  --feature-batch-size <int>    Optional override for feature extraction batch size
  --num-workers <int>           Dataloader worker count for sampling (default: 16)
  --prefetch-factor <int>       Optional prefetch factor for sampling dataloader workers
  --num-augmentations <int>     Number of lightweight augmentations per image (default: 5)
  -h, --help                    Show this message
EOF
}

# ------------------------------- #
# User-adjustable configuration   #
# ------------------------------- #
MODEL="resnet18"
GENERATE_BATCH_SIZE=256
TEST_BATCH_SIZE=512
GROUP_COUNT=4
BETA=0.1
GAMMA=0.5
SAVE_PATH_HEAD="${SAVE_PATH_HEAD:-./data/distilled}"
DATASET_PATH="${DATASET_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
FILE_PREFIX="unified_curated"
SAMPLE_BATCH_SIZE=1024
FEATURE_BATCH_SIZE=""
NUM_WORKERS=16
PREFETCH_FACTOR=""
NUM_AUGMENTATIONS=5

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --generate-batch-size) GENERATE_BATCH_SIZE="$2"; shift 2 ;;
    --test-batch-size) TEST_BATCH_SIZE="$2"; shift 2 ;;
    --group-count) GROUP_COUNT="$2"; shift 2 ;;
    --beta) BETA="$2"; shift 2 ;;
    --gamma) GAMMA="$2"; shift 2 ;;
    --save-path-head) SAVE_PATH_HEAD="$2"; shift 2 ;;
    --dataset-path) DATASET_PATH="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --file-prefix) FILE_PREFIX="$2"; shift 2 ;;
    --sample-batch-size) SAMPLE_BATCH_SIZE="$2"; shift 2 ;;
    --feature-batch-size) FEATURE_BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --prefetch-factor) PREFETCH_FACTOR="$2"; shift 2 ;;
    --num-augmentations) NUM_AUGMENTATIONS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$DATASET_PATH" ]]; then
  echo "Please provide --dataset-path to point to the OOD dataset root." >&2
  exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$SAVE_PATH_HEAD"
fi

mkdir -p "$SAVE_PATH_HEAD" "$OUTPUT_DIR"

echo "Generating distilled data to: $SAVE_PATH_HEAD"
for ((g=1; g<=GROUP_COUNT; g++)); do
  echo "  -> Group $g"
  python data_generate/generate_data.py \
    --model="$MODEL" \
    --batch_size="$GENERATE_BATCH_SIZE" \
    --test_batch_size="$TEST_BATCH_SIZE" \
    --group="$g" \
    --beta="$BETA" \
    --gamma="$GAMMA" \
    --save_path_head="$SAVE_PATH_HEAD"
done

echo "Sampling curated data to: $OUTPUT_DIR (prefix: $FILE_PREFIX)"
SAMPLE_CMD=(python data_generate/sample_data.py
  --model="$MODEL"
  --dataset_path="$DATASET_PATH"
  --output_dir="$OUTPUT_DIR"
  --file_prefix="$FILE_PREFIX"
  --batch_size="$SAMPLE_BATCH_SIZE"
  --num_groups="$GROUP_COUNT"
  --num_workers="$NUM_WORKERS"
  --num_augmentations="$NUM_AUGMENTATIONS"
)

if [[ -n "$FEATURE_BATCH_SIZE" ]]; then
  SAMPLE_CMD+=(--feature_batch_size="$FEATURE_BATCH_SIZE")
fi

if [[ -n "$PREFETCH_FACTOR" ]]; then
  SAMPLE_CMD+=(--prefetch_factor="$PREFETCH_FACTOR")
fi

"${SAMPLE_CMD[@]}"

echo "Data generation and sampling complete."
