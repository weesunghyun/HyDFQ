#!/usr/bin/env bash
set -euo pipefail

# Launch hybrid QAT training with explicit distributed environment variables
# and an on-the-fly config override for the synthetic/OOD shard prefixes.
# Example (single-node DDP with two GPUs):
#   torchrun --nproc_per_node=2 scripts/run_hybrid_qat.sh \
#     --config ./config/cifar10_resnet20.hocon \
#     --synthetic-prefix ./data/cifar10/resnet20_cifar10_refined_gaussian_hardsample_beta10.0_gamma2.0_group \
#     --label-prefix ./data/cifar10/resnet20_cifar10_labels_hardsample_beta10.0_gamma2.0_group \
#     --student-model resnet20_cifar10 --epochs 151 --batch-size 64

show_help() {
  cat <<'EOF'
Usage: scripts/run_hybrid_qat.sh [options]

Options:
  --config PATH            Base HOCON config (default: ./config/cifar10_resnet20.hocon)
  --synthetic-prefix PATH  Prefix to synthetic data shards (generateDataPath)
  --label-prefix PATH      Prefix to synthetic label shards (generateLabelPath)
  --student-model NAME     Model name used for student and teacher (Option.model_name)
  --epochs N               Override nEpochs in config
  --batch-size N           Override batchSize in config
  --nproc-per-node N       torchrun --nproc_per_node (default: 1)
  --master-addr HOST       Distributed master address (default: 127.0.0.1)
  --master-port PORT       Distributed master port (default: 29500)
  --world-size N           WORLD_SIZE export when not set by torchrun (default: nproc-per-node)
  --local-rank N           LOCAL_RANK export when not set by torchrun (default: 0)
  -h, --help               Show this help message

Notes:
  * The script rewrites generateDataPath/generateLabelPath in a temporary config
    so that main_direct.py loads the correct synthetic/OOD shards
    (<prefix>{1..4}.pickle).
  * Option.model_name feeds both teacher and student instantiation in main_direct.py,
    so --student-model selects both roles by default.
EOF
}

CONFIG_PATH=./config/cifar10_resnet20.hocon
SYNTHETIC_PREFIX=./data/cifar10/resnet20_cifar10_refined_gaussian_hardsample_beta10.0_gamma2.0_group
LABEL_PREFIX=./data/cifar10/resnet20_cifar10_labels_hardsample_beta10.0_gamma2.0_group
STUDENT_MODEL=""
NUM_EPOCHS=""
TRAIN_BATCH_SIZE=""
NPROC_PER_NODE=1
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
WORLD_SIZE=${WORLD_SIZE:-}
LOCAL_RANK=${LOCAL_RANK:-}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --synthetic-prefix)
      SYNTHETIC_PREFIX="$2"
      shift 2
      ;;
    --label-prefix)
      LABEL_PREFIX="$2"
      shift 2
      ;;
    --student-model)
      STUDENT_MODEL="$2"
      shift 2
      ;;
    --epochs)
      NUM_EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      TRAIN_BATCH_SIZE="$2"
      shift 2
      ;;
    --nproc-per-node)
      NPROC_PER_NODE="$2"
      shift 2
      ;;
    --master-addr)
      MASTER_ADDR="$2"
      shift 2
      ;;
    --master-port)
      MASTER_PORT="$2"
      shift 2
      ;;
    --world-size)
      WORLD_SIZE="$2"
      shift 2
      ;;
    --local-rank)
      LOCAL_RANK="$2"
      shift 2
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      show_help
      exit 1
      ;;
  esac
done

if [[ -z "${WORLD_SIZE}" ]]; then
  WORLD_SIZE="${NPROC_PER_NODE}"
fi
if [[ -z "${LOCAL_RANK}" ]]; then
  LOCAL_RANK=0
fi

export MASTER_ADDR MASTER_PORT WORLD_SIZE LOCAL_RANK

TMP_CONF="$(mktemp /tmp/hybrid_qat.XXXXXX.hocon)"
python - <<'PY' "${CONFIG_PATH}" "${TMP_CONF}" "${SYNTHETIC_PREFIX}" "${LABEL_PREFIX}" "${STUDENT_MODEL}" "${NUM_EPOCHS}" "${TRAIN_BATCH_SIZE}"
import sys
import os
from pyhocon import ConfigFactory, HOCONConverter

config_path, tmp_path, synthetic_prefix, label_prefix, student_model, epochs, batch_size = sys.argv[1:]
conf = ConfigFactory.parse_file(config_path)

if synthetic_prefix:
    conf["generateDataPath"] = synthetic_prefix
if label_prefix:
    conf["generateLabelPath"] = label_prefix
if student_model:
    conf["model_name"] = student_model
if epochs:
    conf["nEpochs"] = int(epochs)
if batch_size:
    conf["batchSize"] = int(batch_size)

with open(tmp_path, "w", encoding="utf-8") as handle:
    handle.write(HOCONConverter.to_hocon(conf))
print(f"Wrote distributed config to {tmp_path}")
PY

echo "Using LOCAL_RANK=${LOCAL_RANK}, WORLD_SIZE=${WORLD_SIZE}, MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"
echo "Launching main_direct.py with config ${TMP_CONF}"

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  main_direct.py --conf_path "${TMP_CONF}"
