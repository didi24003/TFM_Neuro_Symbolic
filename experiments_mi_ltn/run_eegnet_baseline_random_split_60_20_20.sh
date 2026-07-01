#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/05_train_eegnet_baseline_random_split_60_20_20.py"

PYTHON_BIN="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/BCICIV_2a_mat}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/experiments_mi_ltn/runs/eegnet_baseline_random_split_60_20_20}"
CONFIG_RUN_ROOT="${OUT_DIR}/config_runs"
DEVICE_ARG="${DEVICE_ARG:-auto}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:-}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
CONFIG_SWEEP_SEED="${CONFIG_SWEEP_SEED:-42}"
CONFIG_SWEEP_SPLIT_SEED="${CONFIG_SWEEP_SPLIT_SEED:-${CONFIG_SWEEP_SEED}}"
CONFIG_NAMES="${CONFIG_NAMES:-baseline_short long_training_300 long_training_500 lr_low lr_high weight_decay scheduler_plateau}"

DRY_RUN=false
OVERWRITE=false

usage() {
  cat <<'EOF'
Usage:
  bash experiments_mi_ltn/run_eegnet_baseline_random_split_60_20_20.sh [--dry-run] [--overwrite]

Environment variables:
  DATA_ROOT=/path/to/BCICIV_2a_mat
  OUT_DIR=experiments_mi_ltn/runs/eegnet_baseline_random_split_60_20_20
  CONFIG_SWEEP_SEED=42
  CONFIG_SWEEP_SPLIT_SEED=42
  DEVICE_ARG=auto
  NUM_WORKERS=0
  LIMIT_SAMPLES=
  CHECKPOINT_EVERY=10
  CONFIG_NAMES="baseline_short long_training_300 ..."
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --overwrite)
      OVERWRITE=true
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

run_cmd() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '[dry-run] %q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

check_prereqs() {
  if [[ ! -f "${REPO_ROOT}/setup.py" || ! -d "${REPO_ROOT}/torcheeg" ]]; then
    echo "Run this script from the repository root." >&2
    exit 1
  fi
  if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "Training script not found: ${TRAIN_SCRIPT}" >&2
    exit 1
  fi
  if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "Dataset folder not found: ${DATA_ROOT}" >&2
    exit 1
  fi
  mkdir -p "${CONFIG_RUN_ROOT}"
}

config_args() {
  case "$1" in
    baseline_short)
      printf '%s\n' --epochs 100 --batch-size 64 --learning-rate 5e-4 --weight-decay 0 --scheduler none --early-stopping-patience 50
      ;;
    long_training_300)
      printf '%s\n' --epochs 300 --batch-size 64 --learning-rate 9e-4 --weight-decay 0 --scheduler none --early-stopping-patience 50
      ;;
    long_training_500)
      printf '%s\n' --epochs 500 --batch-size 64 --learning-rate 9e-4 --weight-decay 0 --scheduler none --early-stopping-patience 75
      ;;
    lr_low)
      printf '%s\n' --epochs 300 --batch-size 64 --learning-rate 5e-4 --weight-decay 0 --scheduler none --early-stopping-patience 50
      ;;
    lr_high)
      printf '%s\n' --epochs 300 --batch-size 64 --learning-rate 1e-3 --weight-decay 0 --scheduler none --early-stopping-patience 50
      ;;
    weight_decay)
      printf '%s\n' --epochs 300 --batch-size 64 --learning-rate 9e-4 --weight-decay 1e-4 --scheduler none --early-stopping-patience 50
      ;;
    scheduler_plateau)
      printf '%s\n' --epochs 300 --batch-size 64 --learning-rate 9e-4 --weight-decay 0 --scheduler plateau --plateau-factor 0.5 --plateau-patience 10 --early-stopping-patience 50
      ;;
    *)
      echo "Unknown config: $1" >&2
      exit 1
      ;;
  esac
}

run_config() {
  local config_name="$1"
  local run_id="${config_name}_seed${CONFIG_SWEEP_SEED}_split${CONFIG_SWEEP_SPLIT_SEED}"
  local extra_args=()
  if [[ -n "${LIMIT_SAMPLES}" ]]; then
    extra_args+=(--limit-samples "${LIMIT_SAMPLES}")
  fi
  if [[ -f "${CONFIG_RUN_ROOT}/${run_id}/summary.json" && "${OVERWRITE}" != "true" ]]; then
    echo "SKIP ${run_id}: summary.json already exists."
    return 0
  fi

  mapfile -t config_extra < <(config_args "${config_name}")
  echo "START ${run_id}"
  run_cmd "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --data-root "${DATA_ROOT}" \
    --experiment-root "${OUT_DIR}" \
    --run-root "${CONFIG_RUN_ROOT}" \
    --run-id "${run_id}" \
    --config-name "${config_name}" \
    --phase-tag "config_sweep" \
    --train-ratio 0.6 \
    --val-ratio 0.2 \
    --test-ratio 0.2 \
    --seed "${CONFIG_SWEEP_SEED}" \
    --split-seed "${CONFIG_SWEEP_SPLIT_SEED}" \
    --device "${DEVICE_ARG}" \
    --num-workers "${NUM_WORKERS}" \
    --checkpoint-every "${CHECKPOINT_EVERY}" \
    "${extra_args[@]}" \
    $([[ "${OVERWRITE}" == "true" ]] && printf '%s' "--overwrite") \
    "${config_extra[@]}"
}

check_prereqs

for config_name in ${CONFIG_NAMES}; do
  run_config "${config_name}"
done

echo "Config sweep directory: ${CONFIG_RUN_ROOT}"
