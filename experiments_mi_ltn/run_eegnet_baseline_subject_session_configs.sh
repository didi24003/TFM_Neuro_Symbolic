#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/05_train_eegnet_baseline_subject_session.py"

PYTHON_BIN="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/BCICIV_2a_mat}"
DEVICE_ARG="${DEVICE_ARG:-auto}"
NUM_WORKERS="${NUM_WORKERS:-0}"
CONFIG_SWEEP_SEED="${CONFIG_SWEEP_SEED:-42}"
CONFIG_SWEEP_SPLIT_SEED="${CONFIG_SWEEP_SPLIT_SEED:-${CONFIG_SWEEP_SEED}}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
VAL_MODE="${VAL_MODE:-stratified_trialwise}"
SELECTION_METRIC="${SELECTION_METRIC:-val_acc}"
SKIP_ARTIFACTS="${SKIP_ARTIFACTS:-0}"

if [[ "${SKIP_ARTIFACTS}" == "1" ]]; then
  DEFAULT_RUN_ROOT="${REPO_ROOT}/experiments_mi_ltn/runs/eegnet_baseline_subject_session_artifact_free"
else
  DEFAULT_RUN_ROOT="${REPO_ROOT}/experiments_mi_ltn/runs/eegnet_baseline_subject_session"
fi
RUN_ROOT="${OUT_DIR:-${RUN_ROOT:-${DEFAULT_RUN_ROOT}}}"
CONFIG_RUN_ROOT="${RUN_ROOT}/config_runs"

DRY_RUN=false

usage() {
  cat <<'EOF'
Usage:
  bash experiments_mi_ltn/run_eegnet_baseline_subject_session_configs.sh [--dry-run]

Environment variables:
  DATA_ROOT=/path/to/dataset
  DEVICE_ARG=auto
  NUM_WORKERS=0
  CONFIG_SWEEP_SEED=42
  CONFIG_SWEEP_SPLIT_SEED=42
  CHECKPOINT_EVERY=10
  VAL_MODE=stratified_trialwise
  SELECTION_METRIC=val_acc
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
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

run_cmd() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '[dry-run] %q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run_experiment() {
  local config_name="$1"
  local run_id="$2"
  shift 2

  if [[ -f "${CONFIG_RUN_ROOT}/${run_id}/summary.json" ]]; then
    echo "SKIP ${run_id}: summary.json already exists."
    return 0
  fi

  echo "START ${run_id}"
  run_cmd "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --data-root "${DATA_ROOT}" \
    --run-root "${CONFIG_RUN_ROOT}" \
    --run-id "${run_id}" \
    --config-name "${config_name}" \
    --phase-tag "config_sweep" \
    --val-ratio 0.2 \
    --val-mode "${VAL_MODE}" \
    --selection-metric "${SELECTION_METRIC}" \
    --seed "${CONFIG_SWEEP_SEED}" \
    --split-seed "${CONFIG_SWEEP_SPLIT_SEED}" \
    --device "${DEVICE_ARG}" \
    --num-workers "${NUM_WORKERS}" \
    --checkpoint-every "${CHECKPOINT_EVERY}" \
    --skip-trial-with-artifacts "${SKIP_ARTIFACTS}" \
    "$@"
}

run_experiment "baseline_short" "baseline_short_seed${CONFIG_SWEEP_SEED}_split${CONFIG_SWEEP_SPLIT_SEED}" \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 5e-4 \
  --weight-decay 0 \
  --scheduler none \
  --early-stopping-patience 50

run_experiment "long_training_300" "long_training_300_seed${CONFIG_SWEEP_SEED}_split${CONFIG_SWEEP_SPLIT_SEED}" \
  --epochs 300 \
  --batch-size 64 \
  --learning-rate 9e-4 \
  --weight-decay 0 \
  --scheduler none \
  --early-stopping-patience 50

run_experiment "long_training_500" "long_training_500_seed${CONFIG_SWEEP_SEED}_split${CONFIG_SWEEP_SPLIT_SEED}" \
  --epochs 500 \
  --batch-size 64 \
  --learning-rate 9e-4 \
  --weight-decay 0 \
  --scheduler none \
  --early-stopping-patience 50

run_experiment "lr_low" "lr_low_seed${CONFIG_SWEEP_SEED}_split${CONFIG_SWEEP_SPLIT_SEED}" \
  --epochs 300 \
  --batch-size 64 \
  --learning-rate 5e-4 \
  --weight-decay 0 \
  --scheduler none \
  --early-stopping-patience 50

run_experiment "lr_high" "lr_high_seed${CONFIG_SWEEP_SEED}_split${CONFIG_SWEEP_SPLIT_SEED}" \
  --epochs 300 \
  --batch-size 64 \
  --learning-rate 1e-3 \
  --weight-decay 0 \
  --scheduler none \
  --early-stopping-patience 50

run_experiment "weight_decay" "weight_decay_seed${CONFIG_SWEEP_SEED}_split${CONFIG_SWEEP_SPLIT_SEED}" \
  --epochs 300 \
  --batch-size 64 \
  --learning-rate 9e-4 \
  --weight-decay 1e-4 \
  --scheduler none \
  --early-stopping-patience 50

run_experiment "scheduler_plateau" "scheduler_plateau_seed${CONFIG_SWEEP_SEED}_split${CONFIG_SWEEP_SPLIT_SEED}" \
  --epochs 300 \
  --batch-size 64 \
  --learning-rate 9e-4 \
  --weight-decay 0 \
  --scheduler plateau \
  --plateau-factor 0.5 \
  --plateau-patience 10 \
  --early-stopping-patience 50

echo "Config sweep directory: ${CONFIG_RUN_ROOT}"
