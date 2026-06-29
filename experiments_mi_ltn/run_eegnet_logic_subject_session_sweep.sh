#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/06_train_eegnet_logic_subject_session.py"
RUN_ROOT="${REPO_ROOT}/experiments_mi_ltn/runs/eegnet_logic_subject_session"
LOGIC_RUN_ROOT="${RUN_ROOT}/logic_runs"

PYTHON_BIN="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/BCICIV_2a_mat}"
DEVICE_ARG="${DEVICE_ARG:-auto}"
NUM_WORKERS="${NUM_WORKERS:-0}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
VAL_MODE="${VAL_MODE:-stratified_trialwise}"
SELECTION_METRIC="${SELECTION_METRIC:-val_acc}"
SEEDS="${SEEDS:-0 7 42 123 2024}"
LAMBDA_RULES="${LAMBDA_RULES:-0.0 0.001 0.01 0.05 0.1 0.2 0.5 1.0}"
CONFIG_NAME="${CONFIG_NAME:-weight_decay}"
LEARNING_RATE="${LEARNING_RATE:-9e-4}"
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
SCHEDULER="${SCHEDULER:-none}"
PLATEAU_FACTOR="${PLATEAU_FACTOR:-0.5}"
PLATEAU_PATIENCE="${PLATEAU_PATIENCE:-10}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-50}"

DRY_RUN=false

usage() {
  cat <<'EOF'
Usage:
  bash experiments_mi_ltn/run_eegnet_logic_subject_session_sweep.sh [--dry-run]

Environment variables:
  DATA_ROOT=/path/to/dataset
  SEEDS="0 7 42 123 2024"
  LAMBDA_RULES="0.0 0.001 0.01 0.05 0.1 0.2 0.5 1.0"
  DEVICE_ARG=auto
  NUM_WORKERS=0
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

if [[ "${DRY_RUN}" == "false" && ! -d "${DATA_ROOT}" ]]; then
  echo "Dataset folder not found: ${DATA_ROOT}" >&2
  exit 1
fi

mkdir -p "${LOGIC_RUN_ROOT}"

run_cmd() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '[dry-run] %q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run_experiment() {
  local model_type="$1"
  local lambda_rule="$2"
  local seed="$3"
  local split_seed="$4"
  local lambda_tag
  lambda_tag="$(printf '%s' "${lambda_rule}" | tr '.' 'p')"
  local run_id="${model_type}_lam${lambda_tag}_seed${seed}_split${split_seed}"

  if [[ -f "${LOGIC_RUN_ROOT}/${run_id}/summary.json" ]]; then
    echo "SKIP ${run_id}: summary.json already exists."
    return 0
  fi

  echo "START ${run_id}"
  run_cmd "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --data-root "${DATA_ROOT}" \
    --run-root "${LOGIC_RUN_ROOT}" \
    --run-id "${run_id}" \
    --config-name "${CONFIG_NAME}" \
    --phase-tag "logic_sweep" \
    --model-type "${model_type}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --learning-rate "${LEARNING_RATE}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --scheduler "${SCHEDULER}" \
    --plateau-factor "${PLATEAU_FACTOR}" \
    --plateau-patience "${PLATEAU_PATIENCE}" \
    --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
    --checkpoint-every "${CHECKPOINT_EVERY}" \
    --val-ratio 0.2 \
    --val-mode "${VAL_MODE}" \
    --selection-metric "${SELECTION_METRIC}" \
    --seed "${seed}" \
    --split-seed "${split_seed}" \
    --lambda-rule "${lambda_rule}" \
    --device "${DEVICE_ARG}" \
    --num-workers "${NUM_WORKERS}"
}

for seed in ${SEEDS}; do
  split_seed="${seed}"
  run_experiment "eegnet_gates_no_rule" "0.0" "${seed}" "${split_seed}"
  for lambda_rule in ${LAMBDA_RULES}; do
    if [[ "${lambda_rule}" == "0.0" ]]; then
      continue
    fi
    run_experiment "eegnet_gates_rule" "${lambda_rule}" "${seed}" "${split_seed}"
  done
done

echo "Logic run directory: ${LOGIC_RUN_ROOT}"
