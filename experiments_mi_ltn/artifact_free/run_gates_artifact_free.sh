#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAIN_SCRIPT="${REPO_ROOT}/experiments_mi_ltn/06_train_eegnet_logic_subject_session.py"

PYTHON_BIN="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/BCICIV_2a_mat}"
SEEDS="${SEEDS:-0 7 42 123 2024}"
DEVICE_ARG="${DEVICE_ARG:-auto}"
NUM_WORKERS="${NUM_WORKERS:-0}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/experiments_mi_ltn/runs/eegnet_gates_subject_session_artifact_free}"
RUN_ROOT="${RUN_ROOT:-${OUTPUT_ROOT}/logic_runs}"
CONFIG_NAME="${CONFIG_NAME:-lr_high}"
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
SCHEDULER="${SCHEDULER:-none}"
PLATEAU_FACTOR="${PLATEAU_FACTOR:-0.5}"
PLATEAU_PATIENCE="${PLATEAU_PATIENCE:-10}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-50}"
SPLIT_SEED="${SPLIT_SEED:-42}"
VAL_RATIO="${VAL_RATIO:-0.2}"
VAL_MODE="${VAL_MODE:-stratified_trialwise}"
SELECTION_METRIC="${SELECTION_METRIC:-val_acc}"
SKIP_ARTIFACTS="${SKIP_ARTIFACTS:-1}"

DRY_RUN=false

usage() {
  cat <<'EOF'
Usage:
  bash experiments_mi_ltn/artifact_free/run_gates_artifact_free.sh [--dry-run]

Environment variables:
  DATA_ROOT=/path/to/BCICIV_2a_mat
  SEEDS="0 7 42 123 2024"
  DEVICE_ARG=auto
  NUM_WORKERS=0
  CHECKPOINT_EVERY=10
  OUTPUT_ROOT=/path/to/experiments_mi_ltn/runs/eegnet_gates_subject_session_artifact_free
  RUN_ROOT=/path/to/experiments_mi_ltn/runs/eegnet_gates_subject_session_artifact_free/logic_runs
  CONFIG_NAME=lr_high
  EPOCHS=300
  BATCH_SIZE=64
  LEARNING_RATE=0.001
  WEIGHT_DECAY=0.0
  SCHEDULER=none
  EARLY_STOPPING_PATIENCE=50
  SPLIT_SEED=42
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

run_cmd() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '[dry-run]'
    for arg in "$@"; do
      printf ' %q' "${arg}"
    done
    printf '\n'
  else
    "$@"
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Required file not found: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "Required directory not found: ${path}" >&2
    exit 1
  fi
}

require_dir "${DATA_ROOT}"
require_file "${TRAIN_SCRIPT}"
mkdir -p "${RUN_ROOT}"

echo "Artifact-free EEGNet + gates configuration"
echo "  data_root: ${DATA_ROOT}"
echo "  train_script: ${TRAIN_SCRIPT}"
echo "  output_root: ${OUTPUT_ROOT}"
echo "  run_root: ${RUN_ROOT}"
echo "  seeds: ${SEEDS}"
echo "  subjects: all available subjects"
echo "  protocol: subject-specific cross-session"
echo "  train/validation session: T"
echo "  test session: E"
echo "  artifact handling: exclusion of trials marked as artifact"
echo "  model_type: eegnet_gates_no_rule"
echo "  lambda_rule: 0.0"
echo "  fixed default config: config=${CONFIG_NAME} epochs=${EPOCHS} batch_size=${BATCH_SIZE} lr=${LEARNING_RATE} weight_decay=${WEIGHT_DECAY} scheduler=${SCHEDULER} split_seed=${SPLIT_SEED}"
echo

for seed in ${SEEDS}; do
  split_seed="${SPLIT_SEED}"
  run_id="eegnet_gates_no_rule_artifact_free_seed${seed}_split${split_seed}"
  summary_path="${RUN_ROOT}/${run_id}/summary.json"

  if [[ -f "${summary_path}" ]]; then
    echo "SKIP ${run_id}: ${summary_path} already exists."
    continue
  fi

  cmd=(
    "${PYTHON_BIN}" "${TRAIN_SCRIPT}"
    --data-root "${DATA_ROOT}"
    --run-root "${RUN_ROOT}"
    --run-id "${run_id}"
    --config-name "${CONFIG_NAME}"
    --phase-tag "artifact_free_gates"
    --model-type "eegnet_gates_no_rule"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --learning-rate "${LEARNING_RATE}"
    --weight-decay "${WEIGHT_DECAY}"
    --scheduler "${SCHEDULER}"
    --plateau-factor "${PLATEAU_FACTOR}"
    --plateau-patience "${PLATEAU_PATIENCE}"
    --early-stopping-patience "${EARLY_STOPPING_PATIENCE}"
    --checkpoint-every "${CHECKPOINT_EVERY}"
    --val-ratio "${VAL_RATIO}"
    --val-mode "${VAL_MODE}"
    --selection-metric "${SELECTION_METRIC}"
    --seed "${seed}"
    --split-seed "${split_seed}"
    --device "${DEVICE_ARG}"
    --num-workers "${NUM_WORKERS}"
    --skip-trial-with-artifacts "${SKIP_ARTIFACTS}"
    --lambda-rule "0.0"
  )

  echo "START ${run_id}"
  run_cmd "${cmd[@]}"
done

echo
echo "Artifact-free gates runs saved under:"
echo "  ${RUN_ROOT}"
