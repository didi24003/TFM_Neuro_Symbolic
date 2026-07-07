#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAIN_SCRIPT="${REPO_ROOT}/experiments_mi_ltn/train_eegnex_subject_session.py"

PYTHON_BIN="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/BCICIV_2a_mat}"
SEEDS="${SEEDS:-0 7 42 123 2024}"
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
SCHEDULER="${SCHEDULER:-none}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-50}"
SPLIT_SEED="${SPLIT_SEED:-42}"
DEVICE_ARG="${DEVICE_ARG:-auto}"
NUM_WORKERS="${NUM_WORKERS:-0}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
VAL_RATIO="${VAL_RATIO:-0.2}"
VAL_MODE="${VAL_MODE:-stratified_trialwise}"
SELECTION_METRIC="${SELECTION_METRIC:-val_acc}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/experiments_mi_ltn/runs/eegnex_artifact_free}"
RUN_ROOT="${RUN_ROOT:-${OUTPUT_ROOT}/gates}"
SKIP_ARTIFACTS=1

DRY_RUN=false

usage() {
  cat <<'EOF'
Usage:
  bash experiments_mi_ltn/artifact_free/run_eegnex_gates_artifact_free.sh [--dry-run]
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

echo "Artifact-free EEGNeX + gates configuration"
echo "  data_root: ${DATA_ROOT}"
echo "  train_script: ${TRAIN_SCRIPT}"
echo "  run_root: ${RUN_ROOT}"
echo "  seeds: ${SEEDS}"
echo "  protocol: subject-specific cross-session"
echo "  train/validation session: T"
echo "  test session: E"
echo "  eeg channels as input: 22"
echo "  eog channels as input: no"
echo "  artifact handling: exclusion of trials marked as artifact"
echo "  model_mode: gates"
echo

for seed in ${SEEDS}; do
  run_id="eegnex_gates_artifact_free_seed${seed}_split${SPLIT_SEED}"
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
    --config-name "gates"
    --phase-tag "artifact_free_gates"
    --model-mode "gates"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --learning-rate "${LEARNING_RATE}"
    --weight-decay "${WEIGHT_DECAY}"
    --scheduler "${SCHEDULER}"
    --early-stopping-patience "${EARLY_STOPPING_PATIENCE}"
    --checkpoint-every "${CHECKPOINT_EVERY}"
    --val-ratio "${VAL_RATIO}"
    --val-mode "${VAL_MODE}"
    --selection-metric "${SELECTION_METRIC}"
    --seed "${seed}"
    --split-seed "${SPLIT_SEED}"
    --device "${DEVICE_ARG}"
    --num-workers "${NUM_WORKERS}"
    --skip-trial-with-artifacts "${SKIP_ARTIFACTS}"
    --lambda-rule "0.0"
  )

  echo "START ${run_id}"
  run_cmd "${cmd[@]}"
done
