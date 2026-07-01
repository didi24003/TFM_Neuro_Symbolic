#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/05_train_eegnet_baseline_random_split_60_20_20.py"
CONFIG_SWEEP_SCRIPT="${SCRIPT_DIR}/run_eegnet_baseline_random_split_60_20_20.sh"
SUMMARY_SCRIPT="${SCRIPT_DIR}/summarize_eegnet_baseline_random_split_60_20_20.py"

PYTHON_BIN="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/BCICIV_2a_mat}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/experiments_mi_ltn/runs/eegnet_baseline_random_split_60_20_20}"
CONFIG_RUN_ROOT="${OUT_DIR}/config_runs"
BEST_SEEDS_RUN_ROOT="${OUT_DIR}/best_config_seeds"
LOG_DIR="${OUT_DIR}/logs"
SEEDS="${SEEDS:-0 7 42 123 2024}"
DEVICE_ARG="${DEVICE_ARG:-auto}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:-}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
CONFIG_SWEEP_SEED="${CONFIG_SWEEP_SEED:-42}"
CONFIG_SWEEP_SPLIT_SEED="${CONFIG_SWEEP_SPLIT_SEED:-${CONFIG_SWEEP_SEED}}"

DRY_RUN=false
SUMMARY_ONLY=false
OVERWRITE=false
CURRENT_STEP="initialization"

usage() {
  cat <<'EOF'
Usage:
  bash experiments_mi_ltn/run_eegnet_baseline_random_split_60_20_20_full_pipeline.sh [--dry-run] [--summary-only] [--overwrite]

Environment variables:
  DATA_ROOT=/path/to/BCICIV_2a_mat
  OUT_DIR=experiments_mi_ltn/runs/eegnet_baseline_random_split_60_20_20
  SEEDS="0 7 42 123 2024"
  DEVICE_ARG=auto
  NUM_WORKERS=0
  LIMIT_SAMPLES=
  CHECKPOINT_EVERY=10
  CONFIG_SWEEP_SEED=42
  CONFIG_SWEEP_SPLIT_SEED=42
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --summary-only)
      SUMMARY_ONLY=true
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

on_error() {
  local exit_code=$?
  echo "Pipeline failed at step: ${CURRENT_STEP}" >&2
  echo "Command: ${BASH_COMMAND}" >&2
  exit "${exit_code}"
}
trap on_error ERR

check_prereqs() {
  CURRENT_STEP="check_prereqs"
  if [[ ! -f "${REPO_ROOT}/setup.py" || ! -d "${REPO_ROOT}/torcheeg" ]]; then
    echo "Repository root check failed." >&2
    exit 1
  fi
  for path in "${TRAIN_SCRIPT}" "${CONFIG_SWEEP_SCRIPT}" "${SUMMARY_SCRIPT}"; do
    if [[ ! -f "${path}" ]]; then
      echo "Required file missing: ${path}" >&2
      exit 1
    fi
  done
  if [[ "${SUMMARY_ONLY}" != "true" && ! -d "${DATA_ROOT}" ]]; then
    echo "Dataset directory not found: ${DATA_ROOT}" >&2
    exit 1
  fi
}

prepare_output_dirs() {
  CURRENT_STEP="prepare_output_dirs"
  mkdir -p "${OUT_DIR}" "${CONFIG_RUN_ROOT}" "${BEST_SEEDS_RUN_ROOT}" "${LOG_DIR}"
}

run_config_sweep() {
  CURRENT_STEP="run_config_sweep"
  echo "[Phase 1/4] Launching EEGNet random-split configuration sweep"
  local sweep_args=()
  if [[ "${DRY_RUN}" == "true" ]]; then
    sweep_args+=(--dry-run)
  fi
  if [[ "${OVERWRITE}" == "true" ]]; then
    sweep_args+=(--overwrite)
  fi
  run_cmd env \
    DATA_ROOT="${DATA_ROOT}" \
    OUT_DIR="${OUT_DIR}" \
    DEVICE_ARG="${DEVICE_ARG}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    LIMIT_SAMPLES="${LIMIT_SAMPLES}" \
    CHECKPOINT_EVERY="${CHECKPOINT_EVERY}" \
    CONFIG_SWEEP_SEED="${CONFIG_SWEEP_SEED}" \
    CONFIG_SWEEP_SPLIT_SEED="${CONFIG_SWEEP_SPLIT_SEED}" \
    bash "${CONFIG_SWEEP_SCRIPT}" "${sweep_args[@]}"
}

run_summary() {
  CURRENT_STEP="run_summary"
  echo "[Phase 2/4 or 4/4] Regenerating summaries and lightweight figures"
  run_cmd "${PYTHON_BIN}" "${SUMMARY_SCRIPT}" --run-root "${OUT_DIR}"
}

read_selected_field() {
  local field="$1"
  "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
path = Path(${OUT_DIR@Q}) / "selection_summary.json"
with path.open() as handle:
    data = json.load(handle)
selected = data.get("selected_configuration") or {}
print(selected.get(${field@Q}, ""))
PY
}

run_best_config_seeds() {
  CURRENT_STEP="run_best_config_seeds"
  echo "[Phase 3/4] Launching selected configuration across seeds"

  if [[ "${DRY_RUN}" == "true" && ! -f "${OUT_DIR}/selection_summary.json" ]]; then
    echo "[dry-run] Skipping best-config seed expansion because ${OUT_DIR}/selection_summary.json does not exist yet."
    echo "[dry-run] In a real run, phase 2 writes that file and phase 3 expands the selected configuration across ${SEEDS}."
    return 0
  fi

  local config_name epochs batch_size learning_rate weight_decay scheduler plateau_factor plateau_patience early_stopping_patience
  config_name="$(read_selected_field "config_name")"
  if [[ -z "${config_name}" ]]; then
    echo "selection_summary.json does not contain a selected configuration." >&2
    exit 1
  fi
  epochs="$(read_selected_field "epochs_requested")"
  batch_size="$(read_selected_field "batch_size")"
  learning_rate="$(read_selected_field "learning_rate")"
  weight_decay="$(read_selected_field "weight_decay")"
  scheduler="$(read_selected_field "scheduler")"
  plateau_factor="$(read_selected_field "plateau_factor")"
  plateau_patience="$(read_selected_field "plateau_patience")"
  early_stopping_patience="$(read_selected_field "early_stopping_patience")"

  local extra_args=()
  if [[ -n "${LIMIT_SAMPLES}" ]]; then
    extra_args+=(--limit-samples "${LIMIT_SAMPLES}")
  fi
  if [[ "${OVERWRITE}" == "true" ]]; then
    extra_args+=(--overwrite)
  fi

  for seed in ${SEEDS}; do
    local split_seed="${seed}"
    local run_id="${config_name}_seed${seed}_split${split_seed}"
    if [[ -f "${BEST_SEEDS_RUN_ROOT}/${run_id}/summary.json" && "${OVERWRITE}" != "true" ]]; then
      echo "SKIP ${run_id}: summary.json already exists."
      continue
    fi
    echo "START ${run_id}"
    run_cmd "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
      --data-root "${DATA_ROOT}" \
      --experiment-root "${OUT_DIR}" \
      --run-root "${BEST_SEEDS_RUN_ROOT}" \
      --run-id "${run_id}" \
      --config-name "${config_name}" \
      --phase-tag "best_config_seeds" \
      --train-ratio 0.6 \
      --val-ratio 0.2 \
      --test-ratio 0.2 \
      --epochs "${epochs}" \
      --batch-size "${batch_size}" \
      --learning-rate "${learning_rate}" \
      --weight-decay "${weight_decay}" \
      --scheduler "${scheduler}" \
      --plateau-factor "${plateau_factor}" \
      --plateau-patience "${plateau_patience}" \
      --early-stopping-patience "${early_stopping_patience}" \
      --checkpoint-every "${CHECKPOINT_EVERY}" \
      --seed "${seed}" \
      --split-seed "${split_seed}" \
      --device "${DEVICE_ARG}" \
      --num-workers "${NUM_WORKERS}" \
      "${extra_args[@]}"
  done
}

setup_logs() {
  mkdir -p "${LOG_DIR}"
}

CURRENT_LOG="${LOG_DIR}/run_configs.log"
setup_logs

check_prereqs
prepare_output_dirs

if [[ "${SUMMARY_ONLY}" == "true" ]]; then
  CURRENT_LOG="${LOG_DIR}/summary.log"
  if [[ "${DRY_RUN}" != "true" ]]; then
    exec > >(tee -a "${CURRENT_LOG}") 2>&1
  fi
  run_summary
  exit 0
fi

if [[ "${DRY_RUN}" != "true" ]]; then
  exec > >(tee -a "${CURRENT_LOG}") 2>&1
fi
run_config_sweep

CURRENT_LOG="${LOG_DIR}/summary.log"
if [[ "${DRY_RUN}" != "true" ]]; then
  exec > >(tee -a "${CURRENT_LOG}") 2>&1
fi
run_summary

CURRENT_LOG="${LOG_DIR}/run_seeds.log"
if [[ "${DRY_RUN}" != "true" ]]; then
  exec > >(tee -a "${CURRENT_LOG}") 2>&1
fi
run_best_config_seeds

CURRENT_LOG="${LOG_DIR}/summary.log"
if [[ "${DRY_RUN}" != "true" ]]; then
  exec > >(tee -a "${CURRENT_LOG}") 2>&1
fi
run_summary

echo
echo "Results root: ${OUT_DIR}"
echo "Config runs: ${CONFIG_RUN_ROOT}"
echo "Best seeds: ${BEST_SEEDS_RUN_ROOT}"
echo "Logs: ${LOG_DIR}"
