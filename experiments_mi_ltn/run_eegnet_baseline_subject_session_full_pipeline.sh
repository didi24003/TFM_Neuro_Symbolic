#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/05_train_eegnet_baseline_subject_session.py"
CONFIG_SWEEP_SCRIPT="${SCRIPT_DIR}/run_eegnet_baseline_subject_session_configs.sh"
SUMMARY_SCRIPT="${SCRIPT_DIR}/summarize_eegnet_baseline_subject_session.py"

PYTHON_BIN="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/BCICIV_2a_mat}"
SEEDS="${SEEDS:-0 7 42 123 2024}"
DEVICE_ARG="${DEVICE_ARG:-auto}"
NUM_WORKERS="${NUM_WORKERS:-0}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
CONFIG_SWEEP_SEED="${CONFIG_SWEEP_SEED:-42}"
CONFIG_SWEEP_SPLIT_SEED="${CONFIG_SWEEP_SPLIT_SEED:-${CONFIG_SWEEP_SEED}}"
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
BEST_SEEDS_RUN_ROOT="${RUN_ROOT}/best_config_seeds"
LOG_DIR="${RUN_ROOT}/logs"
ANALYSIS_FIGURES_DIR="${RUN_ROOT}/analysis_figures"
BEST_CONFIG_JSON="${RUN_ROOT}/baseline_subject_session_best_config_summary.json"
BEST_EXPORT_TAR="${RUN_ROOT}/best_export_eegnet_baseline_subject_session.tar.gz"

DRY_RUN=false
SUMMARY_ONLY=false
CURRENT_STEP="initialization"

usage() {
  cat <<'EOF'
Usage:
  bash experiments_mi_ltn/run_eegnet_baseline_subject_session_full_pipeline.sh [--dry-run] [--summary-only] [--help]

Main modes:
  bash experiments_mi_ltn/run_eegnet_baseline_subject_session_full_pipeline.sh
  bash experiments_mi_ltn/run_eegnet_baseline_subject_session_full_pipeline.sh --dry-run
  bash experiments_mi_ltn/run_eegnet_baseline_subject_session_full_pipeline.sh --summary-only

Environment variables:
  DATA_ROOT=/path/to/dataset
  SEEDS="0 7 42 123 2024"
  OUT_DIR=experiments_mi_ltn/runs/eegnet_baseline_subject_session_artifact_free
  DEVICE_ARG=auto
  NUM_WORKERS=0
  CHECKPOINT_EVERY=10
  CONFIG_SWEEP_SEED=42
  CONFIG_SWEEP_SPLIT_SEED=42
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
    --summary-only)
      SUMMARY_ONLY=true
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

on_error() {
  local exit_code=$?
  echo "Pipeline failed at step: ${CURRENT_STEP}" >&2
  echo "Command: ${BASH_COMMAND}" >&2
  exit "${exit_code}"
}
trap on_error ERR

mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/baseline_subject_session_master_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

run_cmd() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '[dry-run] %q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

check_repo_root() {
  CURRENT_STEP="check_repo_root"
  if [[ ! -f "${REPO_ROOT}/setup.py" || ! -d "${REPO_ROOT}/torcheeg" ]]; then
    echo "Repository root check failed. Run this script from the repository checkout." >&2
    exit 1
  fi
}

check_required_files() {
  CURRENT_STEP="check_required_files"
  local required=(
    "${TRAIN_SCRIPT}"
    "${CONFIG_SWEEP_SCRIPT}"
    "${SUMMARY_SCRIPT}"
  )
  for path in "${required[@]}"; do
    if [[ ! -e "${path}" ]]; then
      echo "Required file missing: ${path}" >&2
      exit 1
    fi
  done
}

check_dataset() {
  CURRENT_STEP="check_dataset"
  if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "Dataset directory not found: ${DATA_ROOT}" >&2
    exit 1
  fi
}

prepare_output_dirs() {
  CURRENT_STEP="prepare_output_dirs"
  mkdir -p "${RUN_ROOT}" "${CONFIG_RUN_ROOT}" "${BEST_SEEDS_RUN_ROOT}" "${LOG_DIR}" "${ANALYSIS_FIGURES_DIR}"
  echo "Results root: ${RUN_ROOT}"
  echo "Logs: ${LOG_DIR}"
  echo "Old results are preserved; this pipeline does not delete previous artifacts."
}

run_config_sweep() {
  CURRENT_STEP="run_config_sweep"
  echo "[Phase 1/6] Launching EEGNet baseline subject-session configuration sweep"
  run_cmd env \
    DATA_ROOT="${DATA_ROOT}" \
    DEVICE_ARG="${DEVICE_ARG}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    CHECKPOINT_EVERY="${CHECKPOINT_EVERY}" \
    CONFIG_SWEEP_SEED="${CONFIG_SWEEP_SEED}" \
    CONFIG_SWEEP_SPLIT_SEED="${CONFIG_SWEEP_SPLIT_SEED}" \
    VAL_MODE="${VAL_MODE}" \
    SELECTION_METRIC="${SELECTION_METRIC}" \
    SKIP_ARTIFACTS="${SKIP_ARTIFACTS}" \
    OUT_DIR="${RUN_ROOT}" \
    RUN_ROOT="${RUN_ROOT}" \
    bash "${CONFIG_SWEEP_SCRIPT}" $([[ "${DRY_RUN}" == "true" ]] && printf '%s' "--dry-run")
}

run_summary() {
  CURRENT_STEP="run_summary"
  echo "[Phase 2/6 or 6/6] Regenerating summaries, figures, notes and export bundle"
  run_cmd "${PYTHON_BIN}" "${SUMMARY_SCRIPT}" --run-root "${RUN_ROOT}"
}

read_selected_config_field() {
  local field="$1"
  "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
path = Path(${BEST_CONFIG_JSON@Q})
with path.open() as f:
    data = json.load(f)
selected = data["selected_config"]
value = selected[${field@Q}]
print(value)
PY
}

run_best_config_seeds() {
  CURRENT_STEP="run_best_config_seeds"
  echo "[Phase 4/6] Relaunching the selected configuration across seeds: ${SEEDS}"

  local config_name
  local epochs
  local batch_size
  local learning_rate
  local weight_decay
  local scheduler
  local plateau_factor
  local plateau_patience
  local early_stopping_patience

  config_name="$(read_selected_config_field "config_name")"
  epochs="$(read_selected_config_field "epochs_requested")"
  batch_size="$(read_selected_config_field "batch_size")"
  learning_rate="$(read_selected_config_field "learning_rate")"
  weight_decay="$(read_selected_config_field "weight_decay")"
  scheduler="$(read_selected_config_field "scheduler")"
  plateau_factor="$(read_selected_config_field "plateau_factor")"
  plateau_patience="$(read_selected_config_field "plateau_patience")"
  early_stopping_patience="$(read_selected_config_field "early_stopping_patience")"

  for seed in ${SEEDS}; do
    local split_seed="${seed}"
    local run_id="${config_name}_seed${seed}_split${split_seed}"
    if [[ -f "${BEST_SEEDS_RUN_ROOT}/${run_id}/summary.json" ]]; then
      echo "SKIP ${run_id}: summary.json already exists."
      continue
    fi

    echo "START ${run_id}"
    run_cmd "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
      --data-root "${DATA_ROOT}" \
      --run-root "${BEST_SEEDS_RUN_ROOT}" \
      --run-id "${run_id}" \
      --config-name "${config_name}" \
      --phase-tag "best_config_seeds" \
      --epochs "${epochs}" \
      --batch-size "${batch_size}" \
      --learning-rate "${learning_rate}" \
      --weight-decay "${weight_decay}" \
      --scheduler "${scheduler}" \
      --plateau-factor "${plateau_factor}" \
      --plateau-patience "${plateau_patience}" \
      --early-stopping-patience "${early_stopping_patience}" \
      --checkpoint-every "${CHECKPOINT_EVERY}" \
      --val-ratio 0.2 \
      --val-mode "${VAL_MODE}" \
      --selection-metric "${SELECTION_METRIC}" \
      --seed "${seed}" \
	      --split-seed "${split_seed}" \
	      --device "${DEVICE_ARG}" \
	      --num-workers "${NUM_WORKERS}" \
	      --skip-trial-with-artifacts "${SKIP_ARTIFACTS}"
  done
}

print_final_locations() {
  echo
  echo "Results root: ${RUN_ROOT}"
  echo "Config runs: ${CONFIG_RUN_ROOT}"
  echo "Best-config seeds: ${BEST_SEEDS_RUN_ROOT}"
  echo "Logs: ${LOG_DIR}"
  echo "Figures: ${ANALYSIS_FIGURES_DIR}"
  echo "Config summary CSV: ${RUN_ROOT}/baseline_subject_session_configs_summary.csv"
  echo "Best-config summary JSON: ${BEST_CONFIG_JSON}"
  echo "Best seeds summary CSV: ${RUN_ROOT}/baseline_subject_session_best_seeds_summary.csv"
  echo "Interpretation notes: ${RUN_ROOT}/interpretation_notes.txt"
  echo "Compressed export: ${BEST_EXPORT_TAR}"
  echo "Master log: ${MASTER_LOG}"
}

check_repo_root
check_required_files
check_dataset
prepare_output_dirs

if [[ "${SUMMARY_ONLY}" == "false" ]]; then
  run_config_sweep
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "[Phase 2/6] Dry-run summary step"
  echo "[dry-run] ${PYTHON_BIN} ${SUMMARY_SCRIPT}"
else
  run_summary
fi

echo "[Phase 3/6] Selecting best configuration using validation-only summary"
if [[ "${DRY_RUN}" == "true" && ! -f "${BEST_CONFIG_JSON}" ]]; then
  echo "[dry-run] Selection file will be created by the real summary step: ${BEST_CONFIG_JSON}"
else
  echo "Best-config summary: ${BEST_CONFIG_JSON}"
fi

if [[ "${SUMMARY_ONLY}" == "false" ]]; then
  if [[ "${DRY_RUN}" == "true" && ! -f "${BEST_CONFIG_JSON}" ]]; then
    echo "[dry-run] Skipping seed expansion because ${BEST_CONFIG_JSON} does not exist yet."
  else
    run_best_config_seeds
    echo "[Phase 5/6] Rebuilding summary after multi-seed runs"
    if [[ "${DRY_RUN}" == "true" ]]; then
      echo "[dry-run] ${PYTHON_BIN} ${SUMMARY_SCRIPT}"
    else
      run_summary
    fi
  fi
fi

print_final_locations
