#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SWEEP_SCRIPT="${SCRIPT_DIR}/run_eegnet_logic_subject_session_sweep.sh"
SUMMARY_SCRIPT="${SCRIPT_DIR}/summarize_eegnet_logic_subject_session.py"
BASELINE_RUN_ROOT="${REPO_ROOT}/experiments_mi_ltn/runs/eegnet_baseline_subject_session"
RUN_ROOT="${REPO_ROOT}/experiments_mi_ltn/runs/eegnet_logic_subject_session"
LOG_DIR="${RUN_ROOT}/logs"

PYTHON_BIN="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/BCICIV_2a_mat}"
SEEDS="${SEEDS:-0 7 42 123 2024}"
DEVICE_ARG="${DEVICE_ARG:-auto}"
NUM_WORKERS="${NUM_WORKERS:-0}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
VAL_MODE="${VAL_MODE:-stratified_trialwise}"
SELECTION_METRIC="${SELECTION_METRIC:-val_acc}"
LAMBDA_RULES="${LAMBDA_RULES:-0.0 0.001 0.01 0.05 0.1 0.2 0.5 1.0}"

DRY_RUN=false
SUMMARY_ONLY=false
CURRENT_STEP="initialization"

usage() {
  cat <<'EOF'
Usage:
  bash experiments_mi_ltn/run_eegnet_logic_subject_session_full_pipeline.sh [--dry-run] [--summary-only] [--help]

Main modes:
  bash experiments_mi_ltn/run_eegnet_logic_subject_session_full_pipeline.sh
  bash experiments_mi_ltn/run_eegnet_logic_subject_session_full_pipeline.sh --dry-run
  bash experiments_mi_ltn/run_eegnet_logic_subject_session_full_pipeline.sh --summary-only

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
MASTER_LOG="${LOG_DIR}/eegnet_logic_subject_session_master_$(date '+%Y%m%d_%H%M%S').log"
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
    "${SWEEP_SCRIPT}"
    "${SUMMARY_SCRIPT}"
    "${BASELINE_RUN_ROOT}/baseline_subject_session_best_config_summary.json"
    "${BASELINE_RUN_ROOT}/baseline_subject_session_best_seeds_summary.csv"
  )
  for path in "${required[@]}"; do
    if [[ ! -e "${path}" ]]; then
      echo "Required file missing: ${path}" >&2
      exit 1
    fi
  done
}

check_dataset_if_needed() {
  CURRENT_STEP="check_dataset_if_needed"
  if [[ "${SUMMARY_ONLY}" == "true" || "${DRY_RUN}" == "true" ]]; then
    return 0
  fi
  if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "Dataset directory not found: ${DATA_ROOT}" >&2
    exit 1
  fi
}

prepare_output_dirs() {
  CURRENT_STEP="prepare_output_dirs"
  mkdir -p "${RUN_ROOT}" "${RUN_ROOT}/logic_runs" "${RUN_ROOT}/analysis_figures" "${LOG_DIR}"
  echo "Results root: ${RUN_ROOT}"
  echo "Baseline reference root: ${BASELINE_RUN_ROOT}"
  echo "Old results are preserved; this pipeline does not delete previous artifacts."
}

run_sweep() {
  CURRENT_STEP="run_sweep"
  echo "[Phase 1/2] Launching EEGNet + gates / EEGNet + gates + rule sweep"
  run_cmd env \
    DATA_ROOT="${DATA_ROOT}" \
    SEEDS="${SEEDS}" \
    LAMBDA_RULES="${LAMBDA_RULES}" \
    DEVICE_ARG="${DEVICE_ARG}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    CHECKPOINT_EVERY="${CHECKPOINT_EVERY}" \
    VAL_MODE="${VAL_MODE}" \
    SELECTION_METRIC="${SELECTION_METRIC}" \
    bash "${SWEEP_SCRIPT}" $([[ "${DRY_RUN}" == "true" ]] && printf '%s' "--dry-run")
}

run_summary() {
  CURRENT_STEP="run_summary"
  echo "[Phase 2/2] Regenerating logic summaries, comparisons and figures"
  run_cmd "${PYTHON_BIN}" "${SUMMARY_SCRIPT}" \
    --run-root "${RUN_ROOT}" \
    --logic-run-root "${RUN_ROOT}/logic_runs" \
    --baseline-run-root "${BASELINE_RUN_ROOT}"
}

print_final_locations() {
  echo
  echo "Results root: ${RUN_ROOT}"
  echo "Logic runs: ${RUN_ROOT}/logic_runs"
  echo "Subject comparison CSV: ${RUN_ROOT}/eegnet_logic_subject_session_subject_comparison.csv"
  echo "Global comparison CSV: ${RUN_ROOT}/eegnet_logic_subject_session_global_comparison.csv"
  echo "Selection summary JSON: ${RUN_ROOT}/eegnet_logic_subject_session_selection_summary.json"
  echo "Figures: ${RUN_ROOT}/analysis_figures"
  echo "Master log: ${MASTER_LOG}"
}

check_repo_root
check_required_files
check_dataset_if_needed
prepare_output_dirs

if [[ "${SUMMARY_ONLY}" == "false" ]]; then
  run_sweep
fi

run_summary
print_final_locations
