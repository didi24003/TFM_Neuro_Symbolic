#!/usr/bin/env bash

set -u
set -o pipefail

RUN_ROOT="experiments_mi_ltn/runs/baseline_eegnet"
LOG_FILE="${RUN_ROOT}/phase1_run.log"
TRAIN_SCRIPT="experiments_mi_ltn/02_train_eegnet_bciciv2a.py"
DATA_DIR="data/BCICIV_2a_mat"
PYTHON_BIN="${PYTHON:-python}"
FORCE="${FORCE:-0}"

if [[ ! -f "setup.py" || ! -d "torcheeg" ]]; then
  echo "ERROR: run this script from the repository root."
  echo "Expected: ./experiments_mi_ltn/run_baseline_phase1.sh"
  exit 1
fi

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "ERROR: training script not found: ${TRAIN_SCRIPT}"
  echo "Run this script from the repository root after cloning the complete repository."
  exit 1
fi

mkdir -p "${RUN_ROOT}"
touch "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "======================================================================"
echo "EEGNet baseline phase 1"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Repository: $(pwd)"
echo "Python: ${PYTHON_BIN}"
echo "FORCE=${FORCE}"
echo "Log: ${LOG_FILE}"
echo "======================================================================"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo
  echo "ERROR: dataset folder not found: ${DATA_DIR}"
  echo "The BCI Competition IV 2a .mat files must be placed at:"
  echo "  ${DATA_DIR}/"
  echo "This repository does not include the dataset by default."
  echo "Download or copy it from the shared external folder and rerun this script."
  echo "If the maintainer later enables Git LFS for the dataset, run: git lfs pull"
  exit 1
fi

MAT_COUNT="$(find "${DATA_DIR}" -maxdepth 1 -type f -name '*.mat' | wc -l)"
if [[ "${MAT_COUNT}" -lt 18 ]]; then
  echo
  echo "WARNING: expected 18 .mat files in ${DATA_DIR}, found ${MAT_COUNT}."
  echo "Training may fail if the BCI Competition IV 2a dataset is incomplete."
fi

echo
echo "CUDA check"
if ! "${PYTHON_BIN}" - <<'PY'
import torch

available = torch.cuda.is_available()
print(f"torch.__version__={torch.__version__}")
print(f"torch.cuda.is_available()={available}")
if available:
    print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
    for idx in range(torch.cuda.device_count()):
        print(f"gpu[{idx}]={torch.cuda.get_device_name(idx)}")
else:
    print("WARNING: no CUDA GPU detected. Runs may be very slow on CPU.")
PY
then
  echo "ERROR: Python/PyTorch CUDA check failed. Verify the virtual environment and dependencies."
  exit 1
fi

declare -a COMPLETED_RUNS=()
declare -a FAILED_RUNS=()
declare -a SKIPPED_RUNS=()

format_duration() {
  local total_seconds="$1"
  local hours=$((total_seconds / 3600))
  local minutes=$(((total_seconds % 3600) / 60))
  local seconds=$((total_seconds % 60))
  printf "%02dh:%02dm:%02ds" "${hours}" "${minutes}" "${seconds}"
}

run_experiment() {
  local run_id="$1"
  shift

  local run_dir="${RUN_ROOT}/${run_id}"
  local actual_run_id="${run_id}"

  if [[ -f "${run_dir}/summary.json" && "${FORCE}" != "1" ]]; then
    echo
    echo "----------------------------------------------------------------------"
    echo "SKIP ${run_id}"
    echo "Reason: ${run_dir}/summary.json already exists. Use FORCE=1 to repeat."
    echo "----------------------------------------------------------------------"
    SKIPPED_RUNS+=("${run_id}")
    return 0
  fi

  if [[ -d "${run_dir}" && "${FORCE}" == "1" ]]; then
    actual_run_id="${run_id}_force_$(date '+%Y%m%d_%H%M%S')"
    echo
    echo "FORCE=1: existing run directory found for ${run_id}."
    echo "The repeated run will be saved as ${actual_run_id} to avoid overwriting old results."
  fi

  echo
  echo "----------------------------------------------------------------------"
  echo "START ${actual_run_id}"
  echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "----------------------------------------------------------------------"

  local start_seconds
  start_seconds="$(date +%s)"

  "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --run-id "${actual_run_id}" \
    "$@"

  local exit_code=$?
  local end_seconds
  end_seconds="$(date +%s)"
  local duration=$((end_seconds - start_seconds))

  echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Duration: $(format_duration "${duration}")"

  if [[ "${exit_code}" -eq 0 ]]; then
    echo "STATUS ${actual_run_id}: OK"
    COMPLETED_RUNS+=("${actual_run_id}")
  else
    echo "STATUS ${actual_run_id}: FAILED with exit code ${exit_code}"
    FAILED_RUNS+=("${actual_run_id}")
  fi

  return 0
}

run_experiment "eegnet_canonical_seed42_ep100_lr9e-4_bs64" \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 9e-4 \
  --weight-decay 0 \
  --scheduler none \
  --early-stopping-patience 0 \
  --checkpoint-every 10 \
  --seed 42

run_experiment "eegnet_long300_seed42_lr9e-4_es50" \
  --epochs 300 \
  --batch-size 64 \
  --learning-rate 9e-4 \
  --weight-decay 0 \
  --scheduler none \
  --early-stopping-patience 50 \
  --checkpoint-every 10 \
  --seed 42

run_experiment "eegnet_long500_seed42_lr9e-4_es75" \
  --epochs 500 \
  --batch-size 64 \
  --learning-rate 9e-4 \
  --weight-decay 0 \
  --scheduler none \
  --early-stopping-patience 75 \
  --checkpoint-every 10 \
  --seed 42

run_experiment "eegnet_lr5e-4_seed42_ep300_es50" \
  --epochs 300 \
  --batch-size 64 \
  --learning-rate 5e-4 \
  --weight-decay 0 \
  --scheduler none \
  --early-stopping-patience 50 \
  --checkpoint-every 10 \
  --seed 42

run_experiment "eegnet_lr1e-3_seed42_ep300_es50" \
  --epochs 300 \
  --batch-size 64 \
  --learning-rate 1e-3 \
  --weight-decay 0 \
  --scheduler none \
  --early-stopping-patience 50 \
  --checkpoint-every 10 \
  --seed 42

run_experiment "eegnet_wd1e-4_seed42_ep300_lr9e-4_es50" \
  --epochs 300 \
  --batch-size 64 \
  --learning-rate 9e-4 \
  --weight-decay 1e-4 \
  --scheduler none \
  --early-stopping-patience 50 \
  --checkpoint-every 10 \
  --seed 42

run_experiment "eegnet_plateau_seed42_ep300_lr9e-4_es50" \
  --epochs 300 \
  --batch-size 64 \
  --learning-rate 9e-4 \
  --weight-decay 0 \
  --scheduler plateau \
  --plateau-factor 0.5 \
  --plateau-patience 10 \
  --early-stopping-patience 50 \
  --checkpoint-every 10 \
  --seed 42

echo
echo "======================================================================"
echo "Phase 1 summary"
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Completed (${#COMPLETED_RUNS[@]}): ${COMPLETED_RUNS[*]:-none}"
echo "Skipped (${#SKIPPED_RUNS[@]}): ${SKIPPED_RUNS[*]:-none}"
echo "Failed (${#FAILED_RUNS[@]}): ${FAILED_RUNS[*]:-none}"
echo "Results: ${RUN_ROOT}"
echo "Global log: ${LOG_FILE}"
echo "Summary CSV: ${RUN_ROOT}/baseline_runs_summary.csv"
echo "======================================================================"

if [[ "${#FAILED_RUNS[@]}" -gt 0 ]]; then
  exit 1
fi

exit 0
