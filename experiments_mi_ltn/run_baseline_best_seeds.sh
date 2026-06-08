#!/usr/bin/env bash

set -u
set -o pipefail

RUN_ROOT="experiments_mi_ltn/runs/baseline_eegnet"
LOG_FILE="${RUN_ROOT}/best_seeds_run.log"
TRAIN_SCRIPT="experiments_mi_ltn/02_train_eegnet_bciciv2a.py"
SUMMARY_SCRIPT="experiments_mi_ltn/summarize_baseline_best_seeds.py"
DATA_DIR="data/BCICIV_2a_mat"
PYTHON_BIN="${PYTHON:-python}"
FORCE="${FORCE:-0}"

EPOCHS=300
BATCH_SIZE=64
LEARNING_RATE="5e-4"
WEIGHT_DECAY="0"
SCHEDULER="none"
EARLY_STOPPING_PATIENCE=50
CHECKPOINT_EVERY=10

if [[ ! -f "setup.py" || ! -d "torcheeg" ]]; then
  echo "ERROR: run this script from the repository root."
  echo "Expected: ./experiments_mi_ltn/run_baseline_best_seeds.sh"
  exit 1
fi

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "ERROR: training script not found: ${TRAIN_SCRIPT}"
  exit 1
fi

if [[ ! -f "${SUMMARY_SCRIPT}" ]]; then
  echo "ERROR: summary script not found: ${SUMMARY_SCRIPT}"
  exit 1
fi

mkdir -p "${RUN_ROOT}"
touch "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "======================================================================"
echo "EEGNet baseline best configuration multi-seed runs"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Repository: $(pwd)"
echo "Python: ${PYTHON_BIN}"
echo "FORCE=${FORCE}"
echo "Configuration: epochs=${EPOCHS}, batch_size=${BATCH_SIZE}, lr=${LEARNING_RATE}, weight_decay=${WEIGHT_DECAY}, scheduler=${SCHEDULER}, early_stopping_patience=${EARLY_STOPPING_PATIENCE}"
echo "Log: ${LOG_FILE}"
echo "======================================================================"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo
  echo "ERROR: dataset folder not found: ${DATA_DIR}"
  echo "The BCI Competition IV 2a .mat files must be placed at:"
  echo "  ${DATA_DIR}/"
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
  local seed="$1"
  local run_id="eegnet_best_lr5e-4_seed${seed}_ep300_es50"
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
  echo "Seed: ${seed}"
  echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "----------------------------------------------------------------------"

  local start_seconds
  start_seconds="$(date +%s)"

  "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --learning-rate "${LEARNING_RATE}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --scheduler "${SCHEDULER}" \
    --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
    --checkpoint-every "${CHECKPOINT_EVERY}" \
    --seed "${seed}" \
    --run-id "${actual_run_id}"

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

run_experiment 0
run_experiment 7
run_experiment 123
run_experiment 2024

echo
echo "Updating best-seeds CSV and LaTeX summaries"
if ! "${PYTHON_BIN}" "${SUMMARY_SCRIPT}"; then
  echo "WARNING: failed to update best-seeds summaries."
fi

echo
echo "======================================================================"
echo "Best-seeds summary"
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Completed (${#COMPLETED_RUNS[@]}): ${COMPLETED_RUNS[*]:-none}"
echo "Skipped (${#SKIPPED_RUNS[@]}): ${SKIPPED_RUNS[*]:-none}"
echo "Failed (${#FAILED_RUNS[@]}): ${FAILED_RUNS[*]:-none}"
echo "Results: ${RUN_ROOT}"
echo "Global log: ${LOG_FILE}"
echo "Summary CSV: ${RUN_ROOT}/best_seeds_summary.csv"
echo "LaTeX table: ${RUN_ROOT}/best_seeds_results_table.tex"
echo "======================================================================"

if [[ "${#FAILED_RUNS[@]}" -gt 0 ]]; then
  exit 1
fi

exit 0
