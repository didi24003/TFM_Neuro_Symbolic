#!/usr/bin/env bash

set -euo pipefail

RUN_ROOT="experiments_mi_ltn/runs/logic_loss_eegnex"
RUN_ID="eegnex_logic_lam1p0_seed2024_lr5e-4_ep300_es50"
TRAIN_SCRIPT="experiments_mi_ltn/04_logic_loss_eegnet.py"
DATA_DIR="data/BCICIV_2a_mat"
LOG_FILE="${RUN_ROOT}/eegnex_logic_lam1p0_seed2024.log"
PYTHON_BIN="${PYTHON:-python}"
FORCE="${FORCE:-0}"

if [[ ! -f "setup.py" || ! -d "torcheeg" ]]; then
  echo "ERROR: run this script from the repository root."
  echo "Expected: ./experiments_mi_ltn/run_eegnex_logic_lam1p0_seed2024.sh"
  exit 1
fi

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "ERROR: training script not found: ${TRAIN_SCRIPT}"
  exit 1
fi

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "ERROR: dataset folder not found: ${DATA_DIR}"
  exit 1
fi

mkdir -p "${RUN_ROOT}"
touch "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "======================================================================"
echo "EEGNeX logic loss lambda=1.0 seed 2024"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Repository: $(pwd)"
echo "Python: ${PYTHON_BIN}"
echo "Log: ${LOG_FILE}"
echo "FORCE=${FORCE}"
echo "======================================================================"

MAT_COUNT="$(find "${DATA_DIR}" -maxdepth 1 -type f -name '*.mat' | wc -l)"
if [[ "${MAT_COUNT}" -lt 18 ]]; then
  echo "WARNING: expected 18 .mat files in ${DATA_DIR}, found ${MAT_COUNT}."
fi

RUN_DIR="${RUN_ROOT}/${RUN_ID}"
ACTUAL_RUN_ID="${RUN_ID}"
if [[ -e "${RUN_DIR}" && "${FORCE}" != "1" ]]; then
  echo "ERROR: run directory already exists: ${RUN_DIR}"
  echo "Use FORCE=1 to allow a repeated run without overwriting old results."
  exit 1
fi

if [[ -e "${RUN_DIR}" && "${FORCE}" == "1" ]]; then
  ACTUAL_RUN_ID="${RUN_ID}_force_$(date '+%Y%m%d_%H%M%S')"
  echo "FORCE=1 detected. Repeated run will be saved as ${ACTUAL_RUN_ID}."
fi

echo
echo "Checking Braindecode and EEGNeX import"
"${PYTHON_BIN}" - <<'PY'
import braindecode
from braindecode.models import EEGNeX

print(f"braindecode.__version__={braindecode.__version__}")
print(f"EEGNeX_import_ok={EEGNeX.__name__}")
PY

echo
echo "Checking CUDA"
"${PYTHON_BIN}" - <<'PY'
import sys
import torch

print(f"torch.__version__={torch.__version__}")
print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("ERROR: CUDA is required for this long EEGNeX run.")
    sys.exit(1)

print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
for idx in range(torch.cuda.device_count()):
    print(f"gpu[{idx}]={torch.cuda.get_device_name(idx)}")
PY

echo
echo "Launching training"
"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --model eegnex \
  --data-root "${DATA_DIR}" \
  --epochs 300 \
  --batch-size 64 \
  --learning-rate 5e-4 \
  --weight-decay 0 \
  --seed 2024 \
  --scheduler none \
  --early-stopping-patience 50 \
  --checkpoint-every 10 \
  --lambda-logic 1.0 \
  --run-root "${RUN_ROOT}" \
  --run-id "${ACTUAL_RUN_ID}"

echo
echo "Completed: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Run dir: ${RUN_ROOT}/${ACTUAL_RUN_ID}"
