#!/usr/bin/env bash

set -euo pipefail

RUN_ROOT="experiments_mi_ltn/runs/baseline_eegnet"
ARCHIVE_ROOT="experiments_mi_ltn/runs/archive"
LOG_FILE="${ARCHIVE_ROOT}/archive_baseline_without_checkpoints.log"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
DEST_DIR="${ARCHIVE_ROOT}/baseline_without_checkpoints_${TIMESTAMP}"
CONFIRM="${CONFIRM:-0}"

if [[ ! -d "${RUN_ROOT}" ]]; then
  echo "ERROR: run root not found: ${RUN_ROOT}"
  exit 1
fi

mkdir -p "${ARCHIVE_ROOT}"
touch "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

mapfile -t CANDIDATES < <(
  find "${RUN_ROOT}" -mindepth 1 -maxdepth 1 -type d ! -name figures ! -name checkpoints | sort | while read -r run_dir; do
    if [[ ! -f "${run_dir}/checkpoint_best.pt" ]]; then
      printf '%s\n' "${run_dir}"
    fi
  done
)

echo "======================================================================"
echo "Archive baseline runs without checkpoint_best.pt"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Run root: ${RUN_ROOT}"
echo "Archive dir: ${DEST_DIR}"
echo "Log: ${LOG_FILE}"
echo "======================================================================"

if [[ "${#CANDIDATES[@]}" -eq 0 ]]; then
  echo
  echo "No baseline run directories without checkpoint_best.pt were found."
  exit 0
fi

echo
echo "The following directories will be moved:"
for run_dir in "${CANDIDATES[@]}"; do
  echo "  - ${run_dir}"
done

echo
echo "Total candidates: ${#CANDIDATES[@]}"

if [[ "${CONFIRM}" != "1" ]]; then
  echo
  read -r -p "Type MOVE to archive these directories, or anything else to cancel: " reply
  if [[ "${reply}" != "MOVE" ]]; then
    echo "Cancelled. No directories were moved."
    exit 0
  fi
fi

mkdir -p "${DEST_DIR}"

echo
echo "Moving directories..."
for run_dir in "${CANDIDATES[@]}"; do
  base_name="$(basename "${run_dir}")"
  echo "  mv ${run_dir} ${DEST_DIR}/${base_name}"
  mv "${run_dir}" "${DEST_DIR}/${base_name}"
done

echo
echo "Archive completed successfully."
echo "Moved ${#CANDIDATES[@]} directories to ${DEST_DIR}"
