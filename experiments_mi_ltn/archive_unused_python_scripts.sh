#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="experiments_mi_ltn"
ARCHIVE_DIR="${SCRIPT_DIR}/archive_scripts"
LOG_FILE="${ARCHIVE_DIR}/archive_unused_python_scripts.log"
CONFIRM="${CONFIRM:-0}"

CANDIDATES=(
  "05_plot_channel_importance.py"
  "05_plot_channel_importance_pretty.py"
  "07_sensorimotor_coherence_score.py"
  "12_update_baseline_summary.py"
)

if [[ ! -d "${SCRIPT_DIR}" ]]; then
  echo "ERROR: missing directory ${SCRIPT_DIR}"
  exit 1
fi

mkdir -p "${ARCHIVE_DIR}"
touch "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "======================================================================"
echo "Archive unused Python scripts"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Script dir: ${SCRIPT_DIR}"
echo "Archive dir: ${ARCHIVE_DIR}"
echo "Log: ${LOG_FILE}"
echo "======================================================================"

EXISTING=()
MISSING=()

for name in "${CANDIDATES[@]}"; do
  path="${SCRIPT_DIR}/${name}"
  if [[ -f "${path}" ]]; then
    EXISTING+=("${path}")
  else
    MISSING+=("${path}")
  fi
done

if [[ "${#EXISTING[@]}" -eq 0 ]]; then
  echo
  echo "No candidate scripts found. Nothing to move."
  exit 0
fi

echo
echo "Proposed scripts to move:"
for path in "${EXISTING[@]}"; do
  echo "  - ${path}"
done

if [[ "${#MISSING[@]}" -gt 0 ]]; then
  echo
  echo "Already absent:"
  for path in "${MISSING[@]}"; do
    echo "  - ${path}"
  done
fi

echo
echo "These scripts are intentionally excluded from this archive step because they are needed by the current pipeline or helper scripts:"
echo "  - experiments_mi_ltn/01_check_dataset.py"
echo "  - experiments_mi_ltn/02_train_eegnet_bciciv2a.py"
echo "  - experiments_mi_ltn/03_channel_importance.py"
echo "  - experiments_mi_ltn/04_logic_loss_eegnet.py"
echo "  - experiments_mi_ltn/05_plot_channel_importance_comparison.py"
echo "  - experiments_mi_ltn/08_finalize_channel_analysis.py"
echo "  - experiments_mi_ltn/10_plot_topomap_channel_importance.py"
echo "  - experiments_mi_ltn/mi_ltn_common.py"
echo "  - experiments_mi_ltn/summarize_baseline_best_seeds.py"
echo "  - experiments_mi_ltn/summarize_logic_loss_lambdas.py"

if [[ "${CONFIRM}" != "1" ]]; then
  echo
  echo "Dry run only. Re-run with CONFIRM=1 to move these files."
  exit 0
fi

echo
echo "Moving scripts..."
for path in "${EXISTING[@]}"; do
  dest="${ARCHIVE_DIR}/$(basename "${path}")"
  echo "  mv ${path} ${dest}"
  mv "${path}" "${dest}"
done

echo
echo "Archive completed successfully."
