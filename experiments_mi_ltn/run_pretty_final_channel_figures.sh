#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ ! -f "${REPO_ROOT}/experiments_mi_ltn/11_plot_final_channel_analysis_pretty.py" ]]; then
  echo "Error: no se encontró experiments_mi_ltn/11_plot_final_channel_analysis_pretty.py" >&2
  exit 1
fi

cd "${REPO_ROOT}"

ANALYSIS_DIR="${1:-experiments_mi_ltn/runs/final_channel_analysis}"
OUTPUT_DIR="${ANALYSIS_DIR}/pretty_figures"
LOG_FILE="${OUTPUT_DIR}/pretty_figures.log"

mkdir -p "${OUTPUT_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "== Pretty final channel figures =="
echo "Repository root: ${REPO_ROOT}"
echo "Input dir: ${ANALYSIS_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Log file: ${LOG_FILE}"

required_files=(
  "${ANALYSIS_DIR}/baseline/channel_importance.csv"
  "${ANALYSIS_DIR}/logic_lam1p0/channel_importance.csv"
  "${ANALYSIS_DIR}/comparison/channel_importance_delta.csv"
  "${ANALYSIS_DIR}/comparison/coherence_scores.csv"
)

for path in "${required_files[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Error: falta el CSV final requerido: ${path}" >&2
    exit 1
  fi
done

echo "Todos los CSV finales requeridos existen."
echo "No se recalculará permutation importance."
echo "No se usarán checkpoints."
echo "No se usarán archivos antiguos."

python experiments_mi_ltn/11_plot_final_channel_analysis_pretty.py \
  --analysis-dir "${ANALYSIS_DIR}" \
  --output-dir "${OUTPUT_DIR}"

echo "Figuras bonitas generadas en ${OUTPUT_DIR}"
