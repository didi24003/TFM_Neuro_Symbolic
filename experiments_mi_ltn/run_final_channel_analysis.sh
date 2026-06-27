#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ ! -f "${REPO_ROOT}/experiments_mi_ltn/mi_ltn_common.py" ]]; then
  echo "Error: este script debe ejecutarse desde la raíz del repositorio o desde una copia que contenga experiments_mi_ltn/mi_ltn_common.py." >&2
  exit 1
fi

cd "${REPO_ROOT}"

DATASET_DIR="data/BCICIV_2a_mat"
BASE_OUTPUT_DIR="experiments_mi_ltn/runs/final_channel_analysis"
ARCHIVE_DIR="experiments_mi_ltn/runs/archive"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${BASE_OUTPUT_DIR}"
ARCHIVED_FINAL_DIR="${ARCHIVE_DIR}/final_channel_analysis_old_${TIMESTAMP}"

if [[ -d "${BASE_OUTPUT_DIR}" ]]; then
  mkdir -p "${ARCHIVE_DIR}"
  mv "${BASE_OUTPUT_DIR}" "${ARCHIVED_FINAL_DIR}"
  echo "Aviso: se archivó la salida previa en ${ARCHIVED_FINAL_DIR}"
fi

mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/final_channel_analysis.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "== Final channel analysis =="
echo "Repository root: ${REPO_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Log file: ${LOG_FILE}"

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Error: no existe el dataset en ${DATASET_DIR}" >&2
  exit 1
fi

if ! find "${DATASET_DIR}" -maxdepth 1 -type f | grep -q .; then
  echo "Error: la carpeta del dataset existe pero no contiene archivos visibles en ${DATASET_DIR}" >&2
  exit 1
fi

CUDA_STATUS="$(python - <<'PY'
import torch
print("available" if torch.cuda.is_available() else "not_available")
PY
)"
echo "CUDA: ${CUDA_STATUS}"
if [[ "${CUDA_STATUS}" == "not_available" ]]; then
  echo "Aviso: CUDA no está disponible. El análisis se ejecutará en CPU si continúan los archivos necesarios."
fi

BASELINE_RUN_DIR="experiments_mi_ltn/runs/baseline_eegnet/eegnet_best_lr5e-4_seed2024_ep300_es50"
BASELINE_SUMMARY="${BASELINE_RUN_DIR}/summary.json"
BASELINE_CHECKPOINT="${BASELINE_RUN_DIR}/checkpoint_best.pt"
LOGIC_RUN_DIR="experiments_mi_ltn/runs/logic_loss_eegnet/eegnet_logic_lam1p0_seed2024_lr5e-4_ep300_es50"
LOGIC_SUMMARY="${LOGIC_RUN_DIR}/summary.json"
LOGIC_CHECKPOINT="${LOGIC_RUN_DIR}/checkpoint_best.pt"

if [[ ! -f "${BASELINE_SUMMARY}" ]]; then
  echo "Error: falta summary.json del baseline en ${BASELINE_SUMMARY}" >&2
  exit 1
fi

echo "Baseline run dir: ${BASELINE_RUN_DIR}"
echo "Logic run dir: ${LOGIC_RUN_DIR}"

if [[ ! -f "${BASELINE_CHECKPOINT}" ]]; then
  echo "Error: falta el checkpoint baseline definitivo en ${BASELINE_CHECKPOINT}" >&2
  echo "Proporcione la carpeta del run baseline con checkpoint_best.pt en ${BASELINE_RUN_DIR}" >&2
  exit 1
fi

if [[ ! -f "${LOGIC_SUMMARY}" ]]; then
  echo "Error: falta summary.json del run logic loss en ${LOGIC_SUMMARY}" >&2
  exit 1
fi

if [[ ! -f "${LOGIC_CHECKPOINT}" ]]; then
  echo "Error: falta el checkpoint logic loss definitivo en ${LOGIC_CHECKPOINT}" >&2
  echo "Proporcione la carpeta del run logic loss con checkpoint_best.pt en ${LOGIC_RUN_DIR}" >&2
  exit 1
fi

mkdir -p \
  "${OUTPUT_DIR}/baseline" \
  "${OUTPUT_DIR}/logic_lam1p0" \
  "${OUTPUT_DIR}/comparison"

DEVICE_ARG="cpu"
if [[ "${CUDA_STATUS}" == "available" ]]; then
  DEVICE_ARG="cuda"
fi

echo
echo "1. Permutation importance baseline"
python experiments_mi_ltn/03_channel_importance.py \
  --data-root "${DATASET_DIR}" \
  --checkpoint "${BASELINE_CHECKPOINT}" \
  --output-csv "${OUTPUT_DIR}/baseline/channel_importance.csv" \
  --output-sorted-csv "${OUTPUT_DIR}/baseline/channel_importance_sorted.csv" \
  --output-barplot "${OUTPUT_DIR}/baseline/channel_importance_barplot.png" \
  --plot-title "EEGNet baseline - permutation importance" \
  --seed 2024 \
  --device "${DEVICE_ARG}"

echo
echo "2. Permutation importance logic loss"
python experiments_mi_ltn/03_channel_importance.py \
  --data-root "${DATASET_DIR}" \
  --checkpoint "${LOGIC_CHECKPOINT}" \
  --output-csv "${OUTPUT_DIR}/logic_lam1p0/channel_importance.csv" \
  --output-sorted-csv "${OUTPUT_DIR}/logic_lam1p0/channel_importance_sorted.csv" \
  --output-barplot "${OUTPUT_DIR}/logic_lam1p0/channel_importance_barplot.png" \
  --plot-title "EEGNet + logic loss (lambda_logic=1.0) - permutation importance" \
  --seed 2024 \
  --device "${DEVICE_ARG}"

echo
echo "3. Comparación baseline vs logic"
python experiments_mi_ltn/05_plot_channel_importance_comparison.py \
  --baseline-csv "${OUTPUT_DIR}/baseline/channel_importance.csv" \
  --logic-csv "${OUTPUT_DIR}/logic_lam1p0/channel_importance.csv" \
  --output-csv "${OUTPUT_DIR}/comparison/channel_importance_comparison.csv" \
  --comparison-png "${OUTPUT_DIR}/comparison/channel_importance_comparison.png"

echo
echo "4-6. Delta por canal, ratios de coherencia y resumen final"
python experiments_mi_ltn/08_finalize_channel_analysis.py \
  --baseline-csv "${OUTPUT_DIR}/baseline/channel_importance.csv" \
  --logic-csv "${OUTPUT_DIR}/logic_lam1p0/channel_importance.csv" \
  --baseline-summary "${BASELINE_SUMMARY}" \
  --logic-summary "${LOGIC_SUMMARY}" \
  --comparison-csv "${OUTPUT_DIR}/comparison/channel_importance_comparison.csv" \
  --delta-csv "${OUTPUT_DIR}/comparison/channel_importance_delta.csv" \
  --delta-png "${OUTPUT_DIR}/comparison/channel_importance_delta.png" \
  --coherence-csv "${OUTPUT_DIR}/comparison/coherence_scores.csv" \
  --coherence-png "${OUTPUT_DIR}/comparison/coherence_scores.png" \
  --summary-json "${OUTPUT_DIR}/comparison/final_channel_analysis_summary.json" \
  --latex-table "${OUTPUT_DIR}/comparison/coherence_scores_table.tex"

echo
echo "7. Topomap baseline"
python experiments_mi_ltn/10_plot_topomap_channel_importance.py \
  --input-csv "${OUTPUT_DIR}/baseline/channel_importance.csv" \
  --title "EEGNet baseline - Channel importance" \
  --output-path "${OUTPUT_DIR}/baseline/topomap_channel_importance.png"

echo
echo "8. Topomap logic"
python experiments_mi_ltn/10_plot_topomap_channel_importance.py \
  --input-csv "${OUTPUT_DIR}/logic_lam1p0/channel_importance.csv" \
  --title "EEGNet + logic loss (lambda_logic=1.0) - Channel importance" \
  --output-path "${OUTPUT_DIR}/logic_lam1p0/topomap_channel_importance.png"

echo
echo "9. Importancia por canal y por clase"
python experiments_mi_ltn/12_channel_importance_by_class_final.py \
  --data-root "${DATASET_DIR}" \
  --baseline-checkpoint "${BASELINE_CHECKPOINT}" \
  --logic-checkpoint "${LOGIC_CHECKPOINT}" \
  --baseline-output-csv "${OUTPUT_DIR}/baseline/channel_importance_by_class.csv" \
  --logic-output-csv "${OUTPUT_DIR}/logic_lam1p0/channel_importance_by_class.csv" \
  --comparison-output-csv "${OUTPUT_DIR}/comparison/channel_importance_by_class_comparison.csv" \
  --seed 2024 \
  --device "${DEVICE_ARG}"

echo
echo "Archivos CSV generados:"
echo "  ${OUTPUT_DIR}/baseline/channel_importance.csv"
echo "  ${OUTPUT_DIR}/baseline/channel_importance_sorted.csv"
echo "  ${OUTPUT_DIR}/baseline/channel_importance_by_class.csv"
echo "  ${OUTPUT_DIR}/logic_lam1p0/channel_importance.csv"
echo "  ${OUTPUT_DIR}/logic_lam1p0/channel_importance_sorted.csv"
echo "  ${OUTPUT_DIR}/logic_lam1p0/channel_importance_by_class.csv"
echo "  ${OUTPUT_DIR}/comparison/channel_importance_comparison.csv"
echo "  ${OUTPUT_DIR}/comparison/channel_importance_by_class_comparison.csv"
echo "  ${OUTPUT_DIR}/comparison/channel_importance_delta.csv"
echo "  ${OUTPUT_DIR}/comparison/coherence_scores.csv"

echo "Figuras generadas:"
echo "  ${OUTPUT_DIR}/baseline/channel_importance_barplot.png"
echo "  ${OUTPUT_DIR}/baseline/topomap_channel_importance.png"
echo "  ${OUTPUT_DIR}/logic_lam1p0/channel_importance_barplot.png"
echo "  ${OUTPUT_DIR}/logic_lam1p0/topomap_channel_importance.png"
echo "  ${OUTPUT_DIR}/comparison/channel_importance_comparison.png"
echo "  ${OUTPUT_DIR}/comparison/channel_importance_delta.png"
echo "  ${OUTPUT_DIR}/comparison/coherence_scores.png"

echo "Tabla LaTeX:"
echo "  ${OUTPUT_DIR}/comparison/coherence_scores_table.tex"

python - <<'PY' "${OUTPUT_DIR}/comparison/coherence_scores.csv"
import pandas as pd
import sys
path = sys.argv[1]
df = pd.read_csv(path)
print("Resumen de ratios:")
for _, row in df.iterrows():
    print(
        f"  {row['model']}: sensorimotor_ratio={row['sensorimotor_ratio']:.4f}, "
        f"posterior_ratio={row['posterior_ratio']:.4f}"
    )
PY

echo
echo "10. Figuras bonitas opcionales"
if [[ -f "experiments_mi_ltn/run_pretty_final_channel_figures.sh" ]]; then
  if bash experiments_mi_ltn/run_pretty_final_channel_figures.sh "${OUTPUT_DIR}"; then
    echo "Figuras bonitas generadas correctamente."
  else
    echo "Aviso: falló la generación de figuras bonitas; se conservan todos los resultados previos." >&2
  fi
else
  echo "Aviso: no se encontró experiments_mi_ltn/run_pretty_final_channel_figures.sh; se omite este paso."
fi

echo "Análisis final completado."
