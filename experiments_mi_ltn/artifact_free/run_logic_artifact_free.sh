#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PIPELINE_SCRIPT="${REPO_ROOT}/experiments_mi_ltn/run_eegnet_logic_subject_session_full_pipeline.sh"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/BCICIV_2a_mat}"
SEEDS="${SEEDS:-0 7 42 123 2024}"
BASELINE_OUT_DIR="${BASELINE_OUT_DIR:-${REPO_ROOT}/experiments_mi_ltn/runs/eegnet_baseline_subject_session_artifact_free}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/experiments_mi_ltn/runs/eegnet_logic_subject_session_artifact_free}"
SKIP_ARTIFACTS=1

args=()
for arg in "$@"; do
  case "${arg}" in
    --dry-run|--summary-only)
      args+=("${arg}")
      ;;
    --overwrite)
      echo "--overwrite no esta implementado de forma segura en este wrapper." >&2
      exit 1
      ;;
    *)
      echo "Unknown argument: ${arg}" >&2
      exit 1
      ;;
  esac
done

if [[ " ${args[*]} " != *" --summary-only "* && " ${args[*]} " != *" --dry-run "* && -e "${OUT_DIR}" ]]; then
  echo "Output directory already exists: ${OUT_DIR}" >&2
  echo "No se sobrescriben runs existentes. Usa --summary-only o borra/renombra la carpeta manualmente." >&2
  exit 1
fi

env \
  DATA_ROOT="${DATA_ROOT}" \
  SEEDS="${SEEDS}" \
  BASELINE_RUN_ROOT="${BASELINE_OUT_DIR}" \
  OUT_DIR="${OUT_DIR}" \
  SKIP_ARTIFACTS="${SKIP_ARTIFACTS}" \
  bash "${PIPELINE_SCRIPT}" "${args[@]}"
