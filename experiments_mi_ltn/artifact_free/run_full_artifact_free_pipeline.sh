#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VERIFY_SCRIPT="${SCRIPT_DIR}/verify_artifact_filter.py"
BASELINE_SCRIPT="${SCRIPT_DIR}/run_baseline_artifact_free.sh"
LOGIC_SCRIPT="${SCRIPT_DIR}/run_logic_artifact_free.sh"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/BCICIV_2a_mat}"
SEEDS="${SEEDS:-0 7 42 123 2024}"
BASELINE_OUT_DIR="${BASELINE_OUT_DIR:-${REPO_ROOT}/experiments_mi_ltn/runs/eegnet_baseline_subject_session_artifact_free}"
LOGIC_OUT_DIR="${LOGIC_OUT_DIR:-${REPO_ROOT}/experiments_mi_ltn/runs/eegnet_logic_subject_session_artifact_free}"

DRY_RUN=false
SUMMARY_ONLY=false

for arg in "$@"; do
  case "${arg}" in
    --dry-run)
      DRY_RUN=true
      ;;
    --summary-only)
      SUMMARY_ONLY=true
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

if [[ "${SUMMARY_ONLY}" == "false" && "${DRY_RUN}" == "false" ]]; then
  python "${VERIFY_SCRIPT}"
elif [[ "${SUMMARY_ONLY}" == "false" && "${DRY_RUN}" == "true" ]]; then
  echo "[dry-run] python ${VERIFY_SCRIPT}"
fi

if [[ "${SUMMARY_ONLY}" == "false" ]]; then
  baseline_args=()
  logic_args=()
  if [[ "${DRY_RUN}" == "true" ]]; then
    baseline_args+=(--dry-run)
    logic_args+=(--dry-run)
  fi
  env DATA_ROOT="${DATA_ROOT}" SEEDS="${SEEDS}" OUT_DIR="${BASELINE_OUT_DIR}" bash "${BASELINE_SCRIPT}" "${baseline_args[@]}"
  env DATA_ROOT="${DATA_ROOT}" SEEDS="${SEEDS}" BASELINE_OUT_DIR="${BASELINE_OUT_DIR}" OUT_DIR="${LOGIC_OUT_DIR}" bash "${LOGIC_SCRIPT}" "${logic_args[@]}"
fi

summary_args=(--summary-only)
if [[ "${DRY_RUN}" == "true" ]]; then
  summary_args=(--dry-run)
fi

env DATA_ROOT="${DATA_ROOT}" SEEDS="${SEEDS}" OUT_DIR="${BASELINE_OUT_DIR}" bash "${BASELINE_SCRIPT}" "${summary_args[@]}"
env DATA_ROOT="${DATA_ROOT}" SEEDS="${SEEDS}" BASELINE_OUT_DIR="${BASELINE_OUT_DIR}" OUT_DIR="${LOGIC_OUT_DIR}" bash "${LOGIC_SCRIPT}" "${summary_args[@]}"
