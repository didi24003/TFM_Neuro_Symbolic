#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASELINE_OUT_DIR="${BASELINE_OUT_DIR:-${REPO_ROOT}/experiments_mi_ltn/runs/eegnet_baseline_subject_session_artifact_free}"
LOGIC_OUT_DIR="${LOGIC_OUT_DIR:-${REPO_ROOT}/experiments_mi_ltn/runs/eegnet_logic_subject_session_artifact_free}"
EXPORT_ROOT="${EXPORT_ROOT:-${REPO_ROOT}/experiments_mi_ltn/artifact_free/_export_tmp}"
EXPORT_ZIP="${EXPORT_ZIP:-${REPO_ROOT}/experiments_mi_ltn/artifact_free/artifact_free_export.zip}"

if [[ -e "${EXPORT_ROOT}" ]]; then
  echo "Temporary export directory already exists: ${EXPORT_ROOT}" >&2
  exit 1
fi

rm -f "${EXPORT_ZIP}"
mkdir -p "${EXPORT_ROOT}/baseline" "${EXPORT_ROOT}/logic"

copy_lightweight() {
  local src="$1"
  local dst="$2"
  if [[ ! -d "${src}" ]]; then
    echo "Skip missing directory: ${src}"
    return 0
  fi

  while IFS= read -r path; do
    local rel="${path#${src}/}"
    mkdir -p "${dst}/$(dirname "${rel}")"
    cp "${path}" "${dst}/${rel}"
  done < <(find "${src}" -type f \( -name "*.csv" -o -name "*.json" -o -name "*.txt" \))

  if [[ -d "${src}/analysis_figures" ]]; then
    mkdir -p "${dst}"
    cp -R "${src}/analysis_figures" "${dst}/analysis_figures"
  fi
}

copy_lightweight "${BASELINE_OUT_DIR}" "${EXPORT_ROOT}/baseline"
copy_lightweight "${LOGIC_OUT_DIR}" "${EXPORT_ROOT}/logic"

(
  cd "${EXPORT_ROOT}"
  zip -rq "${EXPORT_ZIP}" .
)

rm -rf "${EXPORT_ROOT}"
echo "export_zip=${EXPORT_ZIP}"
