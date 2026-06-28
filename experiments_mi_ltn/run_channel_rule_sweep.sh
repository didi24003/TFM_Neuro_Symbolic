#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_ROOT="${REPO_ROOT}/experiments_mi_ltn/runs/eegnet_channel_rule_loss"
LOG_DIR="${RUN_ROOT}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_DIR}/channel_rule_sweep_${TIMESTAMP}.log"
DATA_ROOT="${REPO_ROOT}/data/BCICIV_2a_mat"
TRAIN_SCRIPT="${REPO_ROOT}/experiments_mi_ltn/03_train_eegnet_channel_rule_loss.py"
SUMMARY_SCRIPT="${REPO_ROOT}/experiments_mi_ltn/summarize_channel_rule_lambdas.py"
SUMMARY_CSV="${RUN_ROOT}/channel_rule_lambdas_summary.csv"
SUMMARY_TEX="${RUN_ROOT}/channel_rule_lambdas_latex_table.txt"
BEST_EXPORT_DIR="${RUN_ROOT}/best_export"
CHANNEL_ORDER_FILE="${REPO_ROOT}/experiments_mi_ltn/bciciv2a_channel_order.txt"
LAMBDAS=(0.0 0.01 0.05 0.1 0.2 0.5 1.0)

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting channel-rule sweep"
echo "repo_root=${REPO_ROOT}"
echo "log_path=${LOG_PATH}"

if [[ ! -e "${REPO_ROOT}/.git" ]]; then
  echo "ERROR: ${REPO_ROOT} does not look like the repository root (.git not found)."
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

if [[ ! -f "${CHANNEL_ORDER_FILE}" ]]; then
  echo "ERROR: channel order file not found: ${CHANNEL_ORDER_FILE}"
  exit 1
fi

cd "${REPO_ROOT}"

echo
echo "=== git status --short ==="
git status --short || true

if git status --short | grep -q .; then
  echo "WARNING: repository has pre-existing changes. Sweep will continue."
fi
if git status --short | grep -Fq "experiments_mi_ltn/runs/final_channel_analysis/final_channel_analysis.log"; then
  echo "WARNING: detected unrelated modified file: experiments_mi_ltn/runs/final_channel_analysis/final_channel_analysis.log"
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "ERROR: dataset directory not found: ${DATA_ROOT}"
  exit 1
fi
echo "dataset_root=${DATA_ROOT}"

echo
echo "=== dry-run ==="
python "${TRAIN_SCRIPT}" \
  --data-root "${DATA_ROOT}" \
  --seed 2024 \
  --epochs 1 \
  --batch-size 64 \
  --learning-rate 5e-4 \
  --weight-decay 0.0 \
  --scheduler none \
  --early-stopping-patience 50 \
  --checkpoint-every 10 \
  --lambda-rule 0.0 \
  --dry-run

echo
echo "=== sweep ==="
for lambda_rule in "${LAMBDAS[@]}"; do
  echo ">>> Running lambda_rule=${lambda_rule}"
  python "${TRAIN_SCRIPT}" \
    --data-root "${DATA_ROOT}" \
    --seed 2024 \
    --epochs 300 \
    --batch-size 64 \
    --learning-rate 5e-4 \
    --weight-decay 0.0 \
    --scheduler none \
    --early-stopping-patience 50 \
    --checkpoint-every 10 \
    --lambda-rule "${lambda_rule}"
done

echo
echo "=== summarize ==="
python "${SUMMARY_SCRIPT}"

if [[ ! -f "${SUMMARY_CSV}" ]]; then
  echo "ERROR: summary CSV not found: ${SUMMARY_CSV}"
  exit 1
fi
if [[ ! -f "${SUMMARY_TEX}" ]]; then
  echo "ERROR: summary LaTeX table not found: ${SUMMARY_TEX}"
  exit 1
fi

mkdir -p "${BEST_EXPORT_DIR}"
cp "${SUMMARY_CSV}" "${BEST_EXPORT_DIR}/"
cp "${SUMMARY_TEX}" "${BEST_EXPORT_DIR}/"
cp "${LOG_PATH}" "${BEST_EXPORT_DIR}/"

BEST_INFO="$(
python - "${SUMMARY_CSV}" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_csv = Path(sys.argv[1])
rows = []
with summary_csv.open(newline="") as f:
    for row in csv.DictReader(f):
        if row.get("status") in {"pending", "failed"}:
            continue
        if row.get("best_val_acc") in {"", None}:
            continue
        rows.append(row)

if not rows:
    print(json.dumps({"error": "No completed rows with best_val_acc found."}))
    raise SystemExit(0)

best = max(rows, key=lambda row: float(row["best_val_acc"]))
run_dir = summary_csv.parent / best["run_id"]
summary_path = run_dir / "summary.json"
payload = {"run_id": best["run_id"], "run_dir": str(run_dir)}
if summary_path.exists():
    payload.update(json.loads(summary_path.read_text()))
print(json.dumps(payload))
PY
)"

if [[ "${BEST_INFO}" == *'"error"'* ]]; then
  echo "WARNING: could not determine the best run automatically."
  echo "${BEST_INFO}"
  echo "Available run directories:"
  find "${RUN_ROOT}" -maxdepth 1 -mindepth 1 -type d | sort
  exit 0
fi

BEST_RUN_DIR="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["run_dir"])' "${BEST_INFO}")"
BEST_RUN_ID="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["run_id"])' "${BEST_INFO}")"
BEST_LAMBDA="$(python -c 'import json,sys; print(json.loads(sys.argv[1]).get("lambda_rule", ""))' "${BEST_INFO}")"
BEST_VAL_ACC="$(python -c 'import json,sys; print(json.loads(sys.argv[1]).get("best_val_acc", ""))' "${BEST_INFO}")"
BEST_EPOCH="$(python -c 'import json,sys; print(json.loads(sys.argv[1]).get("best_epoch", ""))' "${BEST_INFO}")"
FINAL_TRAIN_ACC="$(python -c 'import json,sys; print(json.loads(sys.argv[1]).get("final_train_acc", ""))' "${BEST_INFO}")"
FINAL_VAL_ACC="$(python -c 'import json,sys; print(json.loads(sys.argv[1]).get("final_val_acc", ""))' "${BEST_INFO}")"
R_SM="$(python -c 'import json,sys; data=json.loads(sys.argv[1]); print(data.get("R_SM", data.get("final_r_sm", "")))' "${BEST_INFO}")"
R_POST="$(python -c 'import json,sys; data=json.loads(sys.argv[1]); print(data.get("R_POST", data.get("final_r_post", "")))' "${BEST_INFO}")"
R_SM_MINUS_POST="$(python -c 'import json,sys; data=json.loads(sys.argv[1]); print(data.get("R_SM_minus_POST", data.get("final_r_diff", "")))' "${BEST_INFO}")"

for path in \
  "${BEST_RUN_DIR}/figures/training_curves.png" \
  "${BEST_RUN_DIR}/figures/channel_gates.png" \
  "${BEST_RUN_DIR}/channel_gates.csv" \
  "${BEST_RUN_DIR}/summary.json"; do
  if [[ -f "${path}" ]]; then
    cp "${path}" "${BEST_EXPORT_DIR}/"
  else
    echo "WARNING: expected export artifact not found: ${path}"
  fi
done

echo
echo "=== final summary ==="
echo "best_run_id=${BEST_RUN_ID}"
echo "best_lambda_rule=${BEST_LAMBDA}"
echo "best_val_acc=${BEST_VAL_ACC}"
echo "best_epoch=${BEST_EPOCH}"
echo "final_train_acc=${FINAL_TRAIN_ACC}"
echo "final_val_acc=${FINAL_VAL_ACC}"
echo "R_SM=${R_SM}"
echo "R_POST=${R_POST}"
echo "R_SM_minus_POST=${R_SM_MINUS_POST}"
echo "best_export=${BEST_EXPORT_DIR}"
