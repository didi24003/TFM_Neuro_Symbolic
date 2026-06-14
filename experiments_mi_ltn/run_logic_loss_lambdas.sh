#!/usr/bin/env bash
set -u
set -o pipefail

SCRIPT_PATH="experiments_mi_ltn/04_logic_loss_eegnet.py"
DATA_ROOT="data/BCICIV_2a_mat"
RUN_ROOT="experiments_mi_ltn/runs/logic_loss_eegnet"
LOG_PATH="${RUN_ROOT}/logic_lambdas_run.log"
SUMMARY_CSV="${RUN_ROOT}/logic_loss_runs_summary.csv"

BASELINE_RUN_ID="eegnet_best_lr5e-4_seed2024_ep300_es50"
# Reference only: 04_logic_loss_eegnet.py evaluates this checkpoint on the same
# validation split for delta_vs_baseline, but trains the logic-loss EEGNet from scratch.
BASELINE_CHECKPOINT="experiments_mi_ltn/runs/baseline_eegnet/${BASELINE_RUN_ID}/checkpoint_best.pt"
BASELINE_BEST_VAL_ACC="0.7731660231660231"

SEED="2024"
EPOCHS="300"
BATCH_SIZE="64"
LEARNING_RATE="5e-4"
WEIGHT_DECAY="0.0"
SCHEDULER="none"
EARLY_STOPPING_PATIENCE="50"
CHECKPOINT_EVERY="10"

LAMBDAS=("0.001" "0.01" "0.05" "0.1" "0.2" "0.5" "1.0")

completed=()
skipped=()
failed=()

lambda_tag() {
    local value="$1"
    value="${value//./p}"
    echo "${value}"
}

log() {
    local msg="$1"
    printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${msg}" | tee -a "${LOG_PATH}"
}

if [[ ! -f "pyproject.toml" ]] || [[ ! -d "experiments_mi_ltn" ]]; then
    echo "ERROR: ejecuta este script desde la raiz del repositorio TorchEEG."
    exit 1
fi

if [[ ! -f "${SCRIPT_PATH}" ]]; then
    echo "ERROR: no existe ${SCRIPT_PATH}."
    exit 1
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "ERROR: no existe el dataset en ${DATA_ROOT}."
    exit 1
fi

if [[ ! -f "${BASELINE_CHECKPOINT}" ]]; then
    echo "WARNING: no existe el checkpoint baseline esperado: ${BASELINE_CHECKPOINT}"
fi

mkdir -p "${RUN_ROOT}"
: > "${LOG_PATH}"

log "Inicio de barrido EEGNet + logic loss"
log "Baseline elegido: ${BASELINE_RUN_ID} best_val_acc=${BASELINE_BEST_VAL_ACC}"
log "Baseline checkpoint de referencia, no inicializacion: ${BASELINE_CHECKPOINT}"
log "Config: seed=${SEED} epochs=${EPOCHS} bs=${BATCH_SIZE} lr=${LEARNING_RATE} wd=${WEIGHT_DECAY} scheduler=${SCHEDULER} es=${EARLY_STOPPING_PATIENCE}"

python - <<'PY' 2>&1 | tee -a "${LOG_PATH}"
import torch

if torch.cuda.is_available():
    index = torch.cuda.current_device()
    print(f"CUDA disponible: {torch.cuda.get_device_name(index)}")
else:
    print("WARNING: CUDA no disponible. Los entrenamientos usaran CPU salvo que --device indique otra cosa.")
PY

for lambda_logic in "${LAMBDAS[@]}"; do
    tag="$(lambda_tag "${lambda_logic}")"
    run_id="eegnet_logic_lam${tag}_seed${SEED}_lr${LEARNING_RATE}_ep${EPOCHS}_es${EARLY_STOPPING_PATIENCE}"
    run_dir="${RUN_ROOT}/${run_id}"
    summary_json="${run_dir}/summary.json"

    if [[ -f "${summary_json}" ]] && [[ "${FORCE:-0}" != "1" ]]; then
        log "SKIP ${run_id}: ya existe ${summary_json}. Usa FORCE=1 para repetir."
        skipped+=("${run_id}")
        continue
    fi

    if [[ -d "${run_dir}" ]] && [[ "${FORCE:-0}" == "1" ]]; then
        forced_run_id="${run_id}_force$(date '+%Y%m%d_%H%M%S')"
        log "FORCE=1 y existe ${run_dir}; se usara run_id alternativo ${forced_run_id} para no borrar resultados."
        run_id="${forced_run_id}"
    fi

    log "RUN ${run_id}: lambda_logic=${lambda_logic}"
    python "${SCRIPT_PATH}" \
        --run-id "${run_id}" \
        --run-root "${RUN_ROOT}" \
        --data-root "${DATA_ROOT}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --learning-rate "${LEARNING_RATE}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --scheduler "${SCHEDULER}" \
        --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
        --checkpoint-every "${CHECKPOINT_EVERY}" \
        --seed "${SEED}" \
        --lambda-logic "${lambda_logic}" \
        --baseline-checkpoint "${BASELINE_CHECKPOINT}" \
        2>&1 | tee -a "${LOG_PATH}"

    status="${PIPESTATUS[0]}"
    if [[ "${status}" -eq 0 ]]; then
        log "OK ${run_id}"
        completed+=("${run_id}")
    else
        log "ERROR ${run_id}: exit_code=${status}"
        failed+=("${run_id}")
    fi
done

if [[ -f "experiments_mi_ltn/summarize_logic_loss_lambdas.py" ]]; then
    log "Actualizando resumen de logic loss"
    python "experiments_mi_ltn/summarize_logic_loss_lambdas.py" 2>&1 | tee -a "${LOG_PATH}"
    summary_status="${PIPESTATUS[0]}"
    if [[ "${summary_status}" -ne 0 ]]; then
        log "WARNING: fallo el resumen de logic loss con exit_code=${summary_status}"
    fi
fi

log "Resumen final"
log "Completados (${#completed[@]}): ${completed[*]:-ninguno}"
log "Saltados (${#skipped[@]}): ${skipped[*]:-ninguno}"
log "Fallidos (${#failed[@]}): ${failed[*]:-ninguno}"
log "Baseline elegido: ${BASELINE_RUN_ID}"
log "Ruta de resultados: ${RUN_ROOT}"
log "Log: ${LOG_PATH}"
log "CSV resumen global de runs: ${SUMMARY_CSV}"
log "CSV resumen lambdas: ${RUN_ROOT}/logic_lambdas_summary.csv"
