# Final Channel Analysis

Este flujo compara los dos modelos finales seleccionados para el TFM:

- `EEGNet baseline`: `experiments_mi_ltn/runs/baseline_eegnet/eegnet_best_lr5e-4_seed2024_ep300_es50`
- `EEGNet + logic loss (\lambda_logic=1.0)`: run detectado automáticamente desde `experiments_mi_ltn/runs/logic_loss_eegnet/logic_lambdas_summary.csv` o `logic_loss_runs_summary.csv`

El análisis usa exclusivamente los `checkpoint_best.pt` reales de esos runs. No entrena modelos, no lanza nuevos experimentos de clasificación y no modifica TorchEEG core.

## Qué genera

- Permutation importance para baseline y logic loss.
- CSV ordenados por importancia por canal.
- Barplots por modelo.
- Comparación baseline vs logic.
- Delta `logic - baseline` por canal.
- `sensorimotor_ratio` y `posterior_ratio` para ambos modelos.
- Figura comparativa de coherencia.
- Topomap final para baseline y logic loss.
- Tabla LaTeX breve para el TFM.
- Resumen JSON final.

## Requisitos

- Ejecutar desde la raíz del repositorio.
- Dataset disponible en `data/BCICIV_2a_mat/`.
- `summary.json` y `checkpoint_best.pt` de ambos runs finales.

Si falta algún checkpoint, el script se detiene e indica el archivo concreto que debe proporcionarse.

## Ejecución

```bash
chmod +x experiments_mi_ltn/run_final_channel_analysis.sh
./experiments_mi_ltn/run_final_channel_analysis.sh
```

Para forzar sobrescritura del directorio estándar:

```bash
FORCE=1 ./experiments_mi_ltn/run_final_channel_analysis.sh
```

Si ya existe `experiments_mi_ltn/runs/final_channel_analysis/` y `FORCE` no vale `1`, el script crea una carpeta con timestamp.

## Salida

Directorio principal esperado:

```text
experiments_mi_ltn/runs/final_channel_analysis/
```

o, si ya existía y no se forzó:

```text
experiments_mi_ltn/runs/final_channel_analysis_YYYYMMDD_HHMMSS/
```

Archivos finales clave para el TFM:

- `baseline/topomap_channel_importance.png`
- `logic_lam1p0/topomap_channel_importance.png`
- `comparison/channel_importance_comparison.png`
- `comparison/channel_importance_delta.png`
- `comparison/coherence_scores.png`
- `comparison/coherence_scores_table.tex`

El log completo se guarda en:

```text
experiments_mi_ltn/runs/final_channel_analysis*/final_channel_analysis.log
```
