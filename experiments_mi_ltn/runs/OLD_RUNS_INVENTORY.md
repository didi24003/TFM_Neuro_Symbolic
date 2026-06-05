# Old Runs Inventory

Inventario realizado el 2026-06-05. No se ha borrado ni movido nada.

## Conservar

- `experiments_mi_ltn/runs/best_eegnet_bciciv2a.pt`: mejor ejecución antigua de 30 épocas encontrada. Se conserva como referencia inicial porque tiene `best_val_acc=0.721042471042471`, superior a los runs estructurados de 30 épocas.
- `experiments_mi_ltn/runs/baseline_eegnet/phase1_seed42_default/`: run estructurado completo de 30 épocas con `history.csv`, `summary.json`, `args.json` y curvas. Aunque rinde menos que el histórico, sirve para trazabilidad del pipeline actual.
- `experiments_mi_ltn/runs/channel_importance_baseline_30ep.csv` y `experiments_mi_ltn/runs/topomap_channel_importance_baseline_30ep.png`: conservar mientras se use el checkpoint histórico baseline como referencia.

## Archivar Propuesto

- `experiments_mi_ltn/runs/baseline_eegnet/phase1_seed42_plateau/`: misma accuracy que `phase1_seed42_default`; el scheduler no llegó a cambiar el LR en 30 épocas.
- `experiments_mi_ltn/runs/baseline_eegnet/phase1_seed42_wd1e-4/`: ejecución incompleta o interrumpida; `args.json` pide 30 épocas pero solo hay 15 épocas en `history.csv`.
- Figuras antiguas sin sufijo `30ep`: `channel_importance_comparison.png`, `channel_importance_delta.png`, `channel_importance_pretty.png`, `channel_importance_delta_pretty.png`. No tienen trazabilidad suficiente para figura final del TFM.

Si se confirma el archivado, mover estos artefactos a `experiments_mi_ltn/runs/archive/` y registrar el movimiento en `experiments_mi_ltn/runs/archive/README.md`.

## Posible Eliminación

- `experiments_mi_ltn/runs/baseline_eegnet/timing_seed42_epoch1/`: run de una época, útil solo como medición/debug. Puede eliminarse después de confirmar que no se necesita para justificar tiempos.
- `experiments_mi_ltn/runs/channel_importance_by_class_baseline_limit500.csv`: parece una ejecución parcial limitada; archivar primero y eliminar solo si se confirma que no aparece citada en el texto del TFM.
