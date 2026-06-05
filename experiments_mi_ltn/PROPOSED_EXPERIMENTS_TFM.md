# Proposed Experiments for TFM

Inventario realizado el 2026-06-05. No se han lanzado nuevos entrenamientos.

## Inventario actual

El resumen global está en `experiments_mi_ltn/runs/global_experiments_summary.csv`.

Resultados principales encontrados:

- Mejor baseline histórico: `legacy_best_eegnet_bciciv2a`, EEGNet + cross entropy, seed 42, 30 épocas, batch size 32, lr 0.001, `best_val_acc=0.721042471042471`, `best_epoch=22`.
- Mejor logic loss histórico: `legacy_best_eegnet_logic_bciciv2a`, EEGNet + `cross_entropy + lambda_logic * logic_loss`, seed 42, 30 épocas, batch size 32, lr 0.001, `lambda_logic=1e-06`, `best_val_acc=0.7152509652509652`, `best_epoch=25`.
- Runs baseline estructurados existentes: `phase1_seed42_default`, `phase1_seed42_plateau`, `phase1_seed42_wd1e-4`, `timing_seed42_epoch1`.
- Channel importance, coherence scores y topomaps existen, pero deben tratarse como válidos solo si se reporta el checkpoint exacto de origen.
- No se ha encontrado ningún experimento EEGNeX en `experiments_mi_ltn`.

## Fase 1: EEGNet Baseline

Baseline de referencia para comparar cambios:

| run_id | Comando exacto | Cambio respecto al baseline | Qué comprueba | Archivos generados |
|---|---|---|---|---|
| `eegnet_canonical_seed42_ep100_lr9e-4_bs64` | `python experiments_mi_ltn/02_train_eegnet_bciciv2a.py --run-id eegnet_canonical_seed42_ep100_lr9e-4_bs64 --epochs 100 --batch-size 64 --learning-rate 9e-4 --weight-decay 0 --scheduler none --early-stopping-patience 0 --seed 42` | Baseline canónico más alineado con literatura: Adam, lr 9e-4, batch 64, 100 épocas, sin regularización extra. | Si el rendimiento mejora frente a 30 épocas sin añadir factores de confusión. | `experiments_mi_ltn/runs/baseline_eegnet/eegnet_canonical_seed42_ep100_lr9e-4_bs64/checkpoint_best.pt`, `history.csv`, `summary.json`, `args.json`, `figures/training_curves.png` |
| `eegnet_long300_seed42_lr9e-4_es50` | `python experiments_mi_ltn/02_train_eegnet_bciciv2a.py --run-id eegnet_long300_seed42_lr9e-4_es50 --epochs 300 --batch-size 64 --learning-rate 9e-4 --weight-decay 0 --scheduler none --early-stopping-patience 50 --seed 42` | Aumenta a 300 épocas con early stopping. | Si el baseline de 30/100 épocas estaba infraentrenado y dónde aparece el mejor checkpoint. | Misma carpeta de run bajo `experiments_mi_ltn/runs/baseline_eegnet/eegnet_long300_seed42_lr9e-4_es50/` |
| `eegnet_long500_seed42_lr9e-4_es75` | `python experiments_mi_ltn/02_train_eegnet_bciciv2a.py --run-id eegnet_long500_seed42_lr9e-4_es75 --epochs 500 --batch-size 64 --learning-rate 9e-4 --weight-decay 0 --scheduler none --early-stopping-patience 75 --seed 42` | Aumenta a 500 épocas con early stopping más paciente. | Si el rendimiento sigue mejorando con entrenamiento largo o si se estabiliza antes. | Misma carpeta de run bajo `experiments_mi_ltn/runs/baseline_eegnet/eegnet_long500_seed42_lr9e-4_es75/` |
| `eegnet_lr5e-4_seed42_ep300_es50` | `python experiments_mi_ltn/02_train_eegnet_bciciv2a.py --run-id eegnet_lr5e-4_seed42_ep300_es50 --epochs 300 --batch-size 64 --learning-rate 5e-4 --weight-decay 0 --scheduler none --early-stopping-patience 50 --seed 42` | Solo cambia lr a 5e-4 respecto al largo 300. | Si un lr más bajo mejora estabilidad/generalización. | Misma carpeta de run bajo `experiments_mi_ltn/runs/baseline_eegnet/eegnet_lr5e-4_seed42_ep300_es50/` |
| `eegnet_lr1e-3_seed42_ep300_es50` | `python experiments_mi_ltn/02_train_eegnet_bciciv2a.py --run-id eegnet_lr1e-3_seed42_ep300_es50 --epochs 300 --batch-size 64 --learning-rate 1e-3 --weight-decay 0 --scheduler none --early-stopping-patience 50 --seed 42` | Solo cambia lr a 1e-3 respecto al largo 300. | Si el lr actual del repo es competitivo al entrenar más épocas. | Misma carpeta de run bajo `experiments_mi_ltn/runs/baseline_eegnet/eegnet_lr1e-3_seed42_ep300_es50/` |
| `eegnet_wd1e-4_seed42_ep300_lr9e-4_es50` | `python experiments_mi_ltn/02_train_eegnet_bciciv2a.py --run-id eegnet_wd1e-4_seed42_ep300_lr9e-4_es50 --epochs 300 --batch-size 64 --learning-rate 9e-4 --weight-decay 1e-4 --scheduler none --early-stopping-patience 50 --seed 42` | Añade weight decay 1e-4. | Si la regularización L2 mejora validación o penaliza demasiado el aprendizaje. | Misma carpeta de run bajo `experiments_mi_ltn/runs/baseline_eegnet/eegnet_wd1e-4_seed42_ep300_lr9e-4_es50/` |
| `eegnet_plateau_seed42_ep300_lr9e-4_es50` | `python experiments_mi_ltn/02_train_eegnet_bciciv2a.py --run-id eegnet_plateau_seed42_ep300_lr9e-4_es50 --epochs 300 --batch-size 64 --learning-rate 9e-4 --weight-decay 0 --scheduler plateau --plateau-factor 0.5 --plateau-patience 10 --early-stopping-patience 50 --seed 42` | Añade `ReduceLROnPlateau`. | Si bajar lr al estancarse mejora el mejor checkpoint frente a lr fijo. | Misma carpeta de run bajo `experiments_mi_ltn/runs/baseline_eegnet/eegnet_plateau_seed42_ep300_lr9e-4_es50/` |

## Fase 1b: Seeds

Repetir solo las 2 mejores configuraciones de Fase 1 con varias seeds, por ejemplo `7`, `21`, `42`, `84`, `123`.

Plantilla de comando:

```bash
python experiments_mi_ltn/02_train_eegnet_bciciv2a.py --run-id <best_config>_seed<seed> --epochs <epochs> --batch-size 64 --learning-rate <lr> --weight-decay <wd> --scheduler <none|plateau> --early-stopping-patience <patience> --seed <seed>
```

Resultado esperado: media y desviación típica de `best_val_acc`, distribución de `best_epoch`, y selección del checkpoint final por criterio predefinido.

## Fase 2: Logic Loss

No lanzar logic loss hasta cerrar el mejor EEGNet baseline. Usar exactamente la mejor configuración de baseline y barrer:

```text
lambda_logic = [0, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5]
```

Antes de ejecutar esta fase conviene actualizar `experiments_mi_ltn/04_logic_loss_eegnet.py` para que guarde el mismo formato estructurado que el baseline: `run_id`, `args.json`, `history.csv`, `summary.json`, `checkpoint_best.pt` y `figures/training_curves.png`. Sin eso, la comparación queda más débil que la del baseline.

Tabla comparativa prevista:

```text
run_id, lambda_logic, seed, epochs, lr, batch_size, weight_decay, scheduler, early_stopping_patience, best_val_acc, best_epoch, final_train_acc, final_val_acc, checkpoint_path, history_csv, summary_json
```

Después, calcular channel importance y coherence scores solo para:

- mejor baseline;
- mejor logic loss por `best_val_acc`;
- logic loss con mejor coherencia sensorimotora si no coincide con el máximo de accuracy.

## Figuras

Regla aplicada: una figura solo es válida para el TFM si procede del checkpoint exacto que aparece en la tabla de resultados.

Figuras actuales que se pueden citar solo con advertencia de trazabilidad:

- `topomap_channel_importance_baseline_30ep.png`: vinculable al checkpoint histórico baseline si se reporta `experiments_mi_ltn/runs/best_eegnet_bciciv2a.pt`.
- `topomap_channel_importance_logic_30ep.png`: vinculable al checkpoint histórico logic si se reporta `experiments_mi_ltn/runs/best_eegnet_logic_bciciv2a.pt`.
- `coherence_scores.png`: válido solo para la comparación histórica basada en `channel_importance_baseline_30ep.csv` y `channel_importance_logic_30ep.csv`.
- `training_curves.png` de runs estructurados actuales: válidas para esos runs, pero no para el checkpoint histórico standalone.

Figuras que habría que regenerar para resultados finales:

- training curves de las mejores configuraciones;
- tabla de resultados EEGNet baseline;
- `lambda_logic` vs `best_val_acc`;
- baseline vs logic channel importance;
- delta channel importance;
- sensorimotor ratio / posterior ratio;
- topomap baseline;
- topomap logic;
- tabla comparativa EEGNet vs EEGNeX solo si se añade y ejecuta un experimento EEGNeX.

Figuras no recomendadas para el TFM final sin regenerar o relinkar:

- figuras antiguas sin sufijo `30ep`;
- `channel_importance_pretty.png` y `channel_importance_delta_pretty.png`, porque el script contiene accuracies hardcodeadas que no coinciden con los checkpoints actuales;
- `channel_importance_by_class.png`, hasta regenerarla desde los checkpoints finales.
