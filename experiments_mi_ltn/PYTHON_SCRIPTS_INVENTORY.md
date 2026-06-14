# Python Scripts Inventory

Generated: 2026-06-14T22:46:04

Directorio revisado: `experiments_mi_ltn/`

## Resumen

- Grupo A (Mantener obligatoriamente): 10
- Grupo B (Mantener opcionalmente): 3
- Grupo C (Archivar): 4
- Grupo D (Eliminar solo si confirmo): 0

## Inventario detallado

### `01_check_dataset.py`

- grupo: A
- que_hace: Comprueba carga del dataset BCI IV 2a y un forward pass de EEGNet.
- usado_en_sh: no
- usado_en_pipeline_final: yes
- duplicado_por: no
- depende_de_resultados_30_epocas: no
- se_puede_archivar: no
- debe_conservarse: yes
- notas: Script de validacion inicial del entorno y dataset; util para reproducibilidad.

### `02_train_eegnet_bciciv2a.py`

- grupo: A
- que_hace: Entrena el baseline EEGNet y guarda history, summaries y checkpoints.
- usado_en_sh: run_baseline_phase1.sh; run_baseline_best_seeds.sh; run_rebuild_baseline_best_seeds.sh
- usado_en_pipeline_final: yes
- duplicado_por: no
- depende_de_resultados_30_epocas: no
- se_puede_archivar: no
- debe_conservarse: yes
- notas: Script central del baseline final.

### `03_channel_importance.py`

- grupo: A
- que_hace: Calcula permutation importance por canal a partir de un checkpoint.
- usado_en_sh: run_final_channel_analysis.sh
- usado_en_pipeline_final: yes
- duplicado_por: no
- depende_de_resultados_30_epocas: no
- se_puede_archivar: no
- debe_conservarse: yes
- notas: Genera CSV, sorted CSV y barplot del analisis final.

### `04_logic_loss_eegnet.py`

- grupo: A
- que_hace: Entrena EEGNet con logic loss y guarda history, summaries y checkpoints.
- usado_en_sh: run_logic_loss_lambdas.sh
- usado_en_pipeline_final: yes
- duplicado_por: no
- depende_de_resultados_30_epocas: no
- se_puede_archivar: no
- debe_conservarse: yes
- notas: Script central de logic loss para el TFM final.

### `05_plot_channel_importance.py`

- grupo: C
- que_hace: Ordena un CSV de channel importance y dibuja un barplot simple.
- usado_en_sh: no
- usado_en_pipeline_final: no
- duplicado_por: 03_channel_importance.py
- depende_de_resultados_30_epocas: no
- se_puede_archivar: yes
- debe_conservarse: no
- notas: Quedo reemplazado porque 03 ya puede escribir el CSV ordenado y el barplot en una sola ejecucion.

### `05_plot_channel_importance_comparison.py`

- grupo: A
- que_hace: Compara importancia baseline vs logic loss y genera CSV y figuras de comparacion/delta.
- usado_en_sh: run_final_channel_analysis.sh
- usado_en_pipeline_final: yes
- duplicado_por: no
- depende_de_resultados_30_epocas: no
- se_puede_archivar: no
- debe_conservarse: yes
- notas: Parte directa del pipeline final de comparacion.

### `05_plot_channel_importance_pretty.py`

- grupo: C
- que_hace: Genera figuras comparativas mas estilizadas para channel importance.
- usado_en_sh: no
- usado_en_pipeline_final: no
- duplicado_por: 05_plot_channel_importance_comparison.py
- depende_de_resultados_30_epocas: indirectly/legacy
- se_puede_archivar: yes
- debe_conservarse: no
- notas: No lo usa ningun .sh final y contiene accuracies hardcodeadas desactualizadas.

### `06_channel_importance_by_class.py`

- grupo: B
- que_hace: Calcula permutation importance por canal y por clase para baseline y logic loss.
- usado_en_sh: no
- usado_en_pipeline_final: no
- duplicado_por: no
- depende_de_resultados_30_epocas: no
- se_puede_archivar: not recommended
- debe_conservarse: optional
- notas: Util para analisis complementario, pero no forma parte del pipeline principal actual.

### `07_sensorimotor_coherence_score.py`

- grupo: C
- que_hace: Calcula ratios de coherencia sensorimotora desde CSVs de importance.
- usado_en_sh: no
- usado_en_pipeline_final: no
- duplicado_por: 08_finalize_channel_analysis.py
- depende_de_resultados_30_epocas: yes
- se_puede_archivar: yes
- debe_conservarse: no
- notas: Sus defaults apuntan a archivos *_30ep.csv y su funcionalidad quedo absorbida por 08.

### `08_finalize_channel_analysis.py`

- grupo: A
- que_hace: Construye delta por canal, coherence CSV/PNG, tabla LaTeX y resumen JSON final.
- usado_en_sh: run_final_channel_analysis.sh
- usado_en_pipeline_final: yes
- duplicado_por: no
- depende_de_resultados_30_epocas: no
- se_puede_archivar: no
- debe_conservarse: yes
- notas: Consolida el resumen final del analisis de canales.

### `09_plot_channel_importance_by_class.py`

- grupo: B
- que_hace: Dibuja figuras top-k por clase a partir del CSV de 06.
- usado_en_sh: no
- usado_en_pipeline_final: no
- duplicado_por: no
- depende_de_resultados_30_epocas: no
- se_puede_archivar: not recommended
- debe_conservarse: optional
- notas: Util para figuras adicionales o apendice, pero no entra en el pipeline principal.

### `10_plot_topomap_channel_importance.py`

- grupo: A
- que_hace: Genera topomaps de importancia por canal.
- usado_en_sh: run_final_channel_analysis.sh
- usado_en_pipeline_final: yes
- duplicado_por: no
- depende_de_resultados_30_epocas: mixed
- se_puede_archivar: no
- debe_conservarse: yes
- notas: El modo final usa --input-csv/--output-path; mantiene tambien un modo legado con defaults *_30ep.

### `11_plot_eegnet_architecture.py`

- grupo: B
- que_hace: Genera una figura conceptual de la arquitectura EEGNet para documentacion del TFM.
- usado_en_sh: no
- usado_en_pipeline_final: no
- duplicado_por: no
- depende_de_resultados_30_epocas: no
- se_puede_archivar: not recommended
- debe_conservarse: optional
- notas: No es parte del pipeline experimental, pero si es util para figuras del documento.

### `12_update_baseline_summary.py`

- grupo: C
- que_hace: Recolecta summaries baseline en un CSV agregado con notas de runs phase1.
- usado_en_sh: no
- usado_en_pipeline_final: no
- duplicado_por: summarize_baseline_best_seeds.py; baseline_runs_summary.csv generado por 02_train_eegnet_bciciv2a.py
- depende_de_resultados_30_epocas: legacy/phase1
- se_puede_archivar: yes
- debe_conservarse: no
- notas: Arrastra notas de phase1 y ya no gobierna el resumen final de seeds seleccionadas.

### `mi_ltn_common.py`

- grupo: A
- que_hace: Modulo comun con dataset, modelo EEGNet, seeds, loaders y utilidades compartidas.
- usado_en_sh: run_final_channel_analysis.sh (comprobacion indirecta); usado por multiples .py
- usado_en_pipeline_final: yes
- duplicado_por: no
- depende_de_resultados_30_epocas: no
- se_puede_archivar: no
- debe_conservarse: yes
- notas: Dependencia comun del bloque experimental; no es un script ejecutable, pero es esencial.

### `summarize_baseline_best_seeds.py`

- grupo: A
- que_hace: Genera el CSV y la tabla LaTeX de los baseline finales por seed.
- usado_en_sh: run_baseline_best_seeds.sh; run_rebuild_baseline_best_seeds.sh
- usado_en_pipeline_final: yes
- duplicado_por: no
- depende_de_resultados_30_epocas: no
- se_puede_archivar: no
- debe_conservarse: yes
- notas: Resumen final de baselines seleccionados.

### `summarize_logic_loss_lambdas.py`

- grupo: A
- que_hace: Resume barrido de lambdas logic loss y compara contra el baseline seleccionado.
- usado_en_sh: run_logic_loss_lambdas.sh
- usado_en_pipeline_final: yes
- duplicado_por: no
- depende_de_resultados_30_epocas: no
- se_puede_archivar: no
- debe_conservarse: yes
- notas: Necesario para la comparacion final de logic loss.

## Propuesta de clasificacion

### Grupo A. Mantener obligatoriamente

- `01_check_dataset.py`
- `02_train_eegnet_bciciv2a.py`
- `03_channel_importance.py`
- `04_logic_loss_eegnet.py`
- `05_plot_channel_importance_comparison.py`
- `08_finalize_channel_analysis.py`
- `10_plot_topomap_channel_importance.py`
- `mi_ltn_common.py`
- `summarize_baseline_best_seeds.py`
- `summarize_logic_loss_lambdas.py`

### Grupo B. Mantener opcionalmente

- `06_channel_importance_by_class.py`
- `09_plot_channel_importance_by_class.py`
- `11_plot_eegnet_architecture.py`

### Grupo C. Archivar

- `05_plot_channel_importance.py`
- `05_plot_channel_importance_pretty.py`
- `07_sensorimotor_coherence_score.py`
- `12_update_baseline_summary.py`

### Grupo D. Eliminar solo si confirmo

- ninguno
