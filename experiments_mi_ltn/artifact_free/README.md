# Artifact-Free Experiments

Esta carpeta encapsula la variante `artifact_free` de los experimentos sobre BCI Competition IV 2a.

## Que hace

- Excluye los trials marcados como artefacto mediante `SKIP_ARTIFACTS=1`.
- Mantiene el protocolo `subject-specific cross-session`.
- Usa sesion `T` para entrenamiento y validacion.
- Usa sesion `E` completa para test final.
- Usa solo `22` canales EEG como entrada.
- No introduce los `3` canales EOG en el modelo.
- No aplica correccion EOG explicita; solo descarta los trials ya marcados como artefacto.

## Rutas de salida

- Baseline artifact-free:
  `experiments_mi_ltn/runs/eegnet_baseline_subject_session_artifact_free/`
- Logic artifact-free:
  `experiments_mi_ltn/runs/eegnet_logic_subject_session_artifact_free/`

Los wrappers de esta carpeta no sobrescriben los runs originales en:

- `experiments_mi_ltn/runs/eegnet_baseline_subject_session/`
- `experiments_mi_ltn/runs/eegnet_logic_subject_session/`

## Verificacion rapida

```bash
python experiments_mi_ltn/artifact_free/verify_artifact_filter.py
```

## Dry-run

```bash
bash experiments_mi_ltn/artifact_free/run_full_artifact_free_pipeline.sh --dry-run
```

## Lanzar baseline artifact-free

```bash
bash experiments_mi_ltn/artifact_free/run_baseline_artifact_free.sh
```

## Lanzar logic artifact-free

```bash
bash experiments_mi_ltn/artifact_free/run_logic_artifact_free.sh
```

## Lanzar todo

```bash
bash experiments_mi_ltn/artifact_free/run_full_artifact_free_pipeline.sh
```

## Solo resumen

```bash
bash experiments_mi_ltn/artifact_free/run_full_artifact_free_pipeline.sh --summary-only
```

## Exportar resultados

```bash
bash experiments_mi_ltn/artifact_free/export_artifact_free_results.sh
```

Esto genera `artifact_free_export.zip` con solo artefactos ligeros: `csv`, `json`, `txt` y `analysis_figures/`.

## Variables de entorno soportadas

- `DATA_ROOT=/ruta/al/dataset`
- `SEEDS="0 7 42 123 2024"`
- `OUT_DIR=/ruta/de/salida`
- `BASELINE_OUT_DIR=/ruta/baseline`
- `LOGIC_OUT_DIR=/ruta/logic`

## Que subir a Git

Subir:

- `experiments_mi_ltn/artifact_free/`
- los cambios minimos en los scripts base para soportar `SKIP_ARTIFACTS` y `OUT_DIR`

No subir:

- `data/`
- `experiments_mi_ltn/io/`
- `experiments_mi_ltn/runs/eegnet_baseline_subject_session_artifact_free/`
- `experiments_mi_ltn/runs/eegnet_logic_subject_session_artifact_free/`
- checkpoints, logs, export zips, cachés TorchEEG y modelos `.pt/.pth/.ckpt`
