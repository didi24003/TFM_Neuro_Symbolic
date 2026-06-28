# Run EEGNeX

Este flujo de EEGNeX se prepara y mantiene en la rama `eegnex-experiments`.

EEGNeX se instancia desde Braindecode. En este repo, la ruta nueva convive con EEGNet y `EEGNet` sigue siendo el comportamiento por defecto en los trainers si no se pasa `--model eegnex`.

## Requisitos

Ejecutar todo desde la raiz del repositorio.

El dataset debe existir en:

```text
data/BCICIV_2a_mat/
```

Braindecode debe estar instalado y debe funcionar:

```bash
python -c "import braindecode; from braindecode.models import EEGNeX; print(braindecode.__version__, EEGNeX)"
```

## Adaptacion del batch

El loader actual entrega batches con forma:

```text
(B, 1, C, T)
```

Para EEGNeX, antes del `forward`, el trainer adapta la entrada a:

```text
(B, C, T)
```

mediante:

```python
x = x.squeeze(1)
```

La adaptacion solo se aplica cuando `--model eegnex`.

## Smoke test

Para comprobar que EEGNeX sigue integrando bien el batch real:

```bash
python experiments_mi_ltn/99_smoke_test_eegnex.py
```

## EEGNeX baseline seed 2024

Dar permisos si hace falta:

```bash
chmod +x experiments_mi_ltn/run_eegnex_baseline_seed2024.sh
```

Lanzar:

```bash
./experiments_mi_ltn/run_eegnex_baseline_seed2024.sh
```

Si ya existe el run y quieres repetir sin borrar resultados previos:

```bash
FORCE=1 ./experiments_mi_ltn/run_eegnex_baseline_seed2024.sh
```

El log se guarda en:

```text
experiments_mi_ltn/runs/baseline_eegnex/eegnex_baseline_seed2024.log
```

## EEGNeX + logic loss lambda=1.0 seed 2024

Dar permisos si hace falta:

```bash
chmod +x experiments_mi_ltn/run_eegnex_logic_lam1p0_seed2024.sh
```

Lanzar:

```bash
./experiments_mi_ltn/run_eegnex_logic_lam1p0_seed2024.sh
```

Si ya existe el run y quieres repetir sin borrar resultados previos:

```bash
FORCE=1 ./experiments_mi_ltn/run_eegnex_logic_lam1p0_seed2024.sh
```

El log se guarda en:

```text
experiments_mi_ltn/runs/logic_loss_eegnex/eegnex_logic_lam1p0_seed2024.log
```

## Compresion de resultados

Baseline:

```bash
tar -czf eegnex_baseline_results.tar.gz experiments_mi_ltn/runs/baseline_eegnex/
```

Logic loss:

```bash
tar -czf eegnex_logic_results.tar.gz experiments_mi_ltn/runs/logic_loss_eegnex/
```

Si prefieres un unico paquete:

```bash
tar -czf eegnex_runs_seed2024.tar.gz \
  experiments_mi_ltn/runs/baseline_eegnex/ \
  experiments_mi_ltn/runs/logic_loss_eegnex/
```

## GitHub

Subir a GitHub:

- codigo Python modificado;
- scripts `.sh`;
- `RUN_EEGNEX_README.md`;
- resultados ligeros de validacion si los necesitas documentar;
- archivos de resumen pequenos como `summary.json`, `history.csv` o figuras solo si decides versionarlos de forma consciente.

No subir a GitHub:

- checkpoints `checkpoint_best.pt`, `checkpoint_last.pt` ni `checkpoints/`;
- logs pesados;
- dataset;
- artefactos grandes comprimidos;
- resultados generados por los entrenamientos largos si no esta acordado versionarlos.

## Responsabilidad de ejecucion

Los entrenamientos largos de 300 epocas no se lanzan desde esta preparacion. Debe ejecutarlos tu compañero despues, usando los scripts anteriores desde la rama `eegnex-experiments`.
