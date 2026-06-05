# Run EEGNet Baseline Phase 1

Este documento explica como ejecutar la Fase 1 de EEGNet baseline en Linux, idealmente con una GPU NVIDIA RTX 3060.

Ejecutar todo desde la raiz del repositorio.

## 1. Clonar repositorio

```bash
git clone https://github.com/didi24003/TFM_Neuro_Symbolic.git
cd TFM_Neuro_Symbolic
```

## 2. Instalar dependencias

Si existe un `requirements.txt` en el entorno que vas a usar:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

El `requirements.txt` instala PyTorch, Matplotlib y el paquete local con `-e .`. Si PyTorch no detecta CUDA, instala una version de PyTorch compatible con la version de CUDA/driver de la maquina.

## 3. Preparar dataset

El dataset no se incluye en GitHub. Debes colocarlo manualmente desde la carpeta externa compartida o desde la fuente oficial de BCI Competition IV, respetando sus terminos de uso. Los `.mat` de BCI Competition IV 2a deben quedar en:

```text
data/BCICIV_2a_mat/
```

Debe contener 18 archivos:

```text
A01T.mat A01E.mat ... A09T.mat A09E.mat
```

Comprobar dataset:

```bash
ls data/BCICIV_2a_mat/
```

## 4. Comprobar CUDA

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Lo esperado en la maquina con RTX 3060 es que imprima `True` y el nombre de la GPU.

## 5. Dar permisos al script

```bash
chmod +x experiments_mi_ltn/run_baseline_phase1.sh
```

## 6. Lanzar los experimentos

```bash
./experiments_mi_ltn/run_baseline_phase1.sh
```

El script ejecuta secuencialmente los siete experimentos de Fase 1. Si falta `data/BCICIV_2a_mat/`, se detiene con un mensaje claro. Si un run ya tiene `summary.json`, lo omite para no repetir trabajo ni sobrescribir resultados.

## 7. Forzar repeticion

```bash
FORCE=1 ./experiments_mi_ltn/run_baseline_phase1.sh
```

Con `FORCE=1`, si ya existe la carpeta de un run, el script crea una repeticion con sufijo `force_YYYYMMDD_HHMMSS` para no borrar resultados antiguos.

## 8. Resultados

Cada run se guarda en:

```text
experiments_mi_ltn/runs/baseline_eegnet/<run_id>/
```

Cada carpeta de run debe contener:

```text
checkpoint_best.pt
checkpoint_last.pt
checkpoints/checkpoint_epoch_010.pt
checkpoints/checkpoint_epoch_020.pt
history.csv
summary.json
args.json
figures/training_curves.png
```

El log global se guarda en:

```text
experiments_mi_ltn/runs/baseline_eegnet/phase1_run.log
```

El CSV resumen se guarda en:

```text
experiments_mi_ltn/runs/baseline_eegnet/baseline_runs_summary.csv
```

## 9. Comprobar si terminaron bien

Al final del script aparece un resumen con:

```text
Completed (...):
Skipped (...):
Failed (...):
```

Tambien se puede revisar:

```bash
tail -80 experiments_mi_ltn/runs/baseline_eegnet/phase1_run.log
```

Si `Failed` aparece como `none` o `0`, no hubo fallos. Los runs completados deben tener `summary.json` y aparecer en `baseline_runs_summary.csv`.

## 10. Comprimir resultados

Cuando terminen los experimentos, comprime esta carpeta:

```bash
tar -czf baseline_eegnet_results.tar.gz experiments_mi_ltn/runs/baseline_eegnet/
```

Ese archivo `baseline_eegnet_results.tar.gz` es el que debes enviar.
