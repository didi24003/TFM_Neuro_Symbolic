# TFM Neuro-Symbolic EEG Experiments

Repositorio para el TFM sobre aprendizaje neurosimbolico aplicado a EEG/BCI. La fase experimental actual entrena EEGNet sobre BCI Competition IV 2a y prepara comparaciones posteriores con logic loss sin modificar TorchEEG core.

## Estructura

```text
experiments_mi_ltn/
  02_train_eegnet_bciciv2a.py      # entrenamiento EEGNet baseline
  run_baseline_phase1.sh           # lanza la Fase 1 completa
  RUN_BASELINE_PHASE1_README.md    # instrucciones detalladas para ejecutar
  runs/baseline_eegnet/            # resultados de EEGNet baseline
torcheeg/                          # codigo TorchEEG local
data/BCICIV_2a_mat/                # dataset local, no incluido en Git
```

## Instalar

Clonar el repositorio:

```bash
git clone https://github.com/didi24003/TFM_Neuro_Symbolic.git
cd TFM_Neuro_Symbolic
```

Crear entorno:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Se recomienda Python 3.10. Para una RTX 3060, comprueba que el PyTorch instalado detecta CUDA. Si no lo detecta, instala la rueda de PyTorch adecuada para el driver/CUDA de esa maquina siguiendo la documentacion oficial de PyTorch.

## Dataset

El dataset BCI Competition IV 2a no se incluye en GitHub. La carpeta completa local pesa unos 744 MB. Aunque los `.mat` individuales no superan 50 MB, subir el dataset a Git normal o LFS no es recomendable por tamano total, cuota/ancho de banda y posible restriccion de redistribucion.

Pagina oficial del dataset/competicion:

```text
https://www.bbci.de/competition/iv/
```

Usa el dataset respetando los terminos de la competicion y citando a los grupos que registraron los datos.

Coloca los 18 archivos `.mat` en:

```text
data/BCICIV_2a_mat/
```

Comprobar dataset:

```bash
ls data/BCICIV_2a_mat/
```

La carpeta esperada contiene archivos como:

```text
A01T.mat A01E.mat ... A09T.mat A09E.mat
```

Si se decide usar Git LFS en el futuro, los comandos base serian:

```bash
git lfs install
git lfs track "data/BCICIV_2a_mat/*.mat"
git add .gitattributes
```

En esta preparacion no se activa Git LFS para el dataset.

## Comprobar CUDA

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

En la maquina con RTX 3060 deberia imprimir `True` y el nombre de la GPU.

## Ejecutar Fase 1 EEGNet Baseline

Dar permisos al script:

```bash
chmod +x experiments_mi_ltn/run_baseline_phase1.sh
```

Ejecutar experimentos:

```bash
./experiments_mi_ltn/run_baseline_phase1.sh
```

Forzar repeticion si hace falta:

```bash
FORCE=1 ./experiments_mi_ltn/run_baseline_phase1.sh
```

El script comprueba que se ejecuta desde la raiz del repositorio, que existe el dataset, que PyTorch carga correctamente y si CUDA esta disponible. Si un run ya tiene `summary.json`, se omite salvo con `FORCE=1`.

## Resultados

Cada run se guarda en:

```text
experiments_mi_ltn/runs/baseline_eegnet/<run_id>/
```

El log global esta en:

```text
experiments_mi_ltn/runs/baseline_eegnet/phase1_run.log
```

El CSV resumen global esta en:

```text
experiments_mi_ltn/runs/baseline_eegnet/baseline_runs_summary.csv
```

Archivos pesados no incluidos por defecto:

- `data/`
- `experiments_mi_ltn/io/`
- checkpoints `*.pt`, `*.pth`, `*.ckpt`
- checkpoints periodicos en `experiments_mi_ltn/runs/**/checkpoints/`
- logs y archivos comprimidos

## Comprimir Resultados

La carpeta que debe devolverse es:

```text
experiments_mi_ltn/runs/baseline_eegnet/
```

O el archivo comprimido:

```bash
tar -czf baseline_eegnet_results.tar.gz experiments_mi_ltn/runs/baseline_eegnet/
```

## Comprobaciones Antes De Commit/Push

Revisar estado:

```bash
git status --short
find . -type f -size +50M
find . -type f -size +100M
find . -type f \( -name "*.mat" -o -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" -o -name "*.tar.gz" -o -name "*.zip" \)
git lfs ls-files
```

Ver que quedaria staged despues de `git add`:

```bash
git diff --cached --name-only
```

Revisar tamanos de archivos staged:

```bash
git diff --cached --name-only -z | xargs -0 -r du -h
```

## Comandos Para Primer Push

No ejecutar sin revisar antes que no haya archivos pesados staged.

```bash
git init
git branch -M main
git remote add origin https://github.com/didi24003/TFM_Neuro_Symbolic.git
git status
git add .
git status
git diff --cached --name-only
git commit -m "Initial TFM neurosymbolic EEG experiments"
git push -u origin main
```

Si el remoto ya existe:

```bash
git remote set-url origin https://github.com/didi24003/TFM_Neuro_Symbolic.git
```
