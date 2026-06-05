# Experiments MI LTN

Experimentos para un TFM sobre IA neurosimbolica aplicada a EEG/BCI. El objetivo inicial es entrenar EEGNet sobre BCI Competition IV 2a como baseline en PyTorch, estimar importancia de canales por permutacion y probar una perdida logica simple inspirada en Logic Tensor Networks.

El core de TorchEEG no se modifica. Los scripts cargan `BCICIV2aDataset` y usan EEGNet desde el codigo local de TorchEEG, evitando la importacion global de `torcheeg.models` porque en este entorno arrastra dependencias GNN opcionales como `torch_scatter`.

## Dataset esperado

Coloca los `.mat` oficiales en:

```text
data/
└── BCICIV_2a_mat/
    ├── A01T.mat
    ├── A01E.mat
    ├── ...
    └── A09E.mat
```

Los scripts esperan esa ruta relativa al repo: `../data/BCICIV_2a_mat/` desde esta carpeta, o `data/BCICIV_2a_mat/` desde la raiz del repo. TorchEEG generara cache de IO en `experiments_mi_ltn/io/`.

## Scripts

Comprobar dataset y forward pass:

```bash
python experiments_mi_ltn/01_check_dataset.py
```

Si aun no existe el dataset:

```bash
python experiments_mi_ltn/01_check_dataset.py --dry-run
python experiments_mi_ltn/01_check_dataset.py --help
```

Entrenar baseline EEGNet:

```bash
python experiments_mi_ltn/02_train_eegnet_bciciv2a.py --epochs 30 --batch-size 64 --learning-rate 0.001
```

Cada ejecucion nueva crea una carpeta propia en `experiments_mi_ltn/runs/baseline_eegnet/`.
Dentro se guardan `checkpoint_best.pt`, `history.csv`, `summary.json`, `args.json`
y `figures/training_curves.png`.

Calcular importancia de canales por permutacion:

```bash
python experiments_mi_ltn/03_channel_importance.py --checkpoint experiments_mi_ltn/runs/baseline_eegnet/<run_id>/checkpoint_best.pt
```

Entrenar EEGNet con perdida logica:

```bash
python experiments_mi_ltn/04_logic_loss_eegnet.py --lambda-logic 0.1 --baseline-checkpoint experiments_mi_ltn/runs/best_eegnet_bciciv2a.pt
```

La perdida usada es:

```text
logic_loss = 1 - mean(probabilidad asignada a la clase verdadera)
total_loss = cross_entropy + lambda_logic * logic_loss
```
