# Baseline Runs Inventory

Generated: 2026-06-14T23:10:21

Root: `experiments_mi_ltn/runs/baseline_eegnet/`

## Summary

- Total run directories: 5
- Runs with `checkpoint_best.pt`: 0
- Runs without `checkpoint_best.pt`: 5
- Historical Phase 1 runs kept for hyperparameter justification: 5

Historical Phase 1 runs are intentionally preserved even without checkpoints because they support the hyperparameter-selection narrative. They are not part of the final channel-analysis rebuild target.

## Runs

### `eegnet_long300_seed42_lr9e-4_es50`

- `summary.json`: yes
- `args.json`: yes
- `history.csv`: yes
- `checkpoint_best.pt`: no
- `checkpoint_last.pt`: no
- `checkpoints/`: no
- `figures/training_curves.png`: yes
- `best_val_acc`: 0.7644787644787645
- `seed`: 42
- config: `model=EEGNet, epochs=300, batch_size=64, lr=0.0009, weight_decay=0.0, scheduler=none, early_stopping_patience=50, checkpoint_every=10, optimizer=Adam`
- historical Phase 1 run kept: yes
- final rebuild target: no
- retention reason: `historical_phase1_hyperparameter_selection`
- archive candidate: no

### `eegnet_long500_seed42_lr9e-4_es75`

- `summary.json`: yes
- `args.json`: yes
- `history.csv`: yes
- `checkpoint_best.pt`: no
- `checkpoint_last.pt`: no
- `checkpoints/`: no
- `figures/training_curves.png`: yes
- `best_val_acc`: 0.7654440154440154
- `seed`: 42
- config: `model=EEGNet, epochs=500, batch_size=64, lr=0.0009, weight_decay=0.0, scheduler=none, early_stopping_patience=75, checkpoint_every=10, optimizer=Adam`
- historical Phase 1 run kept: yes
- final rebuild target: no
- retention reason: `historical_phase1_hyperparameter_selection`
- archive candidate: no

### `eegnet_lr1e-3_seed42_ep300_es50`

- `summary.json`: yes
- `args.json`: yes
- `history.csv`: yes
- `checkpoint_best.pt`: no
- `checkpoint_last.pt`: no
- `checkpoints/`: no
- `figures/training_curves.png`: yes
- `best_val_acc`: 0.752895752895753
- `seed`: 42
- config: `model=EEGNet, epochs=300, batch_size=64, lr=0.001, weight_decay=0.0, scheduler=none, early_stopping_patience=50, checkpoint_every=10, optimizer=Adam`
- historical Phase 1 run kept: yes
- final rebuild target: no
- retention reason: `historical_phase1_hyperparameter_selection`
- archive candidate: no

### `eegnet_plateau_seed42_ep300_lr9e-4_es50`

- `summary.json`: yes
- `args.json`: yes
- `history.csv`: yes
- `checkpoint_best.pt`: no
- `checkpoint_last.pt`: no
- `checkpoints/`: no
- `figures/training_curves.png`: yes
- `best_val_acc`: 0.7422779922779923
- `seed`: 42
- config: `model=EEGNet, epochs=300, batch_size=64, lr=0.0009, weight_decay=0.0, scheduler=plateau, early_stopping_patience=50, checkpoint_every=10, optimizer=Adam`
- historical Phase 1 run kept: yes
- final rebuild target: no
- retention reason: `historical_phase1_hyperparameter_selection`
- archive candidate: no

### `eegnet_wd1e-4_seed42_ep300_lr9e-4_es50`

- `summary.json`: yes
- `args.json`: yes
- `history.csv`: yes
- `checkpoint_best.pt`: no
- `checkpoint_last.pt`: no
- `checkpoints/`: no
- `figures/training_curves.png`: yes
- `best_val_acc`: 0.7480694980694981
- `seed`: 42
- config: `model=EEGNet, epochs=300, batch_size=64, lr=0.0009, weight_decay=0.0001, scheduler=none, early_stopping_patience=50, checkpoint_every=10, optimizer=Adam`
- historical Phase 1 run kept: yes
- final rebuild target: no
- retention reason: `historical_phase1_hyperparameter_selection`
- archive candidate: no
