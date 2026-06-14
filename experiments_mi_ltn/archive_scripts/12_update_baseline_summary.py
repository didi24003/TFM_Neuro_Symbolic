#!/usr/bin/env python
"""Collect EEGNet baseline run summaries into one CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from mi_ltn_common import RUNS_DIR


BASELINE_RUN_ROOT = RUNS_DIR / "baseline_eegnet"
SUMMARY_CSV = BASELINE_RUN_ROOT / "baseline_experiments_summary.csv"

FIELDNAMES = [
    "run_id",
    "seed",
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "scheduler",
    "early_stopping_patience",
    "best_val_acc",
    "best_epoch",
    "final_train_acc",
    "final_val_acc",
    "checkpoint_path",
    "history_csv",
    "summary_json",
    "notes",
]

DEFAULT_NOTES = {
    "phase1_seed42_default": "Baseline canonico: configuracion por defecto.",
    "phase1_seed42_plateau": "Mismo baseline con scheduler ReduceLROnPlateau.",
    "phase1_seed42_wd1e-4": "Mismo baseline con weight_decay=1e-4.",
    "phase1_seed42_wd1e-5": "Mismo baseline con weight_decay=1e-5.",
    "phase1_seed42_plateau_wd1e-4": "Scheduler plateau y weight_decay=1e-4.",
    "phase1_seed42_earlystop10": "Early stopping prudente con paciencia de 10 epocas.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=BASELINE_RUN_ROOT)
    parser.add_argument("--output-csv", type=Path, default=SUMMARY_CSV)
    return parser.parse_args()


def empty_row(output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def final_metrics(history_csv: Path) -> tuple[float | None, float | None]:
    if not history_csv.exists():
        return None, None
    history = pd.read_csv(history_csv)
    if history.empty:
        return None, None
    last = history.iloc[-1]
    return float(last["train_acc"]), float(last["val_acc"])


def row_from_run(run_dir: Path) -> dict:
    summary_json = run_dir / "summary.json"
    summary = read_json(summary_json)
    history_csv = Path(summary["history_csv"])
    checkpoint_path = Path(summary["checkpoint_path"])
    final_train_acc, final_val_acc = final_metrics(history_csv)
    run_id = run_dir.name

    return {
        "run_id": run_id,
        "seed": summary.get("seed"),
        "epochs": summary.get("epochs_completed", summary.get("epochs_requested")),
        "batch_size": summary.get("batch_size"),
        "learning_rate": summary.get("learning_rate"),
        "weight_decay": summary.get("weight_decay"),
        "scheduler": summary.get("scheduler"),
        "early_stopping_patience": summary.get("early_stopping_patience"),
        "best_val_acc": summary.get("best_val_acc"),
        "best_epoch": summary.get("best_epoch"),
        "final_train_acc": final_train_acc,
        "final_val_acc": final_val_acc,
        "checkpoint_path": str(checkpoint_path),
        "history_csv": str(history_csv),
        "summary_json": str(summary_json),
        "notes": DEFAULT_NOTES.get(run_id, ""),
    }


def main() -> None:
    args = parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_dir in sorted(args.run_root.iterdir()):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "summary.json").exists():
            continue
        rows.append(row_from_run(run_dir))

    rows.sort(
        key=lambda row: (
            float("-inf") if row["best_val_acc"] is None else float(row["best_val_acc"]),
            row["run_id"],
        ),
        reverse=True,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        empty_row(args.output_csv)
    else:
        with args.output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

    print(f"wrote {len(rows)} rows: {args.output_csv}")


if __name__ == "__main__":
    main()
