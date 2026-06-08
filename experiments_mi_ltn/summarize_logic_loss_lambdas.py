#!/usr/bin/env python
"""Summarize EEGNet logic-loss lambda runs and compare them with the selected baseline."""

from __future__ import annotations

import csv
import json
from pathlib import Path


RUN_ROOT = Path("experiments_mi_ltn/runs/logic_loss_eegnet")
OUTPUT_CSV = RUN_ROOT / "logic_lambdas_summary.csv"
OUTPUT_TEX = RUN_ROOT / "logic_lambdas_results_table.tex"
COMPARISON_CSV = RUN_ROOT / "logic_lambdas_vs_baseline.csv"
COMPARISON_TEX = RUN_ROOT / "logic_lambdas_vs_baseline_table.tex"

BASELINE_RUN_ID = "eegnet_best_lr5e-4_seed2024_ep300_es50"
BASELINE_SUMMARY = (
    Path("experiments_mi_ltn/runs/baseline_eegnet") / BASELINE_RUN_ID / "summary.json"
)
BASELINE_BEST_VAL_ACC = 0.7731660231660231

BASELINE_CONFIG = {
    "seed": 2024,
    "epochs": 300,
    "lr": 0.0005,
    "batch_size": 64,
    "weight_decay": 0.0,
    "scheduler": "none",
    "early_stopping_patience": 50,
}
LAMBDAS = ["0.001", "0.01", "0.05", "0.1", "0.2", "0.5", "1.0"]

FIELDNAMES = [
    "run_id",
    "lambda_logic",
    "seed",
    "epochs",
    "lr",
    "batch_size",
    "weight_decay",
    "scheduler",
    "early_stopping_patience",
    "best_val_acc",
    "best_epoch",
    "final_train_acc",
    "final_val_acc",
    "checkpoint_best_path",
    "history_csv",
    "summary_json",
    "status",
    "notes",
]

COMPARISON_FIELDNAMES = [
    "baseline_run_id",
    "baseline_best_val_acc",
    "run_id",
    "lambda_logic",
    "logic_best_val_acc",
    "absolute_difference",
    "is_best_lambda",
    "best_epoch",
    "final_val_acc",
    "status",
    "notes",
]


def lambda_tag(value: str) -> str:
    return value.replace(".", "p")


def expected_run_id(value: str) -> str:
    return (
        f"eegnet_logic_lam{lambda_tag(value)}"
        f"_seed{BASELINE_CONFIG['seed']}"
        f"_lr5e-4"
        f"_ep{BASELINE_CONFIG['epochs']}"
        f"_es{BASELINE_CONFIG['early_stopping_patience']}"
    )


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def read_final_metrics(history_csv: Path) -> tuple[str, str]:
    if not history_csv.exists():
        return "", ""
    with history_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return "", ""
    last = rows[-1]
    return last.get("train_acc", ""), last.get("val_acc", "")


def baseline_best_val_acc() -> float:
    summary = read_json(BASELINE_SUMMARY)
    if summary is None:
        return BASELINE_BEST_VAL_ACC
    value = summary.get("best_val_acc")
    return BASELINE_BEST_VAL_ACC if value in ("", None) else float(value)


def summary_for_lambda(value: str) -> dict:
    run_id = expected_run_id(value)
    run_dir = RUN_ROOT / run_id
    summary_path = run_dir / "summary.json"
    summary = read_json(summary_path)

    if summary is None:
        return {
            "run_id": run_id,
            "lambda_logic": value,
            "seed": BASELINE_CONFIG["seed"],
            "epochs": BASELINE_CONFIG["epochs"],
            "lr": BASELINE_CONFIG["lr"],
            "batch_size": BASELINE_CONFIG["batch_size"],
            "weight_decay": BASELINE_CONFIG["weight_decay"],
            "scheduler": BASELINE_CONFIG["scheduler"],
            "early_stopping_patience": BASELINE_CONFIG["early_stopping_patience"],
            "best_val_acc": "",
            "best_epoch": "",
            "final_train_acc": "",
            "final_val_acc": "",
            "checkpoint_best_path": str(run_dir / "checkpoint_best.pt"),
            "history_csv": str(run_dir / "history.csv"),
            "summary_json": str(summary_path),
            "status": "pending",
            "notes": "summary.json not found yet",
        }

    history_csv = Path(summary.get("history_csv", run_dir / "history.csv"))
    if not history_csv.exists():
        history_csv = run_dir / "history.csv"
    final_train_acc, final_val_acc = read_final_metrics(history_csv)

    return {
        "run_id": run_id,
        "lambda_logic": summary.get("lambda_logic", value),
        "seed": summary.get("seed", BASELINE_CONFIG["seed"]),
        "epochs": summary.get("epochs_completed", BASELINE_CONFIG["epochs"]),
        "lr": summary.get("learning_rate", BASELINE_CONFIG["lr"]),
        "batch_size": summary.get("batch_size", BASELINE_CONFIG["batch_size"]),
        "weight_decay": summary.get("weight_decay", BASELINE_CONFIG["weight_decay"]),
        "scheduler": summary.get("scheduler", BASELINE_CONFIG["scheduler"]),
        "early_stopping_patience": summary.get(
            "early_stopping_patience", BASELINE_CONFIG["early_stopping_patience"]
        ),
        "best_val_acc": summary.get("best_val_acc", ""),
        "best_epoch": summary.get("best_epoch", ""),
        "final_train_acc": final_train_acc,
        "final_val_acc": final_val_acc,
        "checkpoint_best_path": summary.get("checkpoint_best_path", str(run_dir / "checkpoint_best.pt")),
        "history_csv": str(history_csv),
        "summary_json": str(summary_path),
        "status": summary.get("status", "completed"),
        "notes": "",
    }


def fmt_acc(value: object) -> str:
    if value in ("", None):
        return "--"
    return f"{float(value):.4f}"


def fmt_value(value: object) -> str:
    if value in ("", None):
        return "--"
    return str(value)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_results_tex(rows: list[dict]) -> None:
    lines = [
        "% Auto-generated by experiments_mi_ltn/summarize_logic_loss_lambdas.py",
        "\\begin{tabular}{rrrrr}",
        "\\toprule",
        "$\\lambda_{logic}$ & Best val. acc. & Best epoch & Final train acc. & Final val. acc. \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['lambda_logic']} & {fmt_acc(row['best_val_acc'])} & "
            f"{fmt_value(row['best_epoch'])} & {fmt_acc(row['final_train_acc'])} & "
            f"{fmt_acc(row['final_val_acc'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    OUTPUT_TEX.write_text("\n".join(lines))


def build_comparison(rows: list[dict]) -> list[dict]:
    baseline_acc = baseline_best_val_acc()
    completed = [
        row
        for row in rows
        if row.get("best_val_acc") not in ("", None)
        and row.get("status") not in ("pending", "failed")
    ]
    best_lambda = None
    if completed:
        best_lambda = max(completed, key=lambda row: float(row["best_val_acc"]))["lambda_logic"]

    comparison = []
    for row in rows:
        logic_acc = row.get("best_val_acc", "")
        if logic_acc in ("", None):
            diff = ""
            notes = "logic-loss result pending"
        else:
            diff = float(logic_acc) - baseline_acc
            notes = ""
        comparison.append(
            {
                "baseline_run_id": BASELINE_RUN_ID,
                "baseline_best_val_acc": baseline_acc,
                "run_id": row["run_id"],
                "lambda_logic": row["lambda_logic"],
                "logic_best_val_acc": logic_acc,
                "absolute_difference": diff,
                "is_best_lambda": "yes" if best_lambda is not None and row["lambda_logic"] == best_lambda else "",
                "best_epoch": row.get("best_epoch", ""),
                "final_val_acc": row.get("final_val_acc", ""),
                "status": row.get("status", ""),
                "notes": notes,
            }
        )
    return comparison


def write_comparison_tex(rows: list[dict]) -> None:
    lines = [
        "% Auto-generated by experiments_mi_ltn/summarize_logic_loss_lambdas.py",
        "\\begin{tabular}{rrrrr}",
        "\\toprule",
        "$\\lambda_{logic}$ & Baseline best val. acc. & Logic best val. acc. & $\\Delta$ & Final val. acc. \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['lambda_logic']} & {fmt_acc(row['baseline_best_val_acc'])} & "
            f"{fmt_acc(row['logic_best_val_acc'])} & {fmt_acc(row['absolute_difference'])} & "
            f"{fmt_acc(row['final_val_acc'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    COMPARISON_TEX.write_text("\n".join(lines))


def main() -> None:
    rows = [summary_for_lambda(value) for value in LAMBDAS]
    comparison = build_comparison(rows)
    write_csv(OUTPUT_CSV, rows, FIELDNAMES)
    write_results_tex(rows)
    write_csv(COMPARISON_CSV, comparison, COMPARISON_FIELDNAMES)
    write_comparison_tex(comparison)
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_TEX}")
    print(f"Wrote {COMPARISON_CSV}")
    print(f"Wrote {COMPARISON_TEX}")


if __name__ == "__main__":
    main()
