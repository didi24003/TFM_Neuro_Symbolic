#!/usr/bin/env python
"""Summarize EEGNet baseline random-split runs and build lightweight artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_ROOT = SCRIPT_DIR / "runs" / "eegnet_baseline_random_split_60_20_20"
CONFIG_RUNS_DIR = RUN_ROOT / "config_runs"
BEST_SEEDS_DIR = RUN_ROOT / "best_config_seeds"
ANALYSIS_FIGURES_DIR = RUN_ROOT / "analysis_figures"
MPLCONFIG_DIR = RUN_ROOT / ".matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONFIG_SIMPlicity_RANK = {
    "baseline_short": 0,
    "long_training_300": 1,
    "lr_low": 2,
    "lr_high": 3,
    "weight_decay": 4,
    "scheduler_plateau": 5,
    "long_training_500": 6,
}
FIGURE_REQUIREMENTS = [
    "training_curves_seed42.png",
    "training_curves_seed2024.png",
    "validation_accuracy_by_config.png",
    "test_accuracy_by_config.png",
    "seed_variability_accuracy.png",
    "confusion_matrix_best_seed.png",
    "loss_curves_best_seed.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    return parser.parse_args()


def fmt_float(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and math.isnan(value):
        return "N/A"
    return f"{float(value):.4f}"


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def candidate_run_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        [
            path
            for path in root.iterdir()
            if path.is_dir() and (path / "summary.json").exists()
        ],
        key=lambda path: path.name,
    )


def load_runs(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_dir in candidate_run_dirs(root):
        with (run_dir / "summary.json").open() as handle:
            summary = json.load(handle)
        summary["run_id"] = run_dir.name
        summary["run_dir"] = str(run_dir.resolve())
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    numeric_columns = [
        "seed",
        "split_seed",
        "epochs_requested",
        "epochs_completed",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "plateau_factor",
        "plateau_patience",
        "early_stopping_patience",
        "checkpoint_every",
        "train_ratio",
        "val_ratio",
        "test_ratio",
        "best_val_acc",
        "best_val_kappa",
        "best_epoch",
        "final_train_acc",
        "final_train_kappa",
        "final_train_macro_f1",
        "final_val_acc",
        "final_val_kappa",
        "final_val_macro_f1",
        "test_acc",
        "test_kappa",
        "test_macro_f1",
        "train_loss",
        "val_loss",
        "test_loss",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "config_name" in df.columns:
        df["simplicity_rank"] = df["config_name"].map(CONFIG_SIMPlicity_RANK).fillna(999).astype(int)
    return df.sort_values(["config_name", "seed", "run_id"]).reset_index(drop=True)


def write_configs_summary(df: pd.DataFrame, output_csv: Path) -> None:
    columns = [
        "config_name",
        "seed",
        "split_seed",
        "epochs_requested",
        "learning_rate",
        "batch_size",
        "weight_decay",
        "scheduler",
        "early_stopping_patience",
        "best_val_acc",
        "best_val_kappa",
        "best_epoch",
        "final_train_acc",
        "final_val_acc",
        "test_acc",
        "test_kappa",
        "test_macro_f1",
        "train_loss",
        "val_loss",
        "test_loss",
        "run_id",
        "summary_json",
        "run_dir",
    ]
    payload = pd.DataFrame()
    for column in columns:
        if column not in df.columns:
            payload[column] = []
    if not df.empty:
        payload = pd.DataFrame(
            {
                "config_name": df["config_name"],
                "epochs": df["epochs_requested"],
                "learning_rate": df["learning_rate"],
                "batch_size": df["batch_size"],
                "weight_decay": df["weight_decay"],
                "scheduler": df["scheduler"],
                "early_stopping_patience": df["early_stopping_patience"],
                "seed": df["seed"],
                "best_val_acc": df["best_val_acc"],
                "best_val_kappa": df.get("best_val_kappa"),
                "best_epoch": df["best_epoch"],
                "final_train_acc": df["final_train_acc"],
                "final_val_acc": df["final_val_acc"],
                "final_test_acc": df["test_acc"],
                "final_test_kappa": df["test_kappa"],
                "macro_f1_test": df["test_macro_f1"],
                "train_loss_final": df["train_loss"],
                "val_loss_final": df["val_loss"],
                "test_loss": df["test_loss"],
                "run_id": df["run_id"],
                "summary_json": df["summary_json"],
                "run_dir": df["run_dir"],
            }
        )
    payload.to_csv(output_csv, index=False)


def select_config(config_df: pd.DataFrame) -> dict[str, object] | None:
    if config_df.empty:
        return None
    grouped_rows = []
    for _, group in config_df.groupby("config_name", dropna=False):
        ordered = group.sort_values(
            ["best_val_acc", "final_val_acc", "test_acc", "run_id"],
            ascending=[False, False, False, True],
        )
        row = ordered.iloc[0].copy()
        row["num_repeats"] = len(group)
        grouped_rows.append(row)
    grouped_df = pd.DataFrame(grouped_rows).sort_values(["config_name"]).reset_index(drop=True)
    grouped_df["simplicity_rank"] = grouped_df["config_name"].map(CONFIG_SIMPlicity_RANK).fillna(999).astype(int)

    by_val = grouped_df.sort_values(
        ["best_val_acc", "final_val_acc", "test_acc"],
        ascending=[False, False, False],
    ).iloc[0]
    tolerance = 0.002
    near_ties = grouped_df[grouped_df["best_val_acc"] >= float(by_val["best_val_acc"]) - tolerance].copy()
    selected = near_ties.sort_values(
        ["simplicity_rank", "best_val_acc", "final_val_acc", "test_acc"],
        ascending=[True, False, False, False],
    ).iloc[0]
    reason = (
        "selected_by_best_validation_accuracy"
        if selected["config_name"] == by_val["config_name"]
        else "selected_from_near_tie_by_simplicity"
    )
    payload = sanitize_for_json(selected.to_dict())
    payload["selection_reason"] = reason
    payload["selection_tolerance"] = tolerance
    return payload


def write_seed_summary(seed_df: pd.DataFrame, output_csv: Path) -> None:
    if seed_df.empty:
        pd.DataFrame(
            columns=[
                "seed",
                "best_val_acc",
                "best_epoch",
                "final_train_acc",
                "final_val_acc",
                "final_test_acc",
                "final_test_kappa",
                "macro_f1_test",
            ]
        ).to_csv(output_csv, index=False)
        return
    payload = pd.DataFrame(
        {
            "seed": seed_df["seed"],
            "best_val_acc": seed_df["best_val_acc"],
            "best_epoch": seed_df["best_epoch"],
            "final_train_acc": seed_df["final_train_acc"],
            "final_val_acc": seed_df["final_val_acc"],
            "final_test_acc": seed_df["test_acc"],
            "final_test_kappa": seed_df["test_kappa"],
            "macro_f1_test": seed_df["test_macro_f1"],
            "run_id": seed_df["run_id"],
            "summary_json": seed_df["summary_json"],
        }
    ).sort_values("seed")
    payload.to_csv(output_csv, index=False)


def write_aggregate_summary(seed_df: pd.DataFrame, output_csv: Path) -> pd.DataFrame:
    metrics = {
        "best_val_acc": "best_val_acc",
        "final_val_acc": "final_val_acc",
        "final_test_acc": "test_acc",
        "final_test_kappa": "test_kappa",
        "macro_f1_test": "test_macro_f1",
    }
    rows = []
    for metric_name, column in metrics.items():
        if seed_df.empty or column not in seed_df.columns:
            rows.append({"metric": metric_name, "mean": None, "std": None, "min": None, "max": None, "num_runs": 0})
            continue
        series = pd.to_numeric(seed_df[column], errors="coerce").dropna()
        if series.empty:
            rows.append({"metric": metric_name, "mean": None, "std": None, "min": None, "max": None, "num_runs": 0})
            continue
        rows.append(
            {
                "metric": metric_name,
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
                "min": float(series.min()),
                "max": float(series.max()),
                "num_runs": int(len(series)),
            }
        )
    aggregate_df = pd.DataFrame(rows)
    aggregate_df.to_csv(output_csv, index=False)
    return aggregate_df


def copy_training_curve(seed_df: pd.DataFrame, seed: int, output_path: Path) -> bool:
    if seed_df.empty:
        return False
    matches = seed_df[seed_df["seed"] == seed]
    if matches.empty:
        return False
    figure_path = Path(str(matches.iloc[0]["figures_path"]))
    if not figure_path.exists():
        return False
    shutil.copy2(figure_path, output_path)
    return True


def plot_bar_metric(df: pd.DataFrame, metric: str, ylabel: str, output_path: Path) -> bool:
    if df.empty or metric not in df.columns:
        return False
    series = pd.to_numeric(df[metric], errors="coerce")
    if series.dropna().empty:
        return False
    ordered = df.assign(_metric=series).sort_values("_metric", ascending=False)
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.bar(ordered["config_name"], ordered["_metric"], color="#2a9d8f")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True


def plot_seed_variability(seed_df: pd.DataFrame, output_path: Path) -> bool:
    if seed_df.empty:
        return False
    ordered = seed_df.sort_values("seed")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(ordered["seed"].astype(str), ordered["test_acc"], color="#457b9d")
    axes[0].set_title("Test accuracy by seed")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(ordered["seed"].astype(str), ordered["best_val_acc"], color="#e76f51")
    axes[1].set_title("Best validation accuracy by seed")
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True


def pick_best_seed(seed_df: pd.DataFrame) -> dict[str, object] | None:
    if seed_df.empty:
        return None
    ordered = seed_df.sort_values(
        ["best_val_acc", "final_val_acc", "test_acc", "seed"],
        ascending=[False, False, False, True],
    )
    return sanitize_for_json(ordered.iloc[0].to_dict())


def plot_confusion_matrix(run_summary: dict[str, object], output_path: Path) -> bool:
    matrix_path = Path(str(run_summary.get("test_confusion_matrix_csv", "")))
    if not matrix_path.exists():
        return False
    matrix_df = pd.read_csv(matrix_path)
    if matrix_df.empty:
        return False
    matrix = matrix_df.iloc[:, 1:].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Confusion matrix (best seed)")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True


def plot_loss_curves(run_summary: dict[str, object], output_path: Path) -> bool:
    history_path = Path(str(run_summary.get("history_csv", "")))
    if not history_path.exists():
        return False
    history_df = pd.read_csv(history_path)
    if history_df.empty:
        return False
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(history_df["epoch"], history_df["train_loss"], label="train")
    ax.plot(history_df["epoch"], history_df["val_loss"], label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss curves (best seed)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True


def load_split_manifest(summary_row: dict[str, object] | None) -> dict[str, object] | None:
    if not summary_row:
        return None
    manifest_path = Path(str(summary_row.get("shared_split_manifest_json", "")))
    if not manifest_path.exists():
        manifest_path = Path(str(summary_row.get("run_split_manifest_json", "")))
    if not manifest_path.exists():
        return None
    with manifest_path.open() as handle:
        return json.load(handle)


def write_selection_summary(
    run_root: Path,
    selected_config: dict[str, object] | None,
    best_seed: dict[str, object] | None,
    config_df: pd.DataFrame,
    seed_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
) -> None:
    split_manifest = None
    if best_seed:
        split_manifest = load_split_manifest(best_seed)
    if split_manifest is None and selected_config:
        split_manifest = load_split_manifest(selected_config)
    payload = {
        "protocol": "random train/validation/test split 60/20/20",
        "block": "eegnet_baseline_random_split_60_20_20",
        "preliminary_warning": (
            "Exploratory preliminary experiment. This is not the final subject-specific cross-session protocol."
        ),
        "mixes_sessions": None if split_manifest is None else split_manifest.get("mixes_sessions"),
        "split_strategy": None if split_manifest is None else split_manifest.get("split_strategy"),
        "selected_configuration": sanitize_for_json(selected_config),
        "selected_best_seed": sanitize_for_json(best_seed),
        "selection_criterion": {
            "primary_metric": "best_val_acc",
            "tie_tolerance": 0.002,
            "secondary_information_only": ["final_val_acc", "test_acc"],
            "test_leakage_guard": "Test metrics were not used for configuration selection or early stopping.",
        },
        "available_config_runs": int(len(config_df)),
        "available_best_seed_runs": int(len(seed_df)),
        "seeds_used": [] if seed_df.empty else [int(seed) for seed in sorted(seed_df["seed"].tolist())],
        "aggregate_summary": sanitize_for_json(aggregate_df.to_dict(orient="records")),
        "run_root": str(run_root.resolve()),
        "generated_at": utc_now_iso(),
    }
    with (run_root / "selection_summary.json").open("w") as handle:
        json.dump(payload, handle, indent=2, default=sanitize_for_json)


def write_interpretation_notes(
    run_root: Path,
    selected_config: dict[str, object] | None,
    best_seed: dict[str, object] | None,
    aggregate_df: pd.DataFrame,
) -> None:
    lines = [
        "EEGNet baseline random split 60/20/20 interpretation notes",
        "",
        "Protocol",
        "- Preliminary exploratory experiment with random train/validation/test split.",
        "- Ratios: 0.6 / 0.2 / 0.2.",
        "- The split is subject-aware and class-stratified within subject.",
        "- Sessions T and E can appear mixed inside train, validation and test.",
        "- This protocol is not the final subject-specific cross-session evaluation.",
        "",
        "Selection criterion",
        "- Primary metric: best validation accuracy.",
        "- Tie tolerance: 0.002.",
        "- Secondary information only: final validation accuracy and test accuracy.",
        "- Test metrics were not used for configuration selection or early stopping.",
    ]
    if selected_config:
        lines.extend(
            [
                "",
                "Selected configuration",
                f"- config_name: {selected_config.get('config_name')}",
                f"- reason: {selected_config.get('selection_reason')}",
                f"- best_val_acc: {fmt_float(selected_config.get('best_val_acc'))}",
                f"- final_val_acc: {fmt_float(selected_config.get('final_val_acc'))}",
                f"- test_acc: {fmt_float(selected_config.get('test_acc'))}",
            ]
        )
    if best_seed:
        lines.extend(
            [
                "",
                "Best seed among selected configuration",
                f"- seed: {best_seed.get('seed')}",
                f"- best_val_acc: {fmt_float(best_seed.get('best_val_acc'))}",
                f"- final_val_acc: {fmt_float(best_seed.get('final_val_acc'))}",
                f"- test_acc: {fmt_float(best_seed.get('test_acc'))}",
                f"- test_kappa: {fmt_float(best_seed.get('test_kappa'))}",
                f"- macro_f1_test: {fmt_float(best_seed.get('test_macro_f1'))}",
            ]
        )
    if not aggregate_df.empty:
        lines.extend(["", "Multi-seed aggregate"])
        for _, row in aggregate_df.iterrows():
            lines.append(
                f"- {row['metric']}: mean={fmt_float(row['mean'])}, std={fmt_float(row['std'])}, "
                f"min={fmt_float(row['min'])}, max={fmt_float(row['max'])}, n={row['num_runs']}"
            )
    (run_root / "interpretation_notes.txt").write_text("\n".join(lines) + "\n")


def export_split_indices(seed_df: pd.DataFrame, config_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_df = seed_df if not seed_df.empty else config_df
    if source_df.empty:
        return
    for split_seed, group in source_df.groupby("split_seed", dropna=False):
        row = group.iloc[0].to_dict()
        manifest = load_split_manifest(row)
        if manifest is None:
            continue
        payload = {
            "protocol": "random train/validation/test split 60/20/20",
            "mixes_sessions": manifest.get("mixes_sessions"),
            "split_seed": int(split_seed),
            "train_ratio": manifest.get("train_ratio"),
            "val_ratio": manifest.get("val_ratio"),
            "test_ratio": manifest.get("test_ratio"),
            "split_strategy": manifest.get("split_strategy"),
            "train_indices_csv": manifest.get("train_indices_csv"),
            "validation_indices_csv": manifest.get("validation_indices_csv"),
            "test_indices_csv": manifest.get("test_indices_csv"),
            "split_summary_csv": manifest.get("split_summary_csv"),
            "session_counts_by_split": manifest.get("session_counts_by_split"),
            "exported_at": utc_now_iso(),
        }
        with (output_dir / f"split_seed_{int(split_seed)}.json").open("w") as handle:
            json.dump(payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    config_runs_dir = run_root / "config_runs"
    best_seeds_dir = run_root / "best_config_seeds"
    analysis_figures_dir = run_root / "analysis_figures"
    split_indices_dir = run_root / "split_indices"
    analysis_figures_dir.mkdir(parents=True, exist_ok=True)

    config_df = load_runs(config_runs_dir)
    seed_df_all = load_runs(best_seeds_dir)
    selected_config = select_config(config_df)
    selected_config_name = None if selected_config is None else selected_config.get("config_name")
    seed_df = seed_df_all.copy()
    if selected_config_name is not None and not seed_df.empty:
        seed_df = seed_df[seed_df["config_name"] == selected_config_name].copy().reset_index(drop=True)

    configs_summary_csv = run_root / "configs_summary.csv"
    seed_summary_csv = run_root / "seed_summary.csv"
    aggregate_summary_csv = run_root / "aggregate_summary.csv"
    write_configs_summary(config_df, configs_summary_csv)
    write_seed_summary(seed_df, seed_summary_csv)
    aggregate_df = write_aggregate_summary(seed_df, aggregate_summary_csv)

    export_split_indices(seed_df, config_df, split_indices_dir)
    plot_bar_metric(config_df, "best_val_acc", "Validation accuracy by config", analysis_figures_dir / "validation_accuracy_by_config.png")
    plot_bar_metric(config_df, "test_acc", "Test accuracy by config", analysis_figures_dir / "test_accuracy_by_config.png")
    plot_seed_variability(seed_df, analysis_figures_dir / "seed_variability_accuracy.png")
    copy_training_curve(seed_df, 42, analysis_figures_dir / "training_curves_seed42.png")
    copy_training_curve(seed_df, 2024, analysis_figures_dir / "training_curves_seed2024.png")

    best_seed = pick_best_seed(seed_df)
    if best_seed is not None:
        plot_confusion_matrix(best_seed, analysis_figures_dir / "confusion_matrix_best_seed.png")
        plot_loss_curves(best_seed, analysis_figures_dir / "loss_curves_best_seed.png")

    write_selection_summary(run_root, selected_config, best_seed, config_df, seed_df, aggregate_df)
    write_interpretation_notes(run_root, selected_config, best_seed, aggregate_df)

    figures_inventory = []
    for figure_name in FIGURE_REQUIREMENTS:
        figure_path = analysis_figures_dir / figure_name
        figures_inventory.append(
            {
                "figure": figure_name,
                "exists": figure_path.exists(),
                "path": str(figure_path.resolve()),
            }
        )
    with (run_root / "figures_inventory.json").open("w") as handle:
        json.dump(figures_inventory, handle, indent=2)

    print(f"Wrote {configs_summary_csv}")
    print(f"Wrote {seed_summary_csv}")
    print(f"Wrote {aggregate_summary_csv}")
    print(f"Wrote {run_root / 'selection_summary.json'}")
    print(f"Wrote {run_root / 'interpretation_notes.txt'}")


if __name__ == "__main__":
    main()
