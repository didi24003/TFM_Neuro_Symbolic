#!/usr/bin/env python
"""Summarize EEGNet baseline subject-session runs and export the best bundle."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tarfile
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_ROOT = SCRIPT_DIR / "runs" / "eegnet_baseline_subject_session"
CONFIG_RUN_ROOT = RUN_ROOT / "config_runs"
BEST_SEEDS_RUN_ROOT = RUN_ROOT / "best_config_seeds"
ANALYSIS_DIR = RUN_ROOT / "analysis_figures"
EXCLUDED_RUN_PREFIXES = ("smoke_",)

MPLCONFIG_DIR = RUN_ROOT / ".matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONFIG_SIMPLICITY_RANK = {
    "baseline_short": 0,
    "lr_low": 1,
    "lr_high": 2,
    "weight_decay": 3,
    "scheduler_plateau": 4,
    "long_training_300": 5,
    "long_training_500": 6,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    return parser.parse_args()


def fmt_float(value: object) -> str:
    if value is None:
        return "--"
    if isinstance(value, float) and math.isnan(value):
        return "--"
    return f"{float(value):.4f}"


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def candidate_run_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        [
            path
            for path in root.iterdir()
            if path.is_dir()
            and (path / "summary.json").exists()
            and not path.name.startswith(EXCLUDED_RUN_PREFIXES)
        ],
        key=lambda path: path.name,
    )


def load_run_summaries(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_dir in candidate_run_dirs(root):
        with (run_dir / "summary.json").open() as f:
            summary = json.load(f)
        summary["run_id"] = run_dir.name
        summary["run_dir"] = str(run_dir)
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    numeric_columns = [
        "seed",
        "split_seed",
        "epochs_requested",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "plateau_factor",
        "plateau_patience",
        "early_stopping_patience",
        "checkpoint_every",
        "val_ratio",
        "mean_best_val_acc",
        "std_best_val_acc",
        "mean_best_val_kappa",
        "mean_final_val_acc",
        "mean_test_acc",
        "std_test_acc",
        "mean_test_kappa",
        "mean_macro_f1",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.sort_values(["config_name", "seed", "run_id"]).reset_index(drop=True)


def load_subject_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        path = Path(str(row["subject_metrics_csv"]))
        if not path.exists():
            continue
        subject_df = pd.read_csv(path)
        subject_df["run_id"] = row["run_id"]
        subject_df["config_name"] = row["config_name"]
        subject_df["seed"] = row["seed"]
        rows.append(subject_df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def aggregate_configs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    grouped = (
        df.groupby("config_name", as_index=False)
        .agg(
            num_runs=("run_id", "count"),
            seeds=("seed", lambda s: " ".join(str(int(x)) for x in sorted(set(s.tolist())))),
            mean_best_val_acc=("mean_best_val_acc", "mean"),
            mean_best_val_kappa=("mean_best_val_kappa", "mean"),
            mean_final_val_acc=("mean_final_val_acc", "mean"),
            mean_test_acc=("mean_test_acc", "mean"),
            mean_test_kappa=("mean_test_kappa", "mean"),
            mean_macro_f1=("mean_macro_f1", "mean"),
            epochs_requested=("epochs_requested", "first"),
            batch_size=("batch_size", "first"),
            learning_rate=("learning_rate", "first"),
            weight_decay=("weight_decay", "first"),
            scheduler=("scheduler", "first"),
            plateau_patience=("plateau_patience", "first"),
            early_stopping_patience=("early_stopping_patience", "first"),
        )
    )
    grouped["simplicity_rank"] = grouped["config_name"].map(CONFIG_SIMPLICITY_RANK).fillna(999).astype(int)
    return grouped.sort_values(["mean_best_val_acc", "mean_best_val_kappa"], ascending=[False, False]).reset_index(drop=True)


def write_configs_summary(df: pd.DataFrame, output_csv: Path) -> None:
    columns = [
        "config_name",
        "num_runs",
        "seeds",
        "mean_best_val_acc",
        "mean_best_val_kappa",
        "mean_final_val_acc",
        "mean_test_acc",
        "mean_test_kappa",
        "mean_macro_f1",
        "epochs_requested",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "scheduler",
        "plateau_patience",
        "early_stopping_patience",
    ]
    df[columns].to_csv(output_csv, index=False)


def write_configs_latex(df: pd.DataFrame, output_tex: Path) -> None:
    lines = [
        "% Auto-generated by summarize_eegnet_baseline_subject_session.py",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Configuracion & Mean best val. acc. & Mean test acc. & Mean test kappa & Epochs & LR \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['config_name']} & {fmt_float(row['mean_best_val_acc'])} & {fmt_float(row['mean_test_acc'])} & "
            f"{fmt_float(row['mean_test_kappa'])} & {int(row['epochs_requested'])} & {row['learning_rate']:.4g} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    output_tex.write_text("\n".join(lines))


def choose_representative_run(config_runs_df: pd.DataFrame, config_name: str) -> dict[str, object]:
    subset = config_runs_df[config_runs_df["config_name"] == config_name].copy()
    if subset.empty:
        raise FileNotFoundError(f"No runs found for config {config_name}")
    ordered = subset.sort_values(
        ["mean_best_val_acc", "mean_best_val_kappa", "mean_final_val_acc", "run_id"],
        ascending=[False, False, False, True],
    )
    return ordered.iloc[0].to_dict()


def pick_best_config(config_df: pd.DataFrame, config_runs_df: pd.DataFrame) -> dict[str, object]:
    if config_df.empty:
        raise FileNotFoundError("No config runs found.")
    ordered = config_df.sort_values(
        ["mean_best_val_acc", "mean_best_val_kappa", "simplicity_rank", "mean_final_val_acc"],
        ascending=[False, False, True, False],
    )
    selected = ordered.iloc[0].to_dict()
    selected["representative_run"] = sanitize_for_json(choose_representative_run(config_runs_df, str(selected["config_name"])))
    return {
        "selection_protocol": {
            "primary_metric": "mean_best_val_acc",
            "secondary_metric": "mean_best_val_kappa",
            "test_leakage_guard": "Session E test metrics were excluded from configuration selection and early stopping.",
        },
        "selected_config": sanitize_for_json(selected),
        "best_config_by_validation": sanitize_for_json(ordered.iloc[0].to_dict()),
        "selection_reason": "Selected by highest mean_best_val_acc across subjects using only validation from session T.",
    }


def plot_config_metrics(df: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("mean_best_val_acc", "Configuration vs mean best validation accuracy", "config_vs_mean_best_val_acc.png"),
        ("mean_test_acc", "Configuration vs mean test accuracy", "config_vs_mean_test_acc.png"),
        ("mean_test_kappa", "Configuration vs mean test kappa", "config_vs_mean_test_kappa.png"),
    ]
    order = df.sort_values("mean_best_val_acc", ascending=False)
    for column, title, filename in metrics:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.bar(order["config_name"], order[column], color="#1d3557")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=200)
        plt.close(fig)


def plot_subject_metric_by_config(
    subject_df: pd.DataFrame,
    value_column: str,
    title: str,
    filename: str,
    output_dir: Path,
) -> None:
    if subject_df.empty:
        return
    pivot = (
        subject_df.groupby(["subject_id", "config_name"], as_index=False)[value_column]
        .mean()
        .pivot(index="subject_id", columns="config_name", values=value_column)
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)


def plot_best_config_training_curves(representative_run: dict[str, object], output_dir: Path) -> None:
    run_dir = Path(str(representative_run["run_dir"]))
    subject_dirs = sorted((run_dir / "subjects").glob("*"))
    if not subject_dirs:
        return
    fig, axes = plt.subplots(len(subject_dirs), 2, figsize=(12, max(4, 3 * len(subject_dirs))), squeeze=False)
    for row_axes, subject_dir in zip(axes, subject_dirs):
        history_path = subject_dir / "history.csv"
        if not history_path.exists():
            continue
        history = pd.read_csv(history_path)
        epochs = history["epoch"]
        subject_id = subject_dir.name
        row_axes[0].plot(epochs, history["train_loss"], label="train")
        row_axes[0].plot(epochs, history["val_loss"], label="validation")
        row_axes[0].set_title(f"{subject_id} loss")
        row_axes[0].grid(alpha=0.3)
        row_axes[0].legend(frameon=False)

        row_axes[1].plot(epochs, history["train_acc"], label="train")
        row_axes[1].plot(epochs, history["val_acc"], label="validation")
        row_axes[1].set_title(f"{subject_id} accuracy")
        row_axes[1].grid(alpha=0.3)
        row_axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "best_config_training_curves.png", dpi=200)
    plt.close(fig)


def summarize_best_seed_runs(seed_df: pd.DataFrame, selected_config_name: str, output_csv: Path, output_tex: Path) -> pd.DataFrame:
    filtered = seed_df[seed_df["config_name"] == selected_config_name].copy()
    columns = [
        "config_name",
        "run_id",
        "seed",
        "split_seed",
        "mean_best_val_acc",
        "mean_best_val_kappa",
        "mean_final_val_acc",
        "mean_test_acc",
        "mean_test_kappa",
        "mean_macro_f1",
        "summary_json",
        "subject_metrics_csv",
    ]
    if filtered.empty:
        pd.DataFrame(columns=columns).to_csv(output_csv, index=False)
        output_tex.write_text("% No best-seed runs found.\n")
        return filtered
    filtered[columns].sort_values("seed").to_csv(output_csv, index=False)

    lines = [
        "% Auto-generated by summarize_eegnet_baseline_subject_session.py",
        "\\begin{tabular}{rrrrrr}",
        "\\toprule",
        "Seed & Mean best val. acc. & Mean val. kappa & Mean test acc. & Mean test kappa & Mean macro-F1 \\\\",
        "\\midrule",
    ]
    ordered = filtered.sort_values("seed")
    for _, row in ordered.iterrows():
        lines.append(
            f"{int(row['seed'])} & {fmt_float(row['mean_best_val_acc'])} & {fmt_float(row['mean_best_val_kappa'])} & "
            f"{fmt_float(row['mean_test_acc'])} & {fmt_float(row['mean_test_kappa'])} & {fmt_float(row['mean_macro_f1'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    output_tex.write_text("\n".join(lines))
    return ordered


def plot_best_seed_results(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, column, title in [
        (axes[0], "mean_best_val_acc", "Best validation accuracy by seed"),
        (axes[1], "mean_test_acc", "Test accuracy by seed"),
        (axes[2], "mean_test_kappa", "Test kappa by seed"),
    ]:
        ax.bar(df["seed"].astype(str), df[column], color="#e76f51")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "best_config_seed_results.png", dpi=200)
    plt.close(fig)


def write_root_subject_metrics(seed_subject_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    if seed_subject_df.empty:
        pd.DataFrame(
            columns=["subject_id", "mean_test_acc", "std_test_acc", "mean_test_kappa", "std_test_kappa", "num_runs"]
        ).to_csv(output_path, index=False)
        return pd.DataFrame()
    summary = (
        seed_subject_df.groupby("subject_id", as_index=False)
        .agg(
            mean_test_acc=("test_acc", "mean"),
            std_test_acc=("test_acc", "std"),
            mean_test_kappa=("test_kappa", "mean"),
            std_test_kappa=("test_kappa", "std"),
            num_runs=("run_id", "nunique"),
        )
        .sort_values("subject_id")
        .reset_index(drop=True)
    )
    summary.to_csv(output_path, index=False)
    return summary


def write_interpretation_notes(
    best_info: dict[str, object],
    config_df: pd.DataFrame,
    best_seed_df: pd.DataFrame,
    output_path: Path,
) -> None:
    selected = best_info["selected_config"]
    lines = [
        "EEGNet baseline subject-session interpretation notes",
        "",
        "Methodology",
        "- Main protocol: subject-specific cross-session.",
        "- For every subject, session T was split into train/validation.",
        "- Session E was reserved as final test only.",
        "- Configuration selection used only mean_best_val_acc across subjects.",
        "- Session E metrics were excluded from early stopping and model selection.",
        "",
        "Selected configuration",
        f"- config_name: {selected['config_name']}",
        f"- mean_best_val_acc: {fmt_float(selected['mean_best_val_acc'])}",
        f"- mean_best_val_kappa: {fmt_float(selected['mean_best_val_kappa'])}",
        f"- mean_test_acc (reference only): {fmt_float(selected['mean_test_acc'])}",
        "",
        "Compared configurations",
    ]
    for _, row in config_df.sort_values("mean_best_val_acc", ascending=False).iterrows():
        lines.append(
            f"- {row['config_name']}: mean_best_val_acc={fmt_float(row['mean_best_val_acc'])}, "
            f"mean_test_acc={fmt_float(row['mean_test_acc'])}, mean_test_kappa={fmt_float(row['mean_test_kappa'])}"
        )
    if not best_seed_df.empty:
        lines.extend(
            [
                "",
                "Best-config multi-seed aggregate",
                f"- seeds evaluated: {', '.join(str(int(seed)) for seed in best_seed_df['seed'].tolist())}",
                f"- mean(mean_best_val_acc): {fmt_float(best_seed_df['mean_best_val_acc'].mean())}",
                f"- mean(mean_test_acc): {fmt_float(best_seed_df['mean_test_acc'].mean())}",
                f"- std(mean_test_acc): {fmt_float(best_seed_df['mean_test_acc'].std())}",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n")


def backup_existing_path(path: Path) -> None:
    if not path.exists():
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.name}_previous_{timestamp}")
    shutil.move(path, backup_path)


def build_best_export(
    best_info: dict[str, object],
    best_seed_df: pd.DataFrame,
    config_summary_csv: Path,
    config_latex: Path,
    best_config_json: Path,
    best_seeds_summary_csv: Path,
    best_seeds_latex: Path,
    root_subject_metrics_csv: Path,
    interpretation_notes: Path,
    export_dir: Path,
    export_tar: Path,
) -> None:
    representative_run_dir = Path(str(best_info["selected_config"]["representative_run"]["run_dir"]))
    backup_existing_path(export_dir)
    backup_existing_path(export_tar)
    export_dir.mkdir(parents=True, exist_ok=True)

    if representative_run_dir.exists():
        shutil.copytree(representative_run_dir, export_dir / "selected_config_run")

    seeds_dir = export_dir / "best_config_seeds"
    seeds_dir.mkdir(exist_ok=True)
    for _, row in best_seed_df.iterrows():
        run_dir = Path(str(row["run_dir"]))
        if run_dir.exists():
            shutil.copytree(run_dir, seeds_dir / run_dir.name)

    if ANALYSIS_DIR.exists():
        shutil.copytree(ANALYSIS_DIR, export_dir / "analysis_figures")

    for path in [
        config_summary_csv,
        config_latex,
        best_config_json,
        best_seeds_summary_csv,
        best_seeds_latex,
        root_subject_metrics_csv,
        interpretation_notes,
    ]:
        if path.exists():
            shutil.copy2(path, export_dir / path.name)

    with tarfile.open(export_tar, "w:gz") as tar:
        tar.add(export_dir, arcname=export_dir.name)


def main() -> None:
    args = parse_args()
    run_root = args.run_root
    config_runs_root = run_root / "config_runs"
    best_seeds_root = run_root / "best_config_seeds"
    analysis_dir = run_root / "analysis_figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    config_summary_csv = run_root / "baseline_subject_session_configs_summary.csv"
    config_latex = run_root / "baseline_subject_session_configs_latex_table.txt"
    best_config_json = run_root / "baseline_subject_session_best_config_summary.json"
    best_seeds_summary_csv = run_root / "baseline_subject_session_best_seeds_summary.csv"
    best_seeds_latex = run_root / "baseline_subject_session_best_seeds_latex_table.txt"
    root_subject_metrics_csv = run_root / "subject_metrics.csv"
    interpretation_notes = run_root / "interpretation_notes.txt"
    best_export_dir = run_root / "best_export"
    best_export_tar = run_root / "best_export_eegnet_baseline_subject_session.tar.gz"

    config_runs_df = load_run_summaries(config_runs_root)
    if config_runs_df.empty:
        raise FileNotFoundError(f"No config runs found under {config_runs_root}")
    config_subject_df = load_subject_metrics(config_runs_df)
    config_df = aggregate_configs(config_runs_df)

    write_configs_summary(config_df, config_summary_csv)
    write_configs_latex(config_df, config_latex)
    plot_config_metrics(config_df, analysis_dir)
    plot_subject_metric_by_config(
        config_subject_df,
        "test_acc",
        "Accuracy by subject for each configuration",
        "subject_accuracy_by_config.png",
        analysis_dir,
    )
    plot_subject_metric_by_config(
        config_subject_df,
        "test_kappa",
        "Kappa by subject for each configuration",
        "subject_kappa_by_config.png",
        analysis_dir,
    )

    best_info = pick_best_config(config_df, config_runs_df)
    with best_config_json.open("w") as f:
        json.dump(best_info, f, indent=2)

    representative_run = best_info["selected_config"]["representative_run"]
    plot_best_config_training_curves(representative_run, analysis_dir)

    best_seed_runs_df = load_run_summaries(best_seeds_root)
    best_seed_df = summarize_best_seed_runs(
        best_seed_runs_df,
        str(best_info["selected_config"]["config_name"]),
        best_seeds_summary_csv,
        best_seeds_latex,
    )
    plot_best_seed_results(best_seed_df, analysis_dir)
    best_seed_subject_df = load_subject_metrics(best_seed_df)
    write_root_subject_metrics(best_seed_subject_df, root_subject_metrics_csv)
    write_interpretation_notes(best_info, config_df, best_seed_df, interpretation_notes)
    build_best_export(
        best_info=best_info,
        best_seed_df=best_seed_df,
        config_summary_csv=config_summary_csv,
        config_latex=config_latex,
        best_config_json=best_config_json,
        best_seeds_summary_csv=best_seeds_summary_csv,
        best_seeds_latex=best_seeds_latex,
        root_subject_metrics_csv=root_subject_metrics_csv,
        interpretation_notes=interpretation_notes,
        export_dir=best_export_dir,
        export_tar=best_export_tar,
    )


if __name__ == "__main__":
    main()
