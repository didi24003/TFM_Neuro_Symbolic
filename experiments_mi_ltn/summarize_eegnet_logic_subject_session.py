#!/usr/bin/env python
"""Summarize EEGNet logic subject-session runs and compare against baseline."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RUN_ROOT = SCRIPT_DIR / "runs" / "eegnet_logic_subject_session"
LOGIC_RUN_ROOT = RUN_ROOT / "logic_runs"
BASELINE_RUN_ROOT = SCRIPT_DIR / "runs" / "eegnet_baseline_subject_session"
BASELINE_BEST_CONFIG_JSON = BASELINE_RUN_ROOT / "baseline_subject_session_best_config_summary.json"
BASELINE_BEST_SEEDS_CSV = BASELINE_RUN_ROOT / "baseline_subject_session_best_seeds_summary.csv"
ANALYSIS_DIR = RUN_ROOT / "analysis_figures"
EXCLUDED_RUN_PREFIXES = ("smoke_",)
LAMBDA_SWEEP = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
CHANNEL_ORDER = [
    "Fz",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "P1",
    "Pz",
    "P2",
    "POz",
]
SENSORIMOTOR_CHANNELS = ["FC3", "FC4", "FCz", "C3", "C4", "Cz", "CP3", "CP4", "CPz"]
POSTERIOR_CHANNELS = ["P1", "Pz", "P2", "POz"]
FLOAT_TOL = 1e-6

MPLCONFIG_DIR = RUN_ROOT / ".matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    matplotlib = None
    plt = None

try:
    import mne
except ModuleNotFoundError:
    mne = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--logic-run-root", type=Path, default=LOGIC_RUN_ROOT)
    parser.add_argument("--baseline-run-root", type=Path, default=BASELINE_RUN_ROOT)
    return parser.parse_args()


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def fmt_float(value: object) -> str:
    if value is None:
        return "--"
    if isinstance(value, float) and math.isnan(value):
        return "--"
    return f"{float(value):.4f}"


def relative_to_repo(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


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


def load_json(path: Path) -> dict[str, object]:
    with path.open() as f:
        return json.load(f)


def load_logic_runs(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_dir in candidate_run_dirs(root):
        summary = load_json(run_dir / "summary.json")
        summary["run_id"] = run_dir.name
        summary["run_dir"] = str(run_dir)
        rows.append(summary)
    if not rows:
        raise FileNotFoundError(f"No logic runs found in {root}")
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
        "lambda_rule",
        "target_sm_mean",
        "target_post_mean",
        "margin_mean",
        "mean_best_val_acc",
        "std_best_val_acc",
        "mean_best_val_kappa",
        "mean_final_val_acc",
        "mean_test_acc",
        "std_test_acc",
        "mean_test_kappa",
        "mean_macro_f1",
        "mean_R_SM_mean",
        "mean_R_POST_mean",
        "mean_R_OTHER_mean",
        "mean_R_SM_mean_minus_POST_mean",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.sort_values(["model_type", "lambda_rule", "seed", "run_id"]).reset_index(drop=True)


def load_baseline_runs(baseline_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    best_json = baseline_root / "baseline_subject_session_best_config_summary.json"
    best_seeds_csv = baseline_root / "baseline_subject_session_best_seeds_summary.csv"
    if not best_json.exists():
        raise FileNotFoundError(f"Missing baseline selected-config JSON: {best_json}")
    if not best_seeds_csv.exists():
        raise FileNotFoundError(f"Missing baseline best-seeds CSV: {best_seeds_csv}")

    baseline_config = load_json(best_json)
    runs_df = pd.read_csv(best_seeds_csv).copy()
    for column in ["seed", "split_seed", "mean_best_val_acc", "mean_best_val_kappa", "mean_final_val_acc", "mean_test_acc", "mean_test_kappa", "mean_macro_f1"]:
        if column in runs_df.columns:
            runs_df[column] = pd.to_numeric(runs_df[column], errors="coerce")

    subject_rows = []
    for _, row in runs_df.iterrows():
        path = REPO_ROOT / Path(str(row["subject_metrics_csv"]))
        subject_df = pd.read_csv(path).copy()
        subject_df["seed"] = int(row["seed"])
        subject_df["split_seed"] = int(row["split_seed"])
        subject_df["run_id"] = str(row["run_id"])
        subject_rows.append(subject_df)
    subjects_df = pd.concat(subject_rows, ignore_index=True)
    numeric_cols = [
        "train_size",
        "validation_size",
        "test_size",
        "best_val_acc",
        "best_val_kappa",
        "best_epoch",
        "final_train_acc",
        "final_val_acc",
        "test_acc",
        "test_loss",
        "test_kappa",
        "macro_f1",
    ]
    for column in numeric_cols:
        if column in subjects_df.columns:
            subjects_df[column] = pd.to_numeric(subjects_df[column], errors="coerce")
    return runs_df, subjects_df, baseline_config


def load_logic_subject_metrics(logic_runs_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in logic_runs_df.iterrows():
        path = Path(str(row["subject_metrics_csv"]))
        subject_df = pd.read_csv(path).copy()
        subject_df["run_id"] = row["run_id"]
        subject_df["model_type"] = row["model_type"]
        subject_df["lambda_rule"] = float(row["lambda_rule"])
        subject_df["seed"] = int(row["seed"])
        subject_df["split_seed"] = int(row["split_seed"])
        rows.append(subject_df)
    df = pd.concat(rows, ignore_index=True)
    numeric_cols = [
        "lambda_rule",
        "seed",
        "split_seed",
        "train_size",
        "validation_size",
        "test_size",
        "best_val_acc",
        "best_val_kappa",
        "best_epoch",
        "final_train_acc",
        "final_val_acc",
        "test_acc",
        "test_loss",
        "test_kappa",
        "macro_f1",
        "R_SM_sum",
        "R_POST_sum",
        "R_OTHER_sum",
        "R_SM_mean",
        "R_POST_mean",
        "R_OTHER_mean",
        "R_SM_mean_minus_POST_mean",
    ]
    for column in numeric_cols:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def aggregate_logic_conditions(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["model_type", "lambda_rule"], as_index=False)
        .agg(
            num_runs=("run_id", "count"),
            num_seeds=("seed", "nunique"),
            mean_best_val_acc=("mean_best_val_acc", "mean"),
            std_best_val_acc=("mean_best_val_acc", "std"),
            mean_best_val_kappa=("mean_best_val_kappa", "mean"),
            mean_final_val_acc=("mean_final_val_acc", "mean"),
            mean_test_acc=("mean_test_acc", "mean"),
            std_test_acc=("mean_test_acc", "std"),
            mean_test_kappa=("mean_test_kappa", "mean"),
            std_test_kappa=("mean_test_kappa", "std"),
            mean_macro_f1=("mean_macro_f1", "mean"),
            mean_R_SM_mean=("mean_R_SM_mean", "mean"),
            mean_R_POST_mean=("mean_R_POST_mean", "mean"),
            mean_R_OTHER_mean=("mean_R_OTHER_mean", "mean"),
            mean_R_SM_mean_minus_POST_mean=("mean_R_SM_mean_minus_POST_mean", "mean"),
        )
        .sort_values(["model_type", "lambda_rule"])
        .reset_index(drop=True)
    )
    return agg


def select_best_rule_lambda(agg: pd.DataFrame) -> tuple[float, pd.Series]:
    rule_agg = agg[(agg["model_type"] == "eegnet_gates_rule") & (agg["lambda_rule"] > 0.0)].copy()
    if rule_agg.empty:
        raise ValueError("No eegnet_gates_rule runs with lambda_rule > 0 were found.")
    rule_agg["lambda_simplicity"] = rule_agg["lambda_rule"]
    rule_agg = rule_agg.sort_values(
        ["mean_best_val_acc", "mean_best_val_kappa", "mean_final_val_acc", "lambda_simplicity"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return float(rule_agg.iloc[0]["lambda_rule"]), rule_agg.iloc[0]


def aggregate_baseline_subjects(subject_df: pd.DataFrame) -> pd.DataFrame:
    return (
        subject_df.groupby("subject_id", as_index=False)
        .agg(
            baseline_test_acc=("test_acc", "mean"),
            baseline_test_kappa=("test_kappa", "mean"),
            baseline_best_val_acc=("best_val_acc", "mean"),
            baseline_best_val_kappa=("best_val_kappa", "mean"),
        )
        .sort_values("subject_id")
    )


def aggregate_logic_subjects(subject_df: pd.DataFrame, model_type: str, lambda_rule: float, prefix: str) -> pd.DataFrame:
    subset = subject_df[
        (subject_df["model_type"] == model_type)
        & (np.isclose(subject_df["lambda_rule"], lambda_rule, atol=FLOAT_TOL))
    ].copy()
    if subset.empty:
        raise ValueError(f"No subject metrics found for model_type={model_type} lambda_rule={lambda_rule}")
    return (
        subset.groupby("subject_id", as_index=False)
        .agg(
            **{
                f"{prefix}_test_acc": ("test_acc", "mean"),
                f"{prefix}_test_kappa": ("test_kappa", "mean"),
                f"{prefix}_best_val_acc": ("best_val_acc", "mean"),
                f"{prefix}_best_val_kappa": ("best_val_kappa", "mean"),
                "R_SM_sum": ("R_SM_sum", "mean"),
                "R_POST_sum": ("R_POST_sum", "mean"),
                "R_OTHER_sum": ("R_OTHER_sum", "mean"),
                "R_SM_mean": ("R_SM_mean", "mean"),
                "R_POST_mean": ("R_POST_mean", "mean"),
                "R_OTHER_mean": ("R_OTHER_mean", "mean"),
                "R_SM_mean_minus_POST_mean": ("R_SM_mean_minus_POST_mean", "mean"),
            }
        )
        .sort_values("subject_id")
    )


def build_subject_comparison(
    baseline_subjects: pd.DataFrame,
    gates_subjects: pd.DataFrame,
    rule_subjects: pd.DataFrame,
) -> pd.DataFrame:
    df = baseline_subjects.merge(gates_subjects, on="subject_id", how="inner").merge(rule_subjects, on="subject_id", how="inner", suffixes=("", "_rule"))
    df["delta_gates_vs_baseline_acc"] = df["gates_no_rule_test_acc"] - df["baseline_test_acc"]
    df["delta_rule_vs_baseline_acc"] = df["gates_rule_test_acc"] - df["baseline_test_acc"]
    df["delta_rule_vs_gates_acc"] = df["gates_rule_test_acc"] - df["gates_no_rule_test_acc"]
    df["delta_gates_vs_baseline_kappa"] = df["gates_no_rule_test_kappa"] - df["baseline_test_kappa"]
    df["delta_rule_vs_baseline_kappa"] = df["gates_rule_test_kappa"] - df["baseline_test_kappa"]
    df["delta_rule_vs_gates_kappa"] = df["gates_rule_test_kappa"] - df["gates_no_rule_test_kappa"]
    df["R_SM_mean"] = df["R_SM_mean_rule"]
    df["R_POST_mean"] = df["R_POST_mean_rule"]
    df["R_OTHER_mean"] = df["R_OTHER_mean_rule"]
    df["R_SM_mean_minus_POST_mean"] = df["R_SM_mean_minus_POST_mean_rule"]
    keep_columns = [
        "subject_id",
        "baseline_test_acc",
        "gates_no_rule_test_acc",
        "gates_rule_test_acc",
        "delta_gates_vs_baseline_acc",
        "delta_rule_vs_baseline_acc",
        "delta_rule_vs_gates_acc",
        "baseline_test_kappa",
        "gates_no_rule_test_kappa",
        "gates_rule_test_kappa",
        "delta_gates_vs_baseline_kappa",
        "delta_rule_vs_baseline_kappa",
        "delta_rule_vs_gates_kappa",
        "R_SM_mean",
        "R_POST_mean",
        "R_OTHER_mean",
        "R_SM_mean_minus_POST_mean",
    ]
    return df[keep_columns].sort_values("subject_id").reset_index(drop=True)


def build_global_comparison(subject_comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = {
        "baseline_test_acc": subject_comparison["baseline_test_acc"],
        "gates_no_rule_test_acc": subject_comparison["gates_no_rule_test_acc"],
        "gates_rule_test_acc": subject_comparison["gates_rule_test_acc"],
        "delta_gates_vs_baseline_acc": subject_comparison["delta_gates_vs_baseline_acc"],
        "delta_rule_vs_baseline_acc": subject_comparison["delta_rule_vs_baseline_acc"],
        "delta_rule_vs_gates_acc": subject_comparison["delta_rule_vs_gates_acc"],
        "baseline_test_kappa": subject_comparison["baseline_test_kappa"],
        "gates_no_rule_test_kappa": subject_comparison["gates_no_rule_test_kappa"],
        "gates_rule_test_kappa": subject_comparison["gates_rule_test_kappa"],
        "delta_gates_vs_baseline_kappa": subject_comparison["delta_gates_vs_baseline_kappa"],
        "delta_rule_vs_baseline_kappa": subject_comparison["delta_rule_vs_baseline_kappa"],
        "delta_rule_vs_gates_kappa": subject_comparison["delta_rule_vs_gates_kappa"],
        "R_SM_mean": subject_comparison["R_SM_mean"],
        "R_POST_mean": subject_comparison["R_POST_mean"],
        "R_OTHER_mean": subject_comparison["R_OTHER_mean"],
        "R_SM_mean_minus_POST_mean": subject_comparison["R_SM_mean_minus_POST_mean"],
    }
    for metric_name, series in metrics.items():
        rows.append(
            {
                "metric": metric_name,
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
                "median": float(series.median()),
                "min": float(series.min()),
                "max": float(series.max()),
                "n_subjects": int(series.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def load_gate_csv(path: Path, value_column: str = "effective_gate") -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["channel_name"] = pd.Categorical(df["channel_name"], categories=CHANNEL_ORDER, ordered=True)
    df = df.sort_values("channel_name")
    df[value_column] = pd.to_numeric(df[value_column], errors="coerce")
    return df


def collect_selected_gate_frames(subject_df: pd.DataFrame, model_type: str, lambda_rule: float, label: str) -> pd.DataFrame:
    subset = subject_df[
        (subject_df["model_type"] == model_type)
        & (np.isclose(subject_df["lambda_rule"], lambda_rule, atol=FLOAT_TOL))
    ].copy()
    frames = []
    for _, row in subset.iterrows():
        gate_path = Path(str(row["effective_channel_gates_csv"]))
        frame = load_gate_csv(gate_path)
        frame["subject_id"] = row["subject_id"]
        frame["seed"] = int(row["seed"])
        frame["model_label"] = label
        frame["lambda_rule"] = float(row["lambda_rule"])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def aggregate_channel_gates(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["model_label", "channel_name", "group"], as_index=False)
        .agg(
            mean_effective_gate=("effective_gate", "mean"),
            std_effective_gate=("effective_gate", "std"),
            median_effective_gate=("effective_gate", "median"),
            n=("effective_gate", "count"),
        )
    )
    agg["channel_name"] = pd.Categorical(agg["channel_name"], categories=CHANNEL_ORDER, ordered=True)
    return agg.sort_values(["model_label", "channel_name"]).reset_index(drop=True)


def build_channel_comparison(no_rule_df: pd.DataFrame, rule_df: pd.DataFrame) -> pd.DataFrame:
    merged = no_rule_df.merge(
        rule_df,
        on=["channel_name", "group"],
        suffixes=("_gates_no_rule", "_gates_rule"),
        how="inner",
    )
    merged["delta_rule_minus_no_rule"] = (
        merged["mean_effective_gate_gates_rule"] - merged["mean_effective_gate_gates_no_rule"]
    )
    merged["abs_delta"] = merged["delta_rule_minus_no_rule"].abs()
    return merged.sort_values("abs_delta", ascending=False).reset_index(drop=True)


def build_group_summary(selected_gates: pd.DataFrame) -> pd.DataFrame:
    return (
        selected_gates.groupby(["model_label", "group"], as_index=False)
        .agg(
            mean_effective_gate=("effective_gate", "mean"),
            std_effective_gate=("effective_gate", "std"),
            median_effective_gate=("effective_gate", "median"),
        )
        .sort_values(["model_label", "group"])
    )


def make_info(channel_names: list[str]):
    info = mne.create_info(ch_names=channel_names, sfreq=250, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, match_case=False, on_missing="ignore")
    return info


def save_figures_required():
    if plt is None:
        raise ModuleNotFoundError("matplotlib is required to generate summary figures.")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def plot_subject_bars(subject_df: pd.DataFrame, metric_prefix: str, ylabel: str, output_path: Path) -> None:
    save_figures_required()
    x = np.arange(len(subject_df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, subject_df[f"baseline_test_{metric_prefix}"], width=width, label="baseline")
    ax.bar(x, subject_df[f"gates_no_rule_test_{metric_prefix}"], width=width, label="gates no rule")
    ax.bar(x + width, subject_df[f"gates_rule_test_{metric_prefix}"], width=width, label="gates + rule")
    ax.set_xticks(x)
    ax.set_xticklabels(subject_df["subject_id"], rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_subject_deltas(subject_df: pd.DataFrame, metric_prefix: str, ylabel: str, output_path: Path) -> None:
    save_figures_required()
    cols = [
        f"delta_gates_vs_baseline_{metric_prefix}",
        f"delta_rule_vs_baseline_{metric_prefix}",
        f"delta_rule_vs_gates_{metric_prefix}",
    ]
    labels = ["gates-baseline", "rule-baseline", "rule-gates"]
    x = np.arange(len(subject_df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, (col, label) in enumerate(zip(cols, labels)):
        ax.bar(x + (idx - 1) * width, subject_df[col], width=width, label=label)
    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(subject_df["subject_id"], rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_global_means(subject_df: pd.DataFrame, metric_prefix: str, ylabel: str, output_path: Path) -> None:
    save_figures_required()
    series = {
        "baseline": subject_df[f"baseline_test_{metric_prefix}"],
        "gates no rule": subject_df[f"gates_no_rule_test_{metric_prefix}"],
        "gates + rule": subject_df[f"gates_rule_test_{metric_prefix}"],
    }
    labels = list(series.keys())
    means = [float(s.mean()) for s in series.values()]
    errs = [float(s.std(ddof=1)) if len(s) > 1 else 0.0 for s in series.values()]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, means, yerr=errs, capsize=5, color=["#4c78a8", "#72b7b2", "#f58518"])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_lambda_curve(agg_df: pd.DataFrame, ycol: str, ylabel: str, output_path: Path) -> None:
    save_figures_required()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(agg_df["lambda_rule"], agg_df[ycol], marker="o")
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel("lambda_rule")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_group_bars(group_df: pd.DataFrame, output_path: Path) -> None:
    save_figures_required()
    pivot = group_df.pivot(index="group", columns="model_label", values="mean_effective_gate").reindex(["sensorimotor", "posterior", "other"])
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="bar", ax=ax, color=["#72b7b2", "#f58518"])
    ax.set_ylabel("Mean effective gate")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_channel_comparison(channel_df: pd.DataFrame, output_path: Path) -> None:
    save_figures_required()
    ordered = channel_df.copy()
    ordered["channel_name"] = pd.Categorical(ordered["channel_name"], categories=CHANNEL_ORDER, ordered=True)
    ordered = ordered.sort_values("channel_name")
    x = np.arange(len(ordered))
    width = 0.4
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - width / 2, ordered["mean_effective_gate_gates_no_rule"], width=width, label="gates no rule")
    ax.bar(x + width / 2, ordered["mean_effective_gate_gates_rule"], width=width, label="gates + rule")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["channel_name"], rotation=60, ha="right")
    ax.set_ylabel("Mean effective gate")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_channel_ranking(channel_df: pd.DataFrame, output_path: Path) -> None:
    save_figures_required()
    top = channel_df.sort_values("mean_effective_gate_gates_rule", ascending=False).head(12).copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top["channel_name"].astype(str), top["mean_effective_gate_gates_rule"], color="#f58518")
    ax.invert_yaxis()
    ax.set_xlabel("Mean effective gate")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_topomap(df: pd.DataFrame, value_column: str, title: str, output_path: Path) -> None:
    if plt is None or mne is None:
        return
    ordered = df.copy()
    ordered["channel_name"] = pd.Categorical(ordered["channel_name"], categories=CHANNEL_ORDER, ordered=True)
    ordered = ordered.sort_values("channel_name")
    values = ordered[value_column].to_numpy()
    info = make_info(ordered["channel_name"].astype(str).tolist())
    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = float(np.max(np.abs(values))) if len(values) else 1.0
    kwargs = {"cmap": "viridis", "contours": 6}
    if "delta" in value_column:
        kwargs["cmap"] = "RdBu_r"
        kwargs["vlim"] = (-vmax, vmax) if vmax > 0 else (-1e-6, 1e-6)
    im, _ = mne.viz.plot_topomap(values, info, axes=ax, show=False, sensors=True, **kwargs)
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(value_column)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_interpretation_notes(
    path: Path,
    baseline_config: dict[str, object],
    best_rule_row: pd.Series,
    subject_comparison: pd.DataFrame,
    channel_comparison: pd.DataFrame,
) -> None:
    top_up = channel_comparison.sort_values("delta_rule_minus_no_rule", ascending=False).head(5)["channel_name"].astype(str).tolist()
    top_down = channel_comparison.sort_values("delta_rule_minus_no_rule", ascending=True).head(5)["channel_name"].astype(str).tolist()
    lines = [
        "1. The selected baseline configuration matches the validation-only baseline export: weight_decay, lr=9e-4, epochs=300, batch_size=64, weight_decay=1e-4, scheduler=none, early_stopping_patience=50.",
        "2. Session T is the only source for train/validation; session E is reserved for final test in every subject-level run.",
        "3. lambda_rule was selected using only mean validation accuracy, then mean validation kappa, then mean final validation accuracy; test E was not part of selection.",
        f"4. Selected lambda_rule for EEGNet + gates + rule: {best_rule_row['lambda_rule']}.",
        f"5. Mean delta test accuracy (rule - baseline): {subject_comparison['delta_rule_vs_baseline_acc'].mean():+.4f}.",
        f"6. Mean delta test accuracy (rule - gates no rule): {subject_comparison['delta_rule_vs_gates_acc'].mean():+.4f}.",
        f"7. Mean R_SM_mean - R_POST_mean for the selected rule model: {subject_comparison['R_SM_mean_minus_POST_mean'].mean():+.4f}.",
        f"8. Channels most reinforced by the rule relative to gates-only: {', '.join(top_up)}.",
        f"9. Channels most reduced by the rule relative to gates-only: {', '.join(top_down)}.",
        "10. Channel gates are global multiplicative weights per EEG channel; they should be interpreted as spatial importance proxies, not causal explanations.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    run_root = args.run_root
    analysis_dir = run_root / "analysis_figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    logic_runs_df = load_logic_runs(args.logic_run_root)
    logic_agg = aggregate_logic_conditions(logic_runs_df)
    logic_agg.to_csv(run_root / "eegnet_logic_subject_session_lambda_summary.csv", index=False)

    baseline_runs_df, baseline_subject_df, baseline_config = load_baseline_runs(args.baseline_run_root)
    logic_subject_df = load_logic_subject_metrics(logic_runs_df)

    best_lambda, best_rule_row = select_best_rule_lambda(logic_agg)

    baseline_subjects = aggregate_baseline_subjects(baseline_subject_df)
    gates_subjects = aggregate_logic_subjects(logic_subject_df, "eegnet_gates_no_rule", 0.0, "gates_no_rule")
    rule_subjects = aggregate_logic_subjects(logic_subject_df, "eegnet_gates_rule", best_lambda, "gates_rule")

    subject_comparison = build_subject_comparison(baseline_subjects, gates_subjects, rule_subjects)
    subject_comparison_path = run_root / "eegnet_logic_subject_session_subject_comparison.csv"
    subject_comparison.to_csv(subject_comparison_path, index=False)

    global_comparison = build_global_comparison(subject_comparison)
    global_comparison_path = run_root / "eegnet_logic_subject_session_global_comparison.csv"
    global_comparison.to_csv(global_comparison_path, index=False)

    no_rule_frames = collect_selected_gate_frames(logic_subject_df, "eegnet_gates_no_rule", 0.0, "gates_no_rule")
    rule_frames = collect_selected_gate_frames(logic_subject_df, "eegnet_gates_rule", best_lambda, "gates_rule")
    selected_gates = pd.concat([no_rule_frames, rule_frames], ignore_index=True)
    selected_gates.to_csv(run_root / "eegnet_logic_subject_session_selected_gate_values_long.csv", index=False)

    no_rule_agg = aggregate_channel_gates(no_rule_frames)
    rule_agg = aggregate_channel_gates(rule_frames)
    no_rule_agg.to_csv(run_root / "eegnet_logic_subject_session_gates_no_rule_channel_summary.csv", index=False)
    rule_agg.to_csv(run_root / "eegnet_logic_subject_session_gates_rule_channel_summary.csv", index=False)

    channel_comparison = build_channel_comparison(no_rule_agg, rule_agg)
    channel_comparison_path = run_root / "eegnet_logic_subject_session_channel_comparison.csv"
    channel_comparison.to_csv(channel_comparison_path, index=False)

    group_summary = build_group_summary(selected_gates)
    group_summary.to_csv(run_root / "eegnet_logic_subject_session_group_gate_summary.csv", index=False)

    lambda_plot_df = logic_agg[logic_agg["model_type"] == "eegnet_gates_rule"].copy()
    if not ((logic_agg["model_type"] == "eegnet_gates_no_rule") & np.isclose(logic_agg["lambda_rule"], 0.0, atol=FLOAT_TOL)).any():
        raise ValueError("Missing eegnet_gates_no_rule lambda=0.0 runs.")
    lambda0_row = logic_agg[(logic_agg["model_type"] == "eegnet_gates_no_rule") & np.isclose(logic_agg["lambda_rule"], 0.0, atol=FLOAT_TOL)].iloc[0].copy()
    lambda0_row["model_type"] = "lambda_plot_reference"
    lambda_plot_df = pd.concat([pd.DataFrame([lambda0_row]), lambda_plot_df], ignore_index=True).sort_values("lambda_rule")

    if plt is not None:
        plot_subject_bars(subject_comparison, "acc", "Test accuracy", analysis_dir / "01_accuracy_by_subject.png")
        plot_subject_bars(subject_comparison, "kappa", "Test kappa", analysis_dir / "02_kappa_by_subject.png")
        plot_subject_deltas(subject_comparison, "acc", "Delta test accuracy", analysis_dir / "03_delta_accuracy_by_subject.png")
        plot_subject_deltas(subject_comparison, "kappa", "Delta test kappa", analysis_dir / "04_delta_kappa_by_subject.png")
        plot_global_means(subject_comparison, "acc", "Mean test accuracy", analysis_dir / "05_global_accuracy_mean_errorbars.png")
        plot_global_means(subject_comparison, "kappa", "Mean test kappa", analysis_dir / "06_global_kappa_mean_errorbars.png")
        plot_lambda_curve(lambda_plot_df, "mean_best_val_acc", "Mean validation accuracy", analysis_dir / "07_lambda_vs_mean_validation_accuracy.png")
        plot_lambda_curve(lambda_plot_df, "mean_test_acc", "Mean test accuracy", analysis_dir / "08_lambda_vs_mean_test_accuracy.png")
        plot_lambda_curve(lambda_plot_df, "mean_test_kappa", "Mean test kappa", analysis_dir / "09_lambda_vs_mean_test_kappa.png")
        plot_lambda_curve(lambda_plot_df, "mean_R_SM_mean_minus_POST_mean", "R_SM_mean - R_POST_mean", analysis_dir / "10_lambda_vs_r_sm_mean_minus_post_mean.png")
        plot_group_bars(group_summary, analysis_dir / "11_group_mean_gates.png")
        plot_channel_comparison(channel_comparison, analysis_dir / "12_channel_gates_no_rule_vs_rule.png")
        plot_channel_ranking(channel_comparison, analysis_dir / "13_channel_ranking_rule.png")
        if mne is not None:
            topomap_input_no_rule = no_rule_agg.rename(columns={"mean_effective_gate": "value"})
            topomap_input_rule = rule_agg.rename(columns={"mean_effective_gate": "value"})
            topomap_input_delta = channel_comparison[["channel_name", "group", "delta_rule_minus_no_rule"]].copy()
            plot_topomap(topomap_input_no_rule, "value", "Gates no rule", analysis_dir / "14_topomap_gates_no_rule.png")
            plot_topomap(topomap_input_rule, "value", "Gates + rule", analysis_dir / "14_topomap_gates_rule.png")
            plot_topomap(topomap_input_delta, "delta_rule_minus_no_rule", "Rule - no rule", analysis_dir / "14_topomap_rule_minus_no_rule.png")

    comparison_summary = {
        "selection_protocol": {
            "primary_metric": "mean_best_val_acc",
            "secondary_metric": "mean_best_val_kappa",
            "tertiary_metric": "mean_final_val_acc",
            "selection_metric": "validation_only",
            "test_leakage_guard": "Session E test metrics were excluded from early stopping, lambda selection, seed comparison and best-model selection.",
        },
        "baseline_reference": sanitize_for_json(baseline_config.get("selected_config", {})),
        "selected_rule_lambda": best_lambda,
        "selected_rule_summary": sanitize_for_json(best_rule_row.to_dict()),
        "subject_comparison_csv": relative_to_repo(subject_comparison_path),
        "global_comparison_csv": relative_to_repo(global_comparison_path),
        "channel_comparison_csv": relative_to_repo(channel_comparison_path),
        "analysis_dir": relative_to_repo(analysis_dir),
    }
    (run_root / "eegnet_logic_subject_session_selection_summary.json").write_text(
        json.dumps(sanitize_for_json(comparison_summary), indent=2) + "\n"
    )

    write_interpretation_notes(
        run_root / "interpretation_notes.txt",
        baseline_config=baseline_config,
        best_rule_row=best_rule_row,
        subject_comparison=subject_comparison,
        channel_comparison=channel_comparison,
    )

    print(f"selected_lambda_rule={best_lambda}")
    print(f"subject_comparison_csv={subject_comparison_path}")
    print(f"global_comparison_csv={global_comparison_path}")


if __name__ == "__main__":
    main()
