#!/usr/bin/env python
"""Create publication-ready EEG channel-importance comparison figures."""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR / "runs"
MPLCONFIG_DIR = RUNS_DIR / ".matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd


EXPECTED_SENSORIMOTOR_CHANNELS = [
    "FC3",
    "FC4",
    "FCz",
    "C3",
    "C4",
    "Cz",
    "CP3",
    "CP4",
    "CPz",
]
EXPECTED_SENSORIMOTOR_SET = set(EXPECTED_SENSORIMOTOR_CHANNELS)

REQUIRED_COLUMNS = {"channel_index", "channel_name", "importance"}
PERMUTED_COLUMN_ALIASES = {"permuted_acc", "permuted_accuracy"}

BASELINE_BEST_VAL_ACC = 0.6921
LOGIC_BEST_VAL_ACC = 0.7114

BASELINE_COLOR = "#4f6f8f"
LOGIC_COLOR = "#b66a35"
POSITIVE_DELTA_COLOR = "#5f8f72"
NEGATIVE_DELTA_COLOR = "#b55a5a"
SENSORIMOTOR_COLOR = "#7a3b7a"
TEXT_COLOR = "#252525"
GRID_COLOR = "#d8d8d8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=RUNS_DIR / "channel_importance.csv",
        help="Channel importance CSV for the baseline EEGNet model.",
    )
    parser.add_argument(
        "--logic-csv",
        type=Path,
        default=RUNS_DIR / "channel_importance_logic.csv",
        help="Channel importance CSV for the EEGNet + logic-loss model.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=RUNS_DIR / "channel_importance_comparison_pretty.csv",
        help="Combined comparison CSV to write.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=RUNS_DIR / "channel_importance_pretty.png",
        help="Publication-ready grouped bar plot PNG to write.",
    )
    parser.add_argument(
        "--output-delta-png",
        type=Path,
        default=RUNS_DIR / "channel_importance_delta_pretty.png",
        help="Publication-ready delta bar plot PNG to write.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 16,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def validate_columns(df: pd.DataFrame, csv_path: Path) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    has_permuted_column = bool(PERMUTED_COLUMN_ALIASES.intersection(df.columns))
    if not has_permuted_column:
        missing.append("permuted_acc")

    if missing:
        expected = sorted(REQUIRED_COLUMNS.union({"permuted_acc"}))
        raise ValueError(
            f"CSV is missing required columns: {', '.join(missing)}\n"
            f"File: {csv_path.resolve()}\n"
            f"Expected columns: {', '.join(expected)}\n"
            f"Found columns: {', '.join(map(str, df.columns))}"
        )


def load_importance_csv(csv_path: Path, label: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"{label} CSV not found: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{label} CSV has no rows: {csv_path.resolve()}")

    validate_columns(df, csv_path)
    return df[["channel_index", "channel_name", "importance"]].copy()


def warn_about_channel_mismatches(baseline: pd.DataFrame, logic: pd.DataFrame) -> None:
    key_columns = ["channel_index", "channel_name"]
    baseline_keys = set(map(tuple, baseline[key_columns].to_numpy()))
    logic_keys = set(map(tuple, logic[key_columns].to_numpy()))

    only_baseline = sorted(baseline_keys - logic_keys)
    only_logic = sorted(logic_keys - baseline_keys)

    if only_baseline:
        warnings.warn(
            "Channels present only in baseline CSV and excluded from merge: "
            + ", ".join(f"{idx}:{name}" for idx, name in only_baseline),
            stacklevel=2,
        )
    if only_logic:
        warnings.warn(
            "Channels present only in logic-loss CSV and excluded from merge: "
            + ", ".join(f"{idx}:{name}" for idx, name in only_logic),
            stacklevel=2,
        )


def build_comparison(baseline_csv: Path, logic_csv: Path) -> pd.DataFrame:
    baseline = load_importance_csv(baseline_csv, "Baseline").rename(
        columns={"importance": "importance_baseline"}
    )
    logic = load_importance_csv(logic_csv, "Logic-loss").rename(
        columns={"importance": "importance_logic"}
    )
    warn_about_channel_mismatches(baseline, logic)

    comparison = baseline.merge(
        logic,
        on=["channel_index", "channel_name"],
        how="inner",
        validate="one_to_one",
    )
    if comparison.empty:
        raise ValueError(
            "No matching channels found after merging on channel_index and channel_name."
        )

    comparison["delta_importance"] = (
        comparison["importance_logic"] - comparison["importance_baseline"]
    )
    return comparison[
        [
            "channel_index",
            "channel_name",
            "importance_baseline",
            "importance_logic",
            "delta_importance",
        ]
    ]


def channel_label(channel_name: str) -> str:
    if channel_name in EXPECTED_SENSORIMOTOR_SET:
        return f"{channel_name} *"
    return channel_name


def apply_clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    ax.grid(axis="x", color=GRID_COLOR, linestyle="-", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)


def style_sensorimotor_labels(ax: plt.Axes) -> None:
    for tick in ax.get_yticklabels():
        channel_name = tick.get_text().replace(" *", "")
        if channel_name in EXPECTED_SENSORIMOTOR_SET:
            tick.set_color(SENSORIMOTOR_COLOR)
            tick.set_fontweight("bold")


def sensorimotor_bar_edges(channel_names: pd.Series) -> tuple[list[str], list[float]]:
    edgecolors = []
    linewidths = []
    for channel_name in channel_names:
        if channel_name in EXPECTED_SENSORIMOTOR_SET:
            edgecolors.append(SENSORIMOTOR_COLOR)
            linewidths.append(1.3)
        else:
            edgecolors.append("none")
            linewidths.append(0.0)
    return edgecolors, linewidths


def write_comparison_csv(comparison: pd.DataFrame, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_csv, index=False)


def plot_main_comparison(comparison: pd.DataFrame, output_png: Path) -> None:
    plot_df = comparison.sort_values("importance_logic", ascending=False).reset_index(
        drop=True
    )
    y_positions = list(range(len(plot_df)))
    bar_height = 0.34
    y_baseline = [pos - bar_height / 2 for pos in y_positions]
    y_logic = [pos + bar_height / 2 for pos in y_positions]
    edgecolors, linewidths = sensorimotor_bar_edges(plot_df["channel_name"])

    fig_height = max(7.2, len(plot_df) * 0.39)
    fig, ax = plt.subplots(figsize=(12.5, fig_height))

    ax.barh(
        y_baseline,
        plot_df["importance_baseline"],
        height=bar_height,
        color=BASELINE_COLOR,
        edgecolor=edgecolors,
        linewidth=linewidths,
        alpha=0.92,
        label="Baseline EEGNet",
    )
    ax.barh(
        y_logic,
        plot_df["importance_logic"],
        height=bar_height,
        color=LOGIC_COLOR,
        edgecolor=edgecolors,
        linewidth=linewidths,
        alpha=0.92,
        label="EEGNet + Logic Loss",
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([channel_label(name) for name in plot_df["channel_name"]])
    ax.invert_yaxis()
    ax.set_xlabel("Permutation Importance")
    ax.set_ylabel("EEG Channel")
    ax.set_title(
        "Channel Importance Comparison for Motor Imagery EEG",
        loc="left",
        pad=24,
        fontweight="bold",
    )
    ax.text(
        0.0,
        1.018,
        "BCI Competition IV 2a - EEGNet baseline vs EEGNet with logic loss",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="#5c5c5c",
        fontsize=10.5,
    )

    top_logic = plot_df.head(5)
    text_offset = max(plot_df["importance_logic"].max() * 0.018, 0.004)
    for rank, row in enumerate(top_logic.itertuples(index=False), start=1):
        row_idx = top_logic.index[rank - 1]
        ax.text(
            row.importance_logic + text_offset,
            y_logic[row_idx],
            f"#{rank}",
            va="center",
            ha="left",
            fontsize=8.8,
            color="#555555",
        )

    accuracy_text = (
        f"baseline best val acc = {BASELINE_BEST_VAL_ACC:.4f}\n"
        f"logic-loss best val acc = {LOGIC_BEST_VAL_ACC:.4f}"
    )
    ax.text(
        0.985,
        0.045,
        accuracy_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.2,
        color=TEXT_COLOR,
        bbox={
            "boxstyle": "round,pad=0.38",
            "facecolor": "#f6f4ef",
            "edgecolor": "#b8b3a8",
            "linewidth": 0.8,
        },
    )

    sensorimotor_patch = mpatches.Patch(
        facecolor="white",
        edgecolor=SENSORIMOTOR_COLOR,
        linewidth=1.3,
        label="Sensorimotor channels",
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles + [sensorimotor_patch],
        labels + [sensorimotor_patch.get_label()],
        loc="lower right",
        bbox_to_anchor=(0.985, 0.16),
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#cfcfcf",
    )

    apply_clean_axes(ax)
    style_sensorimotor_labels(ax)
    ax.margins(x=0.08)
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def plot_delta(comparison: pd.DataFrame, output_png: Path) -> None:
    plot_df = comparison.sort_values("delta_importance", ascending=False).reset_index(
        drop=True
    )
    colors = [
        POSITIVE_DELTA_COLOR if delta >= 0 else NEGATIVE_DELTA_COLOR
        for delta in plot_df["delta_importance"]
    ]
    edgecolors, linewidths = sensorimotor_bar_edges(plot_df["channel_name"])

    fig_height = max(7.2, len(plot_df) * 0.39)
    fig, ax = plt.subplots(figsize=(12.5, fig_height))
    ax.barh(
        [channel_label(name) for name in plot_df["channel_name"]],
        plot_df["delta_importance"],
        color=colors,
        edgecolor=edgecolors,
        linewidth=linewidths,
        alpha=0.94,
    )
    ax.axvline(0, color="#2d2d2d", linewidth=1.1)
    ax.invert_yaxis()
    ax.set_xlabel("Delta Permutation Importance (Logic - Baseline)")
    ax.set_ylabel("EEG Channel")
    ax.set_title(
        "Change in Channel Importance After Adding Logic Loss",
        loc="left",
        pad=14,
        fontweight="bold",
    )

    positive_patch = mpatches.Patch(color=POSITIVE_DELTA_COLOR, label="Positive delta")
    negative_patch = mpatches.Patch(color=NEGATIVE_DELTA_COLOR, label="Negative delta")
    sensorimotor_patch = mpatches.Patch(
        facecolor="white",
        edgecolor=SENSORIMOTOR_COLOR,
        linewidth=1.3,
        label="Sensorimotor channels",
    )
    ax.legend(
        handles=[positive_patch, negative_patch, sensorimotor_patch],
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#cfcfcf",
    )

    apply_clean_axes(ax)
    style_sensorimotor_labels(ax)
    max_abs_delta = plot_df["delta_importance"].abs().max()
    ax.set_xlim(-max_abs_delta * 1.12, max_abs_delta * 1.18)
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def print_summary_tables(comparison: pd.DataFrame) -> None:
    top_logic = comparison.sort_values("importance_logic", ascending=False).head(10)
    top_delta = comparison.sort_values("delta_importance", ascending=False).head(10)
    sensorimotor = comparison[
        comparison["channel_name"].isin(EXPECTED_SENSORIMOTOR_SET)
    ].sort_values("channel_index")

    print("\nTop 10 channels by importance_logic:")
    print(
        top_logic[
            ["channel_index", "channel_name", "importance_logic"]
        ].to_string(index=False)
    )

    print("\nTop 10 channels by delta_importance:")
    print(
        top_delta[
            ["channel_index", "channel_name", "delta_importance"]
        ].to_string(index=False)
    )

    print("\nSensorimotor channels:")
    print(
        sensorimotor[
            [
                "channel_index",
                "channel_name",
                "importance_baseline",
                "importance_logic",
                "delta_importance",
            ]
        ].to_string(index=False)
    )


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    comparison = build_comparison(args.baseline_csv, args.logic_csv)
    write_comparison_csv(comparison, args.output_csv)
    plot_main_comparison(comparison, args.output_png)
    plot_delta(comparison, args.output_delta_png)
    print_summary_tables(comparison)

    print(f"\nSaved comparison CSV: {args.output_csv.resolve()}")
    print(f"Saved main PNG: {args.output_png.resolve()}")
    print(f"Saved delta PNG: {args.output_delta_png.resolve()}")


if __name__ == "__main__":
    main()
