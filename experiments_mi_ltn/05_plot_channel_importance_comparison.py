#!/usr/bin/env python
"""Compare EEG channel importance between baseline and logic-loss EEGNet runs."""

from __future__ import annotations

import argparse
import os
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


EXPECTED_SENSORIMOTOR_CHANNELS = {
    "FC3",
    "FC4",
    "FCz",
    "C3",
    "C4",
    "Cz",
    "CP3",
    "CP4",
    "CPz",
}

REQUIRED_COLUMNS = {"channel_index", "channel_name", "importance"}
PERMUTED_COLUMN_ALIASES = {"permuted_acc", "permuted_accuracy"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=RUNS_DIR / "channel_importance.csv",
        help="Channel importance CSV for the baseline EEGNet run.",
    )
    parser.add_argument(
        "--logic-csv",
        type=Path,
        default=RUNS_DIR / "channel_importance_logic.csv",
        help="Channel importance CSV for the EEGNet + logic-loss run.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=RUNS_DIR / "channel_importance_comparison.csv",
        help="Combined comparison CSV to write.",
    )
    parser.add_argument(
        "--comparison-png",
        type=Path,
        default=RUNS_DIR / "channel_importance_comparison.png",
        help="Grouped horizontal bar plot PNG to write.",
    )
    parser.add_argument(
        "--delta-png",
        type=Path,
        default=RUNS_DIR / "channel_importance_delta.png",
        help="Delta horizontal bar plot PNG to write.",
    )
    return parser.parse_args()


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


def build_comparison(baseline_csv: Path, logic_csv: Path) -> pd.DataFrame:
    baseline = load_importance_csv(baseline_csv, "Baseline").rename(
        columns={"importance": "importance_baseline"}
    )
    logic = load_importance_csv(logic_csv, "Logic-loss").rename(
        columns={"importance": "importance_logic"}
    )

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


def write_comparison_csv(comparison: pd.DataFrame, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_csv, index=False)


def style_sensorimotor_ticks(ax: plt.Axes) -> None:
    for tick in ax.get_yticklabels():
        if tick.get_text().rstrip(" *") in EXPECTED_SENSORIMOTOR_CHANNELS:
            tick.set_fontweight("bold")
            tick.set_color("#222222")


def sensorimotor_edges(channel_names: pd.Series) -> tuple[list[str], list[float]]:
    edgecolors = []
    linewidths = []
    for channel_name in channel_names:
        if channel_name in EXPECTED_SENSORIMOTOR_CHANNELS:
            edgecolors.append("#111111")
            linewidths.append(1.5)
        else:
            edgecolors.append("none")
            linewidths.append(0.0)
    return edgecolors, linewidths


def plot_comparison(comparison: pd.DataFrame, output_png: Path) -> None:
    plot_df = comparison.sort_values("importance_logic", ascending=False).reset_index(
        drop=True
    )
    y_positions = list(range(len(plot_df)))
    bar_height = 0.38
    y_baseline = [pos - bar_height / 2 for pos in y_positions]
    y_logic = [pos + bar_height / 2 for pos in y_positions]
    edgecolors, linewidths = sensorimotor_edges(plot_df["channel_name"])

    fig_height = max(6.0, len(plot_df) * 0.35)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.barh(
        y_baseline,
        plot_df["importance_baseline"],
        height=bar_height,
        color="#4c78a8",
        edgecolor=edgecolors,
        linewidth=linewidths,
        label="EEGNet Baseline",
    )
    ax.barh(
        y_logic,
        plot_df["importance_logic"],
        height=bar_height,
        color="#f58518",
        edgecolor=edgecolors,
        linewidth=linewidths,
        label="EEGNet + Logic Loss",
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["channel_name"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_ylabel("Channel")
    ax.set_title(
        "Channel Importance Comparison: EEGNet Baseline vs EEGNet + Logic Loss"
    )
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    style_sensorimotor_ticks(ax)

    sensorimotor_patch = mpatches.Patch(
        facecolor="white",
        edgecolor="#111111",
        linewidth=1.5,
        label="Expected sensorimotor channel",
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [sensorimotor_patch], labels + [sensorimotor_patch.get_label()])
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def plot_delta(comparison: pd.DataFrame, output_png: Path) -> None:
    plot_df = comparison.sort_values("delta_importance", ascending=False).reset_index(
        drop=True
    )
    colors = [
        "#54a24b" if delta >= 0 else "#e45756"
        for delta in plot_df["delta_importance"]
    ]
    edgecolors, linewidths = sensorimotor_edges(plot_df["channel_name"])

    fig_height = max(6.0, len(plot_df) * 0.35)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.barh(
        plot_df["channel_name"],
        plot_df["delta_importance"],
        color=colors,
        edgecolor=edgecolors,
        linewidth=linewidths,
    )
    ax.axvline(0, color="#222222", linewidth=1.0)
    ax.invert_yaxis()
    ax.set_xlabel("Delta Importance")
    ax.set_ylabel("Channel")
    ax.set_title("Change in Channel Importance After Logic Loss")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    style_sensorimotor_ticks(ax)

    positive_patch = mpatches.Patch(color="#54a24b", label="Increase")
    negative_patch = mpatches.Patch(color="#e45756", label="Decrease")
    sensorimotor_patch = mpatches.Patch(
        facecolor="white",
        edgecolor="#111111",
        linewidth=1.5,
        label="Expected sensorimotor channel",
    )
    ax.legend(handles=[positive_patch, negative_patch, sensorimotor_patch])
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def print_top_channels(comparison: pd.DataFrame) -> None:
    top_logic = comparison.sort_values("importance_logic", ascending=False).head(10)
    top_delta = comparison.sort_values("delta_importance", ascending=False).head(10)

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


def main() -> None:
    args = parse_args()
    comparison = build_comparison(args.baseline_csv, args.logic_csv)

    write_comparison_csv(comparison, args.output_csv)
    plot_comparison(comparison, args.comparison_png)
    plot_delta(comparison, args.delta_png)
    print_top_channels(comparison)

    print(f"\nSaved comparison CSV: {args.output_csv}")
    print(f"Saved comparison PNG: {args.comparison_png}")
    print(f"Saved delta PNG: {args.delta_png}")


if __name__ == "__main__":
    main()
