#!/usr/bin/env python
"""Create polished final channel-analysis figures from final CSV outputs only."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR / "runs"
DEFAULT_ANALYSIS_DIR = RUNS_DIR / "final_channel_analysis"
MPLCONFIG_DIR = RUNS_DIR / ".matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASS_NAMES = {
    0: "left_hand",
    1: "right_hand",
    2: "feet",
    3: "tongue",
}

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

EXPECTED_CHANNELS_BY_CLASS = {
    0: {"C4", "CP4", "FC4"},
    1: {"C3", "CP3", "FC3"},
    2: {"Cz", "CPz", "FCz"},
    3: {"C3", "C4", "Cz", "FC3", "FC4", "FCz"},
}

MODEL_ORDER = ["baseline", "logic"]

BASELINE_COLOR = "#4F6F8F"
LOGIC_COLOR = "#B66A35"
POSITIVE_DELTA_COLOR = "#5F8F72"
NEGATIVE_DELTA_COLOR = "#B55A5A"
SENSORIMOTOR_COLOR = "#7A3B7A"
TEXT_COLOR = "#252525"
GRID_COLOR = "#D8D8D8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=DEFAULT_ANALYSIS_DIR,
        help="Directory containing baseline/, logic_lam1p0/ and comparison/ final CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for pretty figures. Defaults to <analysis-dir>/pretty_figures.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Channels to show per class if a final by-class CSV exists.",
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


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path.resolve()}")


def load_importance_csv(path: Path, label: str) -> pd.DataFrame:
    ensure_exists(path, label)
    df = pd.read_csv(path)
    required = {"channel_index", "channel_name", "importance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{label} missing required columns: {sorted(missing)}")
    df = df.copy()
    df["channel_index"] = pd.to_numeric(df["channel_index"], errors="raise").astype(int)
    df["channel_name"] = df["channel_name"].astype(str)
    df["importance"] = pd.to_numeric(df["importance"], errors="raise")
    return df


def load_delta_csv(path: Path) -> pd.DataFrame:
    ensure_exists(path, "Delta CSV")
    df = pd.read_csv(path)
    required = {"channel", "baseline_importance", "logic_importance", "delta_logic_minus_baseline"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Delta CSV missing required columns: {sorted(missing)}")
    df = df.copy()
    df["channel"] = df["channel"].astype(str)
    for column in ["baseline_importance", "logic_importance", "delta_logic_minus_baseline"]:
        df[column] = pd.to_numeric(df[column], errors="raise")
    if "abs_delta" not in df.columns:
        df["abs_delta"] = df["delta_logic_minus_baseline"].abs()
    else:
        df["abs_delta"] = pd.to_numeric(df["abs_delta"], errors="raise")
    return df


def load_coherence_csv(path: Path) -> pd.DataFrame:
    ensure_exists(path, "Coherence CSV")
    df = pd.read_csv(path)
    required = {"model", "sensorimotor_ratio", "posterior_ratio"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Coherence CSV missing required columns: {sorted(missing)}")
    df = df.copy()
    df["model"] = df["model"].astype(str)
    df["sensorimotor_ratio"] = pd.to_numeric(df["sensorimotor_ratio"], errors="raise")
    df["posterior_ratio"] = pd.to_numeric(df["posterior_ratio"], errors="raise")
    if "sensorimotor_minus_posterior" not in df.columns:
        df["sensorimotor_minus_posterior"] = (
            df["sensorimotor_ratio"] - df["posterior_ratio"]
        )
    else:
        df["sensorimotor_minus_posterior"] = pd.to_numeric(
            df["sensorimotor_minus_posterior"], errors="raise"
        )
    return df


def build_comparison_df(baseline_df: pd.DataFrame, logic_df: pd.DataFrame) -> pd.DataFrame:
    comparison = baseline_df.rename(columns={"importance": "importance_baseline"}).merge(
        logic_df.rename(columns={"importance": "importance_logic"}),
        on=["channel_index", "channel_name"],
        how="inner",
        validate="one_to_one",
    )
    if comparison.empty:
        raise ValueError("No matching channels found between baseline and logic CSVs.")
    comparison["delta_importance"] = (
        comparison["importance_logic"] - comparison["importance_baseline"]
    )
    return comparison


def apply_clean_axes(ax: plt.Axes, grid_axis: str = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    ax.grid(axis=grid_axis, color=GRID_COLOR, linestyle="-", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)


def style_sensorimotor_labels(ax: plt.Axes) -> None:
    for tick in ax.get_yticklabels():
        if tick.get_text().replace(" *", "") in EXPECTED_SENSORIMOTOR_SET:
            tick.set_color(SENSORIMOTOR_COLOR)
            tick.set_fontweight("bold")


def sensorimotor_bar_edges(channel_names: pd.Series) -> tuple[list[str], list[float]]:
    edgecolors = []
    linewidths = []
    for channel_name in channel_names.astype(str):
        if channel_name in EXPECTED_SENSORIMOTOR_SET:
            edgecolors.append(SENSORIMOTOR_COLOR)
            linewidths.append(1.3)
        else:
            edgecolors.append("none")
            linewidths.append(0.0)
    return edgecolors, linewidths


def channel_label(channel_name: str) -> str:
    if channel_name in EXPECTED_SENSORIMOTOR_SET:
        return f"{channel_name} *"
    return channel_name


def plot_comparison(comparison_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = comparison_df.copy()
    plot_df["importance_mean"] = (
        plot_df["importance_baseline"] + plot_df["importance_logic"]
    ) / 2.0
    plot_df = plot_df.sort_values(
        ["importance_mean", "importance_logic"],
        ascending=[False, False],
    ).reset_index(drop=True)

    y_positions = np.arange(len(plot_df))
    bar_height = 0.34
    edgecolors, linewidths = sensorimotor_bar_edges(plot_df["channel_name"])

    fig_height = max(7.2, len(plot_df) * 0.39)
    fig, ax = plt.subplots(figsize=(12.5, fig_height))
    ax.barh(
        y_positions - bar_height / 2,
        plot_df["importance_baseline"],
        height=bar_height,
        color=BASELINE_COLOR,
        edgecolor=edgecolors,
        linewidth=linewidths,
        alpha=0.92,
        label="EEGNet baseline",
    )
    ax.barh(
        y_positions + bar_height / 2,
        plot_df["importance_logic"],
        height=bar_height,
        color=LOGIC_COLOR,
        edgecolor=edgecolors,
        linewidth=linewidths,
        alpha=0.92,
        label="EEGNet + logic loss",
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([channel_label(name) for name in plot_df["channel_name"]])
    ax.invert_yaxis()
    ax.set_xlabel("Permutation importance")
    ax.set_ylabel("EEG channel")
    ax.set_title(
        "Final Channel Importance Comparison",
        loc="left",
        pad=22,
        fontweight="bold",
    )
    ax.text(
        0.0,
        1.018,
        "Final baseline vs final logic-loss model",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="#5C5C5C",
        fontsize=10.5,
    )

    legend_handles = [
        mpatches.Patch(color=BASELINE_COLOR, label="EEGNet baseline"),
        mpatches.Patch(color=LOGIC_COLOR, label="EEGNet + logic loss"),
        mpatches.Patch(
            facecolor="white",
            edgecolor=SENSORIMOTOR_COLOR,
            linewidth=1.3,
            label="Expected sensorimotor channel",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#CFCFCF",
    )

    apply_clean_axes(ax)
    style_sensorimotor_labels(ax)
    ax.margins(x=0.08)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_delta(delta_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = delta_df.sort_values(
        ["delta_logic_minus_baseline", "abs_delta"],
        ascending=[False, False],
    ).reset_index(drop=True)
    colors = [
        POSITIVE_DELTA_COLOR if value >= 0 else NEGATIVE_DELTA_COLOR
        for value in plot_df["delta_logic_minus_baseline"]
    ]
    edgecolors, linewidths = sensorimotor_bar_edges(plot_df["channel"])

    fig_height = max(7.2, len(plot_df) * 0.39)
    fig, ax = plt.subplots(figsize=(12.5, fig_height))
    ax.barh(
        [channel_label(name) for name in plot_df["channel"]],
        plot_df["delta_logic_minus_baseline"],
        color=colors,
        edgecolor=edgecolors,
        linewidth=linewidths,
        alpha=0.94,
    )
    ax.axvline(0, color="#2D2D2D", linewidth=1.1)
    ax.invert_yaxis()
    ax.set_xlabel("Delta permutation importance (Logic - Baseline)")
    ax.set_ylabel("EEG channel")
    ax.set_title(
        "Change in Channel Importance After Adding Logic Loss",
        loc="left",
        pad=14,
        fontweight="bold",
    )

    positive_patch = mpatches.Patch(color=POSITIVE_DELTA_COLOR, label="Increase")
    negative_patch = mpatches.Patch(color=NEGATIVE_DELTA_COLOR, label="Decrease")
    sensorimotor_patch = mpatches.Patch(
        facecolor="white",
        edgecolor=SENSORIMOTOR_COLOR,
        linewidth=1.3,
        label="Expected sensorimotor channel",
    )
    ax.legend(
        handles=[positive_patch, negative_patch, sensorimotor_patch],
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#CFCFCF",
    )

    apply_clean_axes(ax)
    style_sensorimotor_labels(ax)
    max_abs_delta = plot_df["delta_logic_minus_baseline"].abs().max()
    ax.set_xlim(-max_abs_delta * 1.12, max_abs_delta * 1.18)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def normalize_model_name(model: str) -> str:
    lowered = model.lower()
    if "difference" in lowered:
        return "difference"
    if "logic" in lowered:
        return "logic"
    return "baseline"


def plot_coherence(coherence_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = coherence_df.copy()
    plot_df["model_key"] = plot_df["model"].map(normalize_model_name)
    plot_df = plot_df[plot_df["model_key"].isin(["baseline", "logic"])].copy()
    if len(plot_df) != 2:
        raise ValueError("Coherence CSV must contain baseline and logic rows.")

    plot_df = plot_df.sort_values(
        "model_key",
        key=lambda s: s.map({"baseline": 0, "logic": 1}),
    ).reset_index(drop=True)
    plot_df["plot_label"] = ["baseline", "logic"]

    plot_columns = [
        ("sensorimotor_ratio", "Sensorimotor ratio", BASELINE_COLOR),
        ("posterior_ratio", "Posterior ratio", LOGIC_COLOR),
        (
            "sensorimotor_minus_posterior",
            "Sensorimotor - posterior",
            POSITIVE_DELTA_COLOR,
        ),
    ]

    x = np.arange(len(plot_df))
    width = 0.22
    offsets = [-width, 0.0, width]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for offset, (column, label, color) in zip(offsets, plot_columns):
        ax.bar(
            x + offset,
            plot_df[column],
            width=width,
            color=color,
            label=label,
            alpha=0.92,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["plot_label"])
    ax.set_title("Sensorimotor coherence score", loc="left", pad=12, fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("Ratio / difference")
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    apply_clean_axes(ax, grid_axis="y")
    ax.legend(frameon=True, facecolor="white", edgecolor="#CFCFCF")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def by_class_comparison_csv_path(analysis_dir: Path) -> Path:
    return analysis_dir / "comparison" / "channel_importance_by_class_comparison.csv"


def load_by_class_csv(path: Path) -> pd.DataFrame:
    ensure_exists(path, "Per-class comparison CSV")
    df = pd.read_csv(path)
    required = {
        "class_id",
        "class_name",
        "channel",
        "baseline_importance",
        "logic_importance",
        "delta_logic_minus_baseline",
        "is_sensorimotor_channel",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Per-class CSV missing required columns: {sorted(missing)}")

    df = df.copy()
    df["class_id"] = pd.to_numeric(df["class_id"], errors="raise").astype(int)
    df["class_name"] = df["class_name"].astype(str)
    df["channel"] = df["channel"].astype(str)
    df["baseline_importance"] = pd.to_numeric(df["baseline_importance"], errors="raise")
    df["logic_importance"] = pd.to_numeric(df["logic_importance"], errors="raise")
    df["delta_logic_minus_baseline"] = pd.to_numeric(
        df["delta_logic_minus_baseline"], errors="raise"
    )
    df["is_sensorimotor_channel"] = (
        df["is_sensorimotor_channel"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
    )
    if df["is_sensorimotor_channel"].isna().any():
        raise ValueError("Per-class CSV contains invalid is_sensorimotor_channel values.")
    return df


def top_channels_for_class(class_df: pd.DataFrame, top_k: int) -> list[str]:
    pivot = class_df.set_index("channel")[
        ["baseline_importance", "logic_importance"]
    ].copy()
    pivot["top_score"] = pivot[["baseline_importance", "logic_importance"]].max(axis=1)
    return pivot.sort_values("top_score", ascending=False).head(top_k).index.tolist()


def plot_class_panel(ax: plt.Axes, df: pd.DataFrame, class_id: int, top_k: int) -> None:
    class_df = df[df["class_id"] == class_id]
    channels = top_channels_for_class(class_df, top_k)
    plot_df = class_df[class_df["channel"].isin(channels)].set_index("channel")[
        ["baseline_importance", "logic_importance"]
    ]
    plot_df = plot_df.rename(
        columns={
            "baseline_importance": "baseline",
            "logic_importance": "logic",
        }
    )
    plot_df = plot_df.reindex(index=channels, columns=MODEL_ORDER)

    plot_df.plot(
        kind="bar",
        ax=ax,
        width=0.78,
        rot=45,
        color=[BASELINE_COLOR, LOGIC_COLOR],
    )
    expected = EXPECTED_CHANNELS_BY_CLASS[class_id]
    for tick in ax.get_xticklabels():
        if tick.get_text() in expected:
            tick.set_color(SENSORIMOTOR_COLOR)
            tick.set_fontweight("bold")

    for patch in ax.patches:
        center = patch.get_x() + patch.get_width() / 2
        channel_idx = int(round(center))
        if 0 <= channel_idx < len(channels) and channels[channel_idx] in expected:
            patch.set_edgecolor(SENSORIMOTOR_COLOR)
            patch.set_linewidth(1.6)

    ax.set_title(f"{class_id} = {CLASS_NAMES[class_id]}")
    ax.set_xlabel("")
    ax.set_ylabel("Importance")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend_.remove()


def plot_by_class(df: pd.DataFrame, output_path: Path, top_k: int) -> None:
    missing_classes = set(CLASS_NAMES) - set(df["class_id"])
    if missing_classes:
        raise ValueError(f"Per-class CSV missing class rows for: {sorted(missing_classes)}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=False)
    axes = axes.flatten()
    for ax, class_id in zip(axes, CLASS_NAMES):
        plot_class_panel(ax, df, class_id, top_k)

    handles, labels = axes[0].get_legend_handles_labels()
    expected_patch = mpatches.Patch(
        facecolor="white",
        edgecolor=SENSORIMOTOR_COLOR,
        linewidth=1.6,
        label="Expected sensorimotor channel",
    )
    fig.legend(
        handles + [expected_patch],
        labels + ["Expected sensorimotor channel"],
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Top channel permutation importance by class", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1.")

    configure_matplotlib()

    analysis_dir = args.analysis_dir.resolve()
    output_dir = (args.output_dir or (analysis_dir / "pretty_figures")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_csv = analysis_dir / "baseline" / "channel_importance.csv"
    logic_csv = analysis_dir / "logic_lam1p0" / "channel_importance.csv"
    delta_csv = analysis_dir / "comparison" / "channel_importance_delta.csv"
    coherence_csv = analysis_dir / "comparison" / "coherence_scores.csv"

    baseline_df = load_importance_csv(baseline_csv, "Baseline CSV")
    logic_df = load_importance_csv(logic_csv, "Logic CSV")
    delta_df = load_delta_csv(delta_csv)
    coherence_df = load_coherence_csv(coherence_csv)
    comparison_df = build_comparison_df(baseline_df, logic_df)

    comparison_path = output_dir / "channel_importance_comparison_pretty.png"
    delta_path = output_dir / "channel_importance_delta_pretty.png"
    coherence_path = output_dir / "coherence_scores_pretty.png"

    plot_comparison(comparison_df, comparison_path)
    plot_delta(delta_df, delta_path)
    plot_coherence(coherence_df, coherence_path)

    by_class_csv = by_class_comparison_csv_path(analysis_dir)
    if not by_class_csv.exists():
        print(
            "Warning: no final per-class comparison CSV found in analysis dir; "
            "skipping channel_importance_by_class_pretty.png"
        )
    else:
        by_class_df = load_by_class_csv(by_class_csv)
        by_class_path = output_dir / "channel_importance_by_class_pretty.png"
        plot_by_class(by_class_df, by_class_path, args.top_k)
        print(f"Saved {by_class_path}")

    print("Saved:")
    print(comparison_path)
    print(delta_path)
    print(coherence_path)


if __name__ == "__main__":
    main()
