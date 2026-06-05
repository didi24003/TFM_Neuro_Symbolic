#!/usr/bin/env python
"""Plot per-class channel permutation importance for baseline and logic models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from mi_ltn_common import RUNS_DIR


CLASS_NAMES = {
    0: "left_hand",
    1: "right_hand",
    2: "feet",
    3: "tongue",
}

EXPECTED_CHANNELS = {
    0: {"C4", "CP4", "FC4"},
    1: {"C3", "CP3", "FC3"},
    2: {"Cz", "CPz", "FCz"},
    3: {"C3", "C4", "Cz", "FC3", "FC4", "FCz"},
}

MODEL_ORDER = ["baseline", "logic"]
REQUIRED_COLUMNS = {"model_name", "class_id", "channel_name", "importance"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=RUNS_DIR / "channel_importance_by_class.csv",
        help="Input per-class channel-importance CSV.",
    )
    parser.add_argument(
        "--output-fig",
        type=Path,
        default=RUNS_DIR / "channel_importance_by_class.png",
        help="Output PNG figure.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Channels to show per class.")
    return parser.parse_args()


def load_importance(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found at {path.resolve()}")

    df = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Input CSV is missing required columns: {missing}. "
            "Regenerate it with 06_channel_importance_by_class.py so it includes "
            "both baseline and logic rows."
        )

    df = df.copy()
    df["class_id"] = pd.to_numeric(df["class_id"], errors="raise").astype(int)
    df["channel_name"] = df["channel_name"].astype(str)
    df["model_name"] = df["model_name"].astype(str)
    df["importance"] = pd.to_numeric(df["importance"], errors="raise")

    missing_models = set(MODEL_ORDER) - set(df["model_name"])
    if missing_models:
        missing = ", ".join(sorted(missing_models))
        raise ValueError(f"Input CSV is missing model rows for: {missing}")

    missing_classes = set(CLASS_NAMES) - set(df["class_id"])
    if missing_classes:
        missing = ", ".join(str(class_id) for class_id in sorted(missing_classes))
        raise ValueError(f"Input CSV is missing class_id rows for: {missing}")

    return df


def top_channels_for_class(class_df: pd.DataFrame, top_k: int) -> list[str]:
    pivot = class_df.pivot_table(
        index="channel_name",
        columns="model_name",
        values="importance",
        aggfunc="mean",
    )
    pivot = pivot.reindex(columns=MODEL_ORDER)
    pivot["top_score"] = pivot[MODEL_ORDER].max(axis=1)
    return pivot.sort_values("top_score", ascending=False).head(top_k).index.tolist()


def plot_class_panel(ax: plt.Axes, df: pd.DataFrame, class_id: int, top_k: int) -> None:
    class_df = df[df["class_id"] == class_id]
    channels = top_channels_for_class(class_df, top_k)
    plot_df = (
        class_df[class_df["channel_name"].isin(channels)]
        .pivot_table(
            index="channel_name",
            columns="model_name",
            values="importance",
            aggfunc="mean",
        )
        .reindex(index=channels, columns=MODEL_ORDER)
    )

    plot_df.plot(kind="bar", ax=ax, width=0.78, rot=45)
    expected = EXPECTED_CHANNELS[class_id]
    for tick in ax.get_xticklabels():
        if tick.get_text() in expected:
            tick.set_color("#b00020")
            tick.set_fontweight("bold")

    for patch in ax.patches:
        center = patch.get_x() + patch.get_width() / 2
        channel_idx = int(round(center))
        if 0 <= channel_idx < len(channels) and channels[channel_idx] in expected:
            patch.set_edgecolor("#b00020")
            patch.set_linewidth(1.6)

    ax.set_title(f"{class_id} = {CLASS_NAMES[class_id]}")
    ax.set_xlabel("")
    ax.set_ylabel("Importance")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend_.remove()


def plot_importance(df: pd.DataFrame, output_fig: Path, top_k: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=False)
    axes = axes.flatten()

    for ax, class_id in zip(axes, CLASS_NAMES):
        plot_class_panel(ax, df, class_id, top_k)

    handles, labels = axes[0].get_legend_handles_labels()
    expected_patch = mpatches.Patch(
        facecolor="white",
        edgecolor="#b00020",
        linewidth=1.6,
        label="expected sensorimotor channel",
    )
    fig.legend(
        handles + [expected_patch],
        labels + ["expected sensorimotor channel"],
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Top channel permutation importance by class", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    output_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_fig, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1.")

    df = load_importance(args.input_csv)
    plot_importance(df, args.output_fig, args.top_k)
    print(f"saved figure: {args.output_fig}")


if __name__ == "__main__":
    main()
