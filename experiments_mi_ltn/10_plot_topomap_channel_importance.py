#!/usr/bin/env python
"""Plot a topomap from a channel-importance CSV."""

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

import matplotlib.pyplot as plt
import mne
import pandas as pd


BCI_IV_2A_CHANNELS = [
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

SENSORIMOTOR_CHANNELS = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--title", type=str, default="Channel importance")
    return parser.parse_args()


def load_importance(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = {"channel_name", "importance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

    df = df.copy()
    df["channel_name"] = df["channel_name"].astype(str)
    df["importance"] = pd.to_numeric(df["importance"], errors="raise")

    # Keep only dataset channels and restore the canonical 2a ordering.
    df = df[df["channel_name"].isin(BCI_IV_2A_CHANNELS)]
    df["channel_name"] = pd.Categorical(
        df["channel_name"],
        categories=BCI_IV_2A_CHANNELS,
        ordered=True,
    )
    df = df.sort_values("channel_name")

    if len(df) != len(BCI_IV_2A_CHANNELS):
        found = set(df["channel_name"].astype(str))
        missing_channels = sorted(set(BCI_IV_2A_CHANNELS) - found)
        print(f"Warning: missing channels in {csv_path}: {missing_channels}")

    return df


def create_info(channel_names: list[str]) -> mne.Info:
    info = mne.create_info(
        ch_names=channel_names,
        sfreq=250,
        ch_types="eeg",
    )
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, match_case=False, on_missing="ignore")
    return info


def plot_topomap(df: pd.DataFrame, title: str, output_path: Path) -> None:
    channel_names = df["channel_name"].astype(str).tolist()
    values = df["importance"].to_numpy()

    info = create_info(channel_names)
    mask = df["channel_name"].astype(str).isin(SENSORIMOTOR_CHANNELS).to_numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    im, _ = mne.viz.plot_topomap(
        values,
        info,
        axes=ax,
        show=False,
        names=channel_names,
        sensors=True,
        mask=mask,
        mask_params=dict(
            marker="o",
            markerfacecolor="none",
            markeredgecolor="black",
            linewidth=1.5,
            markersize=10,
        ),
        contours=6,
        cmap="viridis",
    )

    ax.set_title(title, fontsize=12)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Permutation importance")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = load_importance(args.input_csv)
    plot_topomap(df, args.title, args.output_path)
    print("Saved:")
    print(args.output_path)


if __name__ == "__main__":
    main()
