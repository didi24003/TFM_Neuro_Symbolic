import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import pandas as pd


BCI_IV_2A_CHANNELS = [
    "Fz",
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]

SENSORIMOTOR_CHANNELS = {
    "FC3", "FC4", "FCz",
    "C3", "C4", "Cz",
    "CP3", "CP4", "CPz",
}


def load_importance(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = {"channel_name", "importance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    df = df.copy()
    df["channel_name"] = df["channel_name"].astype(str)

    # Mantener solo canales del dataset BCI IV 2a y en orden estándar
    df = df[df["channel_name"].isin(BCI_IV_2A_CHANNELS)]
    df["channel_name"] = pd.Categorical(
        df["channel_name"],
        categories=BCI_IV_2A_CHANNELS,
        ordered=True,
    )
    df = df.sort_values("channel_name")

    if len(df) != len(BCI_IV_2A_CHANNELS):
        found = set(df["channel_name"].astype(str))
        missing_channels = set(BCI_IV_2A_CHANNELS) - found
        print(f"Warning: missing channels in {csv_path}: {sorted(missing_channels)}")

    return df


def create_info(channel_names):
    info = mne.create_info(
        ch_names=list(channel_names),
        sfreq=250,
        ch_types="eeg",
    )

    # El montaje standard_1020 contiene posiciones estándar para electrodos EEG.
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, match_case=False, on_missing="ignore")

    return info


def plot_topomap(df: pd.DataFrame, title: str, output_path: Path):
    channel_names = df["channel_name"].astype(str).tolist()
    values = df["importance"].to_numpy()

    info = create_info(channel_names)

    # Resaltar canales sensorimotores esperados
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-csv",
        default="experiments_mi_ltn/runs/channel_importance_baseline_30ep.csv",
    )
    parser.add_argument(
        "--logic-csv",
        default="experiments_mi_ltn/runs/channel_importance_logic_30ep.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments_mi_ltn/runs",
    )
    args = parser.parse_args()

    baseline_csv = Path(args.baseline_csv)
    logic_csv = Path(args.logic_csv)
    output_dir = Path(args.output_dir)

    baseline_df = load_importance(baseline_csv)
    logic_df = load_importance(logic_csv)

    plot_topomap(
        baseline_df,
        "EEGNet baseline - Channel importance",
        output_dir / "topomap_channel_importance_baseline_30ep.png",
    )

    plot_topomap(
        logic_df,
        "EEGNet + logic loss - Channel importance",
        output_dir / "topomap_channel_importance_logic_30ep.png",
    )

    print("Saved:")
    print(output_dir / "topomap_channel_importance_baseline_30ep.png")
    print(output_dir / "topomap_channel_importance_logic_30ep.png")


if __name__ == "__main__":
    main()



