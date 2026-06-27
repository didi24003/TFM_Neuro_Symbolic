#!/usr/bin/env python
"""Estimate EEG channel importance by validation-set permutation."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR / "runs"
MPLCONFIG_DIR = RUNS_DIR / ".matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from mi_ltn_common import (
    BCI_IV_2A_CHANNELS,
    DEFAULT_DATA_ROOT,
    NUM_ELECTRODES,
    RUNS_DIR,
    build_dataset,
    build_model,
    get_device,
    make_loaders,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=RUNS_DIR / "channel_importance.csv")
    parser.add_argument(
        "--output-sorted-csv",
        type=Path,
        default=None,
        help="Optional CSV sorted by descending importance.",
    )
    parser.add_argument(
        "--output-barplot",
        type=Path,
        default=None,
        help="Optional horizontal bar plot PNG.",
    )
    parser.add_argument(
        "--plot-title",
        type=str,
        default="EEG Channel Importance",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-samples", type=int, default=None)
    return parser.parse_args()


@torch.no_grad()
def accuracy_with_optional_permutation(model, loader, device, channel_idx=None):
    model.eval()
    total_correct = 0
    total_examples = 0
    generator = torch.Generator(device="cpu").manual_seed(10_000 + (channel_idx or 0))

    for x, y in loader:
        if channel_idx is not None:
            perm = torch.randperm(x.size(0), generator=generator)
            # Input shape is [batch, 1, channels, time].
            x[:, :, channel_idx, :] = x[perm, :, channel_idx, :]

        x = x.to(device)
        y = y.long().to(device)
        logits = model(x)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_examples += y.size(0)

    return total_correct / total_examples


def load_checkpoint_state_dict(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Unsupported checkpoint format at {checkpoint_path}: "
            f"expected a dict, got {type(checkpoint).__name__}"
        )

    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def write_csv(rows: list[dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_importance_barplot(
    rows_sorted: list[dict[str, object]],
    output_path: Path,
    title: str,
) -> None:
    channel_names = [str(row["channel_name"]) for row in rows_sorted]
    importances = [float(row["importance"]) for row in rows_sorted]

    fig_height = max(6.0, len(rows_sorted) * 0.35)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.barh(channel_names, importances, color="#4c78a8")
    ax.invert_yaxis()
    ax.set_xlabel("Permutation importance")
    ax.set_ylabel("Channel")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.data_root.exists():
        raise FileNotFoundError(f"Dataset not found at {args.data_root.resolve()}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint.resolve()}")

    seed_everything(args.seed)
    device = get_device(args.device)
    dataset = build_dataset(args.data_root, verbose=True)
    _, val_loader = make_loaders(
        dataset,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        limit_samples=args.limit_samples,
    )

    model = build_model().to(device)
    state_dict = load_checkpoint_state_dict(args.checkpoint, device)
    model.load_state_dict(state_dict)

    base_acc = accuracy_with_optional_permutation(model, val_loader, device)
    rows = []
    for channel_idx in range(NUM_ELECTRODES):
        perm_acc = accuracy_with_optional_permutation(model, val_loader, device, channel_idx)
        importance = base_acc - perm_acc
        rows.append(
            {
                "rank": 0,
                "channel_index": channel_idx,
                "channel_name": BCI_IV_2A_CHANNELS[channel_idx],
                "base_accuracy": base_acc,
                "permuted_accuracy": perm_acc,
                "importance": importance,
            }
        )
        print(
            f"channel={channel_idx:02d} {BCI_IV_2A_CHANNELS[channel_idx]:>3s} "
            f"permuted_acc={perm_acc:.4f} importance={importance:.4f}"
        )

    rows.sort(key=lambda row: row["importance"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    rows_standard_order = sorted(rows, key=lambda row: int(row["channel_index"]))

    write_csv(rows_standard_order, args.output_csv)
    if args.output_sorted_csv is not None:
        write_csv(rows, args.output_sorted_csv)
    if args.output_barplot is not None:
        plot_importance_barplot(rows, args.output_barplot, args.plot_title)

    print(f"base_accuracy={base_acc:.4f}")
    print(f"saved CSV: {args.output_csv}")
    if args.output_sorted_csv is not None:
        print(f"saved sorted CSV: {args.output_sorted_csv}")
    if args.output_barplot is not None:
        print(f"saved bar plot: {args.output_barplot}")


if __name__ == "__main__":
    main()
