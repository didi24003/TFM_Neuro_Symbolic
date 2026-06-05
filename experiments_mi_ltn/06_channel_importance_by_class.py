#!/usr/bin/env python
"""Estimate EEG channel permutation importance per class for baseline and logic EEGNet."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from mi_ltn_common import (
    BCI_IV_2A_CHANNELS,
    DEFAULT_DATA_ROOT,
    NUM_CLASSES,
    NUM_ELECTRODES,
    RUNS_DIR,
    build_dataset,
    build_model,
    get_device,
    make_loaders,
    seed_everything,
)


CLASS_NAMES = {
    0: "left_hand",
    1: "right_hand",
    2: "feet",
    3: "tongue",
}

FIELDNAMES = [
    "model_name",
    "class_id",
    "class_name",
    "rank",
    "channel_index",
    "channel_name",
    "class_base_accuracy",
    "permuted_accuracy",
    "importance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=RUNS_DIR / "best_eegnet_bciciv2a.pt",
        help="Baseline EEGNet checkpoint.",
    )
    parser.add_argument(
        "--logic-checkpoint",
        type=Path,
        default=RUNS_DIR / "best_eegnet_logic_bciciv2a.pt",
        help="Logic-loss EEGNet checkpoint.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("baseline", "logic"),
        default=("baseline", "logic"),
        help="Models to evaluate.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=RUNS_DIR / "channel_importance_by_class.csv",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-samples", type=int, default=None)
    return parser.parse_args()


def checkpoint_specs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    paths = {
        "baseline": args.baseline_checkpoint,
        "logic": args.logic_checkpoint,
    }
    return [(model_name, paths[model_name]) for model_name in args.models]


@torch.no_grad()
def accuracy_with_optional_permutation(
    model: torch.nn.Module,
    x_class: torch.Tensor,
    y_class: torch.Tensor,
    device: torch.device,
    batch_size: int,
    class_id: int,
    channel_idx: int | None = None,
) -> float:
    model.eval()
    x_eval = x_class.clone()
    y_eval = y_class.long()

    if channel_idx is not None and x_eval.size(0) > 1:
        generator = torch.Generator(device="cpu").manual_seed(
            10_000 + class_id * 1_000 + channel_idx
        )
        perm = torch.randperm(x_eval.size(0), generator=generator)
        # Input shape is [examples, 1, channels, time].
        x_eval[:, :, channel_idx, :] = x_eval[perm, :, channel_idx, :]

    loader = DataLoader(
        TensorDataset(x_eval, y_eval),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    total_correct = 0
    total_examples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.long().to(device)
        logits = model(x)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_examples += y.size(0)

    return total_correct / total_examples


def collect_class_examples(loader: DataLoader, class_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    x_parts = []
    y_parts = []

    for x, y in loader:
        y = y.long()
        class_mask = y == class_id
        if class_mask.any():
            x_parts.append(x[class_mask].cpu())
            y_parts.append(y[class_mask].cpu())

    if not x_parts:
        raise ValueError(
            f"No validation samples found for class_id={class_id} "
            f"({CLASS_NAMES.get(class_id, 'unknown')})."
        )

    return torch.cat(x_parts, dim=0), torch.cat(y_parts, dim=0)


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


def validate_args(args: argparse.Namespace) -> None:
    if not args.data_root.exists():
        raise FileNotFoundError(f"Dataset not found at {args.data_root.resolve()}")
    for model_name, checkpoint in checkpoint_specs(args):
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"{model_name} checkpoint not found at {checkpoint.resolve()}"
            )
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.limit_samples is not None and args.limit_samples < 1:
        raise ValueError("--limit-samples must be >= 1 when provided.")


def main() -> None:
    args = parse_args()
    validate_args(args)

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

    class_examples = {
        class_id: collect_class_examples(val_loader, class_id)
        for class_id in CLASS_NAMES
    }
    rows = []

    for model_name, checkpoint in checkpoint_specs(args):
        print(f"model={model_name} checkpoint={checkpoint}")
        model = build_model().to(device)
        state_dict = load_checkpoint_state_dict(checkpoint, device)
        model.load_state_dict(state_dict)

        for class_id, class_name in CLASS_NAMES.items():
            x_class, y_class = class_examples[class_id]
            base_acc = accuracy_with_optional_permutation(
                model,
                x_class,
                y_class,
                device,
                args.batch_size,
                class_id,
            )
            print(
                f"class={class_id} {class_name} "
                f"n_val={y_class.numel()} class_base_accuracy={base_acc:.4f}"
            )

            class_rows = []
            for channel_idx in range(NUM_ELECTRODES):
                perm_acc = accuracy_with_optional_permutation(
                    model,
                    x_class,
                    y_class,
                    device,
                    args.batch_size,
                    class_id,
                    channel_idx,
                )
                importance = base_acc - perm_acc
                class_rows.append(
                    {
                        "model_name": model_name,
                        "class_id": class_id,
                        "class_name": class_name,
                        "rank": 0,
                        "channel_index": channel_idx,
                        "channel_name": BCI_IV_2A_CHANNELS[channel_idx],
                        "class_base_accuracy": base_acc,
                        "permuted_accuracy": perm_acc,
                        "importance": importance,
                    }
                )
                print(
                    f"  channel={channel_idx:02d} {BCI_IV_2A_CHANNELS[channel_idx]:>3s} "
                    f"permuted_acc={perm_acc:.4f} importance={importance:.4f}"
                )

            class_rows.sort(key=lambda row: row["importance"], reverse=True)
            for rank, row in enumerate(class_rows, start=1):
                row["rank"] = rank
            rows.extend(class_rows)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    expected_rows = len(checkpoint_specs(args)) * NUM_CLASSES * NUM_ELECTRODES
    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, wrote {len(rows)} rows.")

    print(f"saved per-class ranking: {args.output_csv}")


if __name__ == "__main__":
    main()
