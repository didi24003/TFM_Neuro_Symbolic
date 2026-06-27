#!/usr/bin/env python
"""Compute final channel permutation importance by class for baseline and logic models."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
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

SENSORIMOTOR_CHANNELS = {"FC3", "FC4", "FCz", "C3", "C4", "Cz", "CP3", "CP4", "CPz"}

PER_MODEL_FIELDNAMES = [
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

COMPARISON_COLUMNS = [
    "class_id",
    "class_name",
    "channel_index",
    "channel",
    "baseline_rank",
    "logic_rank",
    "baseline_class_base_accuracy",
    "logic_class_base_accuracy",
    "baseline_permuted_accuracy",
    "logic_permuted_accuracy",
    "baseline_importance",
    "logic_importance",
    "delta_logic_minus_baseline",
    "abs_delta",
    "is_sensorimotor_channel",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        required=True,
        help="Final baseline EEGNet checkpoint.",
    )
    parser.add_argument(
        "--logic-checkpoint",
        type=Path,
        required=True,
        help="Final logic-loss EEGNet checkpoint.",
    )
    parser.add_argument(
        "--baseline-output-csv",
        type=Path,
        default=RUNS_DIR / "final_channel_analysis" / "baseline" / "channel_importance_by_class.csv",
    )
    parser.add_argument(
        "--logic-output-csv",
        type=Path,
        default=RUNS_DIR / "final_channel_analysis" / "logic_lam1p0" / "channel_importance_by_class.csv",
    )
    parser.add_argument(
        "--comparison-output-csv",
        type=Path,
        default=RUNS_DIR / "final_channel_analysis" / "comparison" / "channel_importance_by_class_comparison.csv",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-samples", type=int, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.data_root.exists():
        raise FileNotFoundError(f"Dataset not found at {args.data_root.resolve()}")
    for label, path in (
        ("Baseline checkpoint", args.baseline_checkpoint),
        ("Logic checkpoint", args.logic_checkpoint),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found at {path.resolve()}")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.limit_samples is not None and args.limit_samples < 1:
        raise ValueError("--limit-samples must be >= 1 when provided.")


def load_checkpoint_state_dict(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Unsupported checkpoint format at {checkpoint_path}: "
            f"expected dict, got {type(checkpoint).__name__}"
        )
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def collect_class_examples(loader: DataLoader, class_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    x_parts = []
    y_parts = []

    for x, y in loader:
        y = y.long()
        mask = y == class_id
        if mask.any():
            x_parts.append(x[mask].cpu())
            y_parts.append(y[mask].cpu())

    if not x_parts:
        raise ValueError(
            f"No validation samples found for class_id={class_id} "
            f"({CLASS_NAMES.get(class_id, 'unknown')})."
        )

    return torch.cat(x_parts, dim=0), torch.cat(y_parts, dim=0)


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
        x_eval[:, :, channel_idx, :] = x_eval[perm, :, channel_idx, :]

    loader = DataLoader(
        TensorDataset(x_eval, y_eval),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    total_correct = 0
    total_examples = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.long().to(device)
        logits = model(x_batch)
        total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total_examples += y_batch.size(0)

    return total_correct / total_examples


def compute_model_rows(
    model_name: str,
    checkpoint_path: Path,
    class_examples: dict[int, tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    batch_size: int,
) -> list[dict[str, object]]:
    model = build_model().to(device)
    model.load_state_dict(load_checkpoint_state_dict(checkpoint_path, device))

    rows: list[dict[str, object]] = []
    for class_id, class_name in CLASS_NAMES.items():
        x_class, y_class = class_examples[class_id]
        base_acc = accuracy_with_optional_permutation(
            model,
            x_class,
            y_class,
            device,
            batch_size,
            class_id,
        )

        class_rows = []
        for channel_idx in range(NUM_ELECTRODES):
            perm_acc = accuracy_with_optional_permutation(
                model,
                x_class,
                y_class,
                device,
                batch_size,
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

        class_rows.sort(key=lambda row: float(row["importance"]), reverse=True)
        for rank, row in enumerate(class_rows, start=1):
            row["rank"] = rank
        rows.extend(class_rows)

    return rows


def write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_MODEL_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def build_comparison_df(
    baseline_rows: list[dict[str, object]],
    logic_rows: list[dict[str, object]],
) -> pd.DataFrame:
    baseline_df = pd.DataFrame(baseline_rows).rename(
        columns={
            "rank": "baseline_rank",
            "class_base_accuracy": "baseline_class_base_accuracy",
            "permuted_accuracy": "baseline_permuted_accuracy",
            "importance": "baseline_importance",
        }
    )
    logic_df = pd.DataFrame(logic_rows).rename(
        columns={
            "rank": "logic_rank",
            "class_base_accuracy": "logic_class_base_accuracy",
            "permuted_accuracy": "logic_permuted_accuracy",
            "importance": "logic_importance",
        }
    )

    baseline_df = baseline_df[
        [
            "class_id",
            "class_name",
            "channel_index",
            "channel_name",
            "baseline_rank",
            "baseline_class_base_accuracy",
            "baseline_permuted_accuracy",
            "baseline_importance",
        ]
    ]
    logic_df = logic_df[
        [
            "class_id",
            "class_name",
            "channel_index",
            "channel_name",
            "logic_rank",
            "logic_class_base_accuracy",
            "logic_permuted_accuracy",
            "logic_importance",
        ]
    ]

    comparison = baseline_df.merge(
        logic_df,
        on=["class_id", "class_name", "channel_index", "channel_name"],
        how="inner",
        validate="one_to_one",
    )
    if comparison.empty:
        raise ValueError("No matching class/channel rows found for baseline and logic.")

    comparison["channel"] = comparison["channel_name"]
    comparison["delta_logic_minus_baseline"] = (
        comparison["logic_importance"] - comparison["baseline_importance"]
    )
    comparison["abs_delta"] = comparison["delta_logic_minus_baseline"].abs()
    comparison["is_sensorimotor_channel"] = comparison["channel"].isin(
        SENSORIMOTOR_CHANNELS
    )

    comparison = comparison.sort_values(
        ["class_id", "delta_logic_minus_baseline", "logic_importance"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    return comparison[COMPARISON_COLUMNS]


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

    baseline_rows = compute_model_rows(
        model_name="baseline",
        checkpoint_path=args.baseline_checkpoint,
        class_examples=class_examples,
        device=device,
        batch_size=args.batch_size,
    )
    logic_rows = compute_model_rows(
        model_name="logic",
        checkpoint_path=args.logic_checkpoint,
        class_examples=class_examples,
        device=device,
        batch_size=args.batch_size,
    )

    expected_rows = NUM_CLASSES * NUM_ELECTRODES
    if len(baseline_rows) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} baseline rows, got {len(baseline_rows)}"
        )
    if len(logic_rows) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} logic rows, got {len(logic_rows)}"
        )

    write_rows_csv(args.baseline_output_csv, baseline_rows)
    write_rows_csv(args.logic_output_csv, logic_rows)

    comparison_df = build_comparison_df(baseline_rows, logic_rows)
    args.comparison_output_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(args.comparison_output_csv, index=False)

    print(f"saved baseline per-class CSV: {args.baseline_output_csv}")
    print(f"saved logic per-class CSV: {args.logic_output_csv}")
    print(f"saved comparison per-class CSV: {args.comparison_output_csv}")


if __name__ == "__main__":
    main()
