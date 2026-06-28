#!/usr/bin/env python
"""Train EEGNet with a channel-gate auxiliary rule loss on BCI Competition IV 2a."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from mi_ltn_common import (
    CHUNK_SIZE,
    DEFAULT_DATA_ROOT,
    NUM_CLASSES,
    NUM_ELECTRODES,
    RUNS_DIR,
    build_dataset,
    evaluate,
    get_device,
    load_eegnet_class,
    make_loaders,
    seed_everything,
)


CANONICAL_CHANNEL_ORDER = [
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
CHANNEL_ORDER_FILE = Path(__file__).resolve().parent / "bciciv2a_channel_order.txt"
SUMMARY_CSV = RUNS_DIR / "eegnet_channel_rule_loss" / "channel_rule_runs_summary.csv"
SUMMARY_FIELDNAMES = [
    "run_id",
    "lambda_rule",
    "seed",
    "epochs",
    "optimizer",
    "lr",
    "batch_size",
    "weight_decay",
    "scheduler",
    "early_stopping_patience",
    "checkpoint_every",
    "best_val_acc",
    "best_epoch",
    "final_train_acc",
    "final_val_acc",
    "final_r_sm",
    "final_r_post",
    "final_r_diff",
    "checkpoint_best_path",
    "checkpoint_last_path",
    "history_csv",
    "summary_json",
    "args_json",
    "channel_gates_csv",
    "training_curves_path",
    "channel_gates_figure_path",
    "status",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lambda-rule", type=float, default=0.1)
    parser.add_argument("--target-sm", type=float, default=0.45)
    parser.add_argument("--target-post", type=float, default=0.20)
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-samples", type=int, default=None, help="Optional quick-debug subset size.")
    parser.add_argument("--run-root", type=Path, default=RUNS_DIR / "eegnet_channel_rule_loss")
    parser.add_argument("--run-id", default=None, help="Optional explicit run folder name.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional extra best checkpoint path.")
    parser.add_argument("--scheduler", choices=("none",), default="none")
    parser.add_argument("--early-stopping-patience", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--channel-order-file", type=Path, default=CHANNEL_ORDER_FILE)
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset/model/channel order without training.")
    return parser.parse_args()


class ChannelGate(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.raw_gate = nn.Parameter(torch.zeros(num_channels))

    def gate_values(self) -> torch.Tensor:
        return 2.0 * torch.sigmoid(self.raw_gate)

    def normalized_gate_values(self) -> torch.Tensor:
        gate = self.gate_values()
        return gate / gate.sum().clamp_min(torch.finfo(gate.dtype).eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected EEG input with 4 dims [B, 1, C, T], got shape {tuple(x.shape)}")
        gate = self.gate_values().view(1, 1, -1, 1)
        return x * gate


class EEGNetChannelGate(nn.Module):
    def __init__(self, num_channels: int, chunk_size: int, num_classes: int):
        super().__init__()
        EEGNet = load_eegnet_class()
        self.channel_gate = ChannelGate(num_channels)
        self.backbone = EEGNet(
            chunk_size=chunk_size,
            num_electrodes=num_channels,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.channel_gate(x))

    def gate_values(self) -> torch.Tensor:
        return self.channel_gate.gate_values()

    def normalized_gate_values(self) -> torch.Tensor:
        return self.channel_gate.normalized_gate_values()


def lambda_tag(value: float) -> str:
    text = f"{value:.12f}".rstrip("0").rstrip(".")
    if "." not in text:
        text = f"{text}.0"
    return text


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_channel_order(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Channel-order file not found: {path.resolve()}. "
            "Refusing to assume BCI IV 2a channel indices."
        )
    order = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if order != CANONICAL_CHANNEL_ORDER:
        raise ValueError(
            "Channel-order file does not exactly match the canonical BCI IV 2a order expected by this experiment.\n"
            f"Expected: {CANONICAL_CHANNEL_ORDER}\n"
            f"Found:    {order}"
        )
    if len(order) != NUM_ELECTRODES or len(set(order)) != NUM_ELECTRODES:
        raise ValueError("Channel-order file must contain exactly 22 unique EEG channel names.")
    return order


def verify_dataset_shape(dataset, expected_channels: list[str]) -> tuple[int, ...]:
    sample_x, _ = dataset[0]
    sample_shape = tuple(sample_x.shape)
    if len(sample_shape) != 3:
        raise ValueError(
            f"Unexpected sample shape {sample_shape}. Expected transformed EEG shaped like [1, 22, {CHUNK_SIZE}]."
        )
    if sample_shape[1] != len(expected_channels):
        raise ValueError(
            f"Dataset sample has {sample_shape[1]} channels but the verified channel order has {len(expected_channels)}."
        )
    if sample_shape[2] != CHUNK_SIZE:
        raise ValueError(
            f"Dataset sample has chunk size {sample_shape[2]} but EEGNet is configured for {CHUNK_SIZE}."
        )
    return sample_shape


def channel_indices(order: list[str], names: list[str], group_name: str) -> list[int]:
    missing = [name for name in names if name not in order]
    if missing:
        raise ValueError(f"Missing {group_name} channels in verified BCI IV 2a order: {missing}")
    return [order.index(name) for name in names]


def rule_terms(
    gate_norm: torch.Tensor,
    sm_indices: list[int],
    post_indices: list[int],
    target_sm: float,
    target_post: float,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r_sm = gate_norm[sm_indices].sum()
    r_post = gate_norm[post_indices].sum()
    rule_loss = (
        torch.relu(gate_norm.new_tensor(target_sm) - r_sm)
        + torch.relu(r_post - gate_norm.new_tensor(target_post))
        + torch.relu(gate_norm.new_tensor(margin) - (r_sm - r_post))
    )
    return rule_loss, r_sm, r_post, r_sm - r_post


def compute_gate_metrics(
    model: EEGNetChannelGate,
    sm_indices: list[int],
    post_indices: list[int],
    target_sm: float,
    target_post: float,
    margin: float,
) -> dict[str, float]:
    gate = model.gate_values().detach().cpu()
    gate_norm = model.normalized_gate_values().detach().cpu()
    rule_loss, r_sm, r_post, r_diff = rule_terms(
        gate_norm,
        sm_indices,
        post_indices,
        target_sm,
        target_post,
        margin,
    )
    return {
        "rule_loss": float(rule_loss.item()),
        "r_sm": float(r_sm.item()),
        "r_post": float(r_post.item()),
        "r_diff": float(r_diff.item()),
        "gate_sum": float(gate.sum().item()),
    }


def train_one_epoch(
    model: EEGNetChannelGate,
    loader,
    ce_loss,
    optimizer,
    device: torch.device,
    lambda_rule: float,
    sm_indices: list[int],
    post_indices: list[int],
    target_sm: float,
    target_post: float,
    margin: float,
):
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_rule = 0.0
    total_correct = 0
    total_examples = 0
    non_blocking = device.type == "cuda"

    for x, y in loader:
        x = x.to(device, non_blocking=non_blocking)
        y = y.long().to(device, non_blocking=non_blocking)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        ce = ce_loss(logits, y)
        gate_norm = model.normalized_gate_values()
        rule_loss, _, _, _ = rule_terms(
            gate_norm,
            sm_indices,
            post_indices,
            target_sm,
            target_post,
            margin,
        )
        loss = ce + lambda_rule * rule_loss
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_ce += ce.item() * batch_size
        total_rule += rule_loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_examples += batch_size

    return (
        total_loss / total_examples,
        total_ce / total_examples,
        total_rule / total_examples,
        total_correct / total_examples,
    )


def evaluate_with_rule(
    model: EEGNetChannelGate,
    loader,
    device: torch.device,
    lambda_rule: float,
    sm_indices: list[int],
    post_indices: list[int],
    target_sm: float,
    target_post: float,
    margin: float,
):
    ce_loss, acc = evaluate(model, loader, device)
    gate_norm = model.normalized_gate_values().detach().cpu()
    rule_loss, r_sm, r_post, r_diff = rule_terms(
        gate_norm,
        sm_indices,
        post_indices,
        target_sm,
        target_post,
        margin,
    )
    total_loss = ce_loss + lambda_rule * float(rule_loss.item())
    return {
        "val_loss": float(total_loss),
        "val_ce": float(ce_loss),
        "val_rule": float(rule_loss.item()),
        "val_acc": float(acc),
        "r_sm": float(r_sm.item()),
        "r_post": float(r_post.item()),
        "r_diff": float(r_diff.item()),
    }


def make_run_dir(args: argparse.Namespace) -> Path:
    args.run_root.mkdir(parents=True, exist_ok=True)
    if args.run_id:
        base_run_id = args.run_id
    else:
        base_run_id = f"lambda_{lambda_tag(args.lambda_rule)}_seed{args.seed}"
    run_dir = args.run_root / base_run_id
    if run_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = args.run_root / f"{base_run_id}_{timestamp}"
    run_dir.mkdir(parents=True)
    (run_dir / "figures").mkdir()
    (run_dir / "checkpoints").mkdir()
    return run_dir


def write_args(args: argparse.Namespace, run_dir: Path) -> Path:
    args_path = run_dir / "args.json"
    with args_path.open("w") as f:
        json.dump(vars(args), f, indent=2, default=json_default)
    return args_path


def init_history(history_path: Path) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "train_ce",
        "train_rule",
        "train_acc",
        "val_loss",
        "val_ce",
        "val_rule",
        "val_acc",
        "lr",
        "r_sm",
        "r_post",
        "r_diff",
    ]
    with history_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_history_row(row: dict[str, float | int], history_path: Path) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "train_ce",
        "train_rule",
        "train_acc",
        "val_loss",
        "val_ce",
        "val_rule",
        "val_acc",
        "lr",
        "r_sm",
        "r_post",
        "r_diff",
    ]
    with history_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def final_metrics(history: list[dict[str, float | int]]) -> tuple[float | None, float | None]:
    if not history:
        return None, None
    last = history[-1]
    return float(last["train_acc"]), float(last["val_acc"])


def final_losses(history: list[dict[str, float | int]]) -> tuple[float | None, float | None]:
    if not history:
        return None, None
    last = history[-1]
    return float(last["train_loss"]), float(last["val_loss"])


def checkpoint_payload(
    args: argparse.Namespace,
    model: EEGNetChannelGate,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_acc: float,
    best_epoch: int,
    channel_order: list[str],
    sm_indices: list[int],
    post_indices: list[int],
) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "epoch": epoch,
        "lambda_rule": args.lambda_rule,
        "channel_order": channel_order,
        "sensorimotor_indices": sm_indices,
        "posterior_indices": post_indices,
        "args": vars(args),
    }


def save_checkpoint(
    path: Path,
    args: argparse.Namespace,
    model: EEGNetChannelGate,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_acc: float,
    best_epoch: int,
    channel_order: list[str],
    sm_indices: list[int],
    post_indices: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        checkpoint_payload(
            args,
            model,
            optimizer,
            epoch,
            best_val_acc,
            best_epoch,
            channel_order,
            sm_indices,
            post_indices,
        ),
        path,
    )


def write_channel_gates_csv(
    output_path: Path,
    channel_order: list[str],
    model: EEGNetChannelGate,
    sm_indices: list[int],
    post_indices: list[int],
) -> None:
    gate = model.gate_values().detach().cpu().tolist()
    gate_norm = model.normalized_gate_values().detach().cpu().tolist()
    sm_set = set(sm_indices)
    post_set = set(post_indices)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["channel_index", "channel_name", "gate", "gate_norm", "group"],
        )
        writer.writeheader()
        for idx, channel_name in enumerate(channel_order):
            if idx in sm_set:
                group = "sensorimotor"
            elif idx in post_set:
                group = "posterior"
            else:
                group = "other"
            writer.writerow(
                {
                    "channel_index": idx,
                    "channel_name": channel_name,
                    "gate": gate[idx],
                    "gate_norm": gate_norm[idx],
                    "group": group,
                }
            )


def plot_history(history: list[dict[str, float | int]], output_path: Path) -> None:
    mpl_config_dir = output_path.parent.parent / ".matplotlib"
    mpl_config_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train total")
    axes[0].plot(epochs, [row["train_ce"] for row in history], label="train CE")
    axes[0].plot(epochs, [row["train_rule"] for row in history], label="train rule")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val total")
    axes[0].plot(epochs, [row["val_ce"] for row in history], label="val CE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Channel-rule loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, [row["train_acc"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend(frameon=False)

    axes[2].plot(epochs, [row["r_sm"] for row in history], label="R_SM")
    axes[2].plot(epochs, [row["r_post"] for row in history], label="R_POST")
    axes[2].plot(epochs, [row["r_diff"] for row in history], label="R_SM - R_POST")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Gate mass")
    axes[2].set_title("Rule statistics")
    axes[2].grid(alpha=0.3)
    axes[2].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_channel_gates(csv_path: Path, output_path: Path) -> None:
    mpl_config_dir = output_path.parent.parent / ".matplotlib"
    mpl_config_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    labels = [row["channel_name"] for row in rows]
    values = [float(row["gate_norm"]) for row in rows]
    colors = []
    for row in rows:
        if row["group"] == "sensorimotor":
            colors.append("#2a9d8f")
        elif row["group"] == "posterior":
            colors.append("#e76f51")
        else:
            colors.append("#8d99ae")

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.bar(range(len(labels)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("Normalized gate weight")
    ax.set_title("Final channel gates")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_summary(
    args: argparse.Namespace,
    run_dir: Path,
    checkpoint_best_path: Path,
    checkpoint_last_path: Path,
    history_path: Path,
    args_path: Path,
    figure_path: Path,
    channel_gates_csv: Path,
    channel_gates_figure: Path,
    best_val_acc: float,
    best_epoch: int,
    epochs_completed: int,
    stopped_early: bool,
    status: str,
    channel_order: list[str],
    gate_metrics: dict[str, float],
    history: list[dict[str, float | int]],
) -> None:
    final_train_acc, final_val_acc = final_metrics(history)
    final_train_loss, final_val_loss = final_losses(history)
    summary = {
        "model": "EEGNetChannelGate",
        "block": "eegnet_channel_rule_loss",
        "seed": args.seed,
        "device": str(get_device(args.device)),
        "optimizer": "Adam",
        "epochs_requested": args.epochs,
        "epochs_completed": epochs_completed,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "early_stopping_patience": args.early_stopping_patience,
        "checkpoint_every": args.checkpoint_every,
        "lambda_rule": args.lambda_rule,
        "target_sm": args.target_sm,
        "target_post": args.target_post,
        "margin": args.margin,
        "channel_order": channel_order,
        "sensorimotor_channels": SENSORIMOTOR_CHANNELS,
        "posterior_channels": POSTERIOR_CHANNELS,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "final_train_acc": final_train_acc,
        "final_val_acc": final_val_acc,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "R_SM": gate_metrics["r_sm"],
        "R_POST": gate_metrics["r_post"],
        "R_SM_minus_POST": gate_metrics["r_diff"],
        "final_r_sm": gate_metrics["r_sm"],
        "final_r_post": gate_metrics["r_post"],
        "final_r_diff": gate_metrics["r_diff"],
        "final_rule_loss": gate_metrics["rule_loss"],
        "best_checkpoint_path": str(checkpoint_best_path),
        "last_checkpoint_path": str(checkpoint_last_path),
        "checkpoint_best_path": str(checkpoint_best_path),
        "checkpoint_last_path": str(checkpoint_last_path),
        "checkpoint_path": str(checkpoint_best_path),
        "history_csv": str(history_path),
        "summary_json": str(run_dir / "summary.json"),
        "args_json": str(args_path),
        "training_curves_path": str(figure_path),
        "channel_gates_csv": str(channel_gates_csv),
        "channel_gates_figure_path": str(channel_gates_figure),
        "run_dir": str(run_dir),
        "stopped_early": stopped_early,
        "status": status,
    }
    with (run_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


def summary_row(
    args: argparse.Namespace,
    run_dir: Path,
    checkpoint_best_path: Path,
    checkpoint_last_path: Path,
    history_path: Path,
    args_path: Path,
    figure_path: Path,
    channel_gates_csv: Path,
    channel_gates_figure: Path,
    best_val_acc: float,
    best_epoch: int,
    history: list[dict[str, float | int]],
    status: str,
    gate_metrics: dict[str, float],
    notes: str = "",
) -> dict:
    final_train_acc, final_val_acc = final_metrics(history)
    return {
        "run_id": run_dir.name,
        "lambda_rule": args.lambda_rule,
        "seed": args.seed,
        "epochs": len(history),
        "optimizer": "Adam",
        "lr": args.learning_rate,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "early_stopping_patience": args.early_stopping_patience,
        "checkpoint_every": args.checkpoint_every,
        "best_val_acc": best_val_acc if best_val_acc >= 0 else "",
        "best_epoch": best_epoch if best_epoch > 0 else "",
        "final_train_acc": "" if final_train_acc is None else final_train_acc,
        "final_val_acc": "" if final_val_acc is None else final_val_acc,
        "final_r_sm": gate_metrics["r_sm"],
        "final_r_post": gate_metrics["r_post"],
        "final_r_diff": gate_metrics["r_diff"],
        "checkpoint_best_path": str(checkpoint_best_path),
        "checkpoint_last_path": str(checkpoint_last_path),
        "history_csv": str(history_path),
        "summary_json": str(run_dir / "summary.json"),
        "args_json": str(args_path),
        "channel_gates_csv": str(channel_gates_csv),
        "training_curves_path": str(figure_path),
        "channel_gates_figure_path": str(channel_gates_figure),
        "status": status,
        "notes": notes,
    }


def upsert_summary_csv(row: dict, output_csv: Path = SUMMARY_CSV) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if output_csv.exists():
        with output_csv.open(newline="") as f:
            rows = list(csv.DictReader(f))

    rows = [existing for existing in rows if existing.get("run_id") != row["run_id"]]
    rows.append(row)
    rows.sort(key=lambda existing: existing.get("run_id", ""))

    tmp_path = output_csv.with_suffix(output_csv.suffix + ".tmp")
    with tmp_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    shutil.move(tmp_path, output_csv)


def dry_run(args: argparse.Namespace) -> None:
    channel_order = load_channel_order(args.channel_order_file)
    sm_indices = channel_indices(channel_order, SENSORIMOTOR_CHANNELS, "sensorimotor")
    post_indices = channel_indices(channel_order, POSTERIOR_CHANNELS, "posterior")
    if not args.data_root.exists():
        raise FileNotFoundError(f"Dataset not found at {args.data_root.resolve()}")
    dataset = build_dataset(args.data_root, verbose=True)
    sample_shape = verify_dataset_shape(dataset, channel_order)
    model = EEGNetChannelGate(NUM_ELECTRODES, CHUNK_SIZE, NUM_CLASSES)
    sample_x, _ = dataset[0]
    with torch.inference_mode():
        logits = model(sample_x.unsqueeze(0))
    gate_metrics = compute_gate_metrics(
        model,
        sm_indices,
        post_indices,
        args.target_sm,
        args.target_post,
        args.margin,
    )
    print(f"channel_order_verified={channel_order}")
    print(f"sensorimotor_indices={sm_indices}")
    print(f"posterior_indices={post_indices}")
    print(f"sample_shape={sample_shape}")
    print(f"logits_shape={tuple(logits.shape)}")
    print(f"initial_r_sm={gate_metrics['r_sm']:.6f}")
    print(f"initial_r_post={gate_metrics['r_post']:.6f}")
    print(f"initial_r_diff={gate_metrics['r_diff']:.6f}")


def main() -> None:
    args = parse_args()
    if args.dry_run:
        dry_run(args)
        return

    if not args.data_root.exists():
        raise FileNotFoundError(f"Dataset not found at {args.data_root.resolve()}")

    channel_order = load_channel_order(args.channel_order_file)
    sm_indices = channel_indices(channel_order, SENSORIMOTOR_CHANNELS, "sensorimotor")
    post_indices = channel_indices(channel_order, POSTERIOR_CHANNELS, "posterior")

    seed_everything(args.seed)
    device = get_device(args.device)
    dataset = build_dataset(args.data_root, verbose=True)
    verify_dataset_shape(dataset, channel_order)
    train_loader, val_loader = make_loaders(
        dataset,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        limit_samples=args.limit_samples,
    )

    run_dir = make_run_dir(args)
    checkpoint_best_path = run_dir / "checkpoint_best.pt"
    checkpoint_last_path = run_dir / "checkpoint_last.pt"
    history_path = run_dir / "history.csv"
    figure_path = run_dir / "figures" / "training_curves.png"
    channel_gates_csv = run_dir / "channel_gates.csv"
    channel_gates_figure = run_dir / "figures" / "channel_gates.png"
    args_path = write_args(args, run_dir)
    init_history(history_path)

    model = EEGNetChannelGate(NUM_ELECTRODES, CHUNK_SIZE, NUM_CLASSES).to(device)
    ce_loss = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val_acc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    history = []
    status = "running"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_ce, train_rule, train_acc = train_one_epoch(
            model,
            train_loader,
            ce_loss,
            optimizer,
            device,
            args.lambda_rule,
            sm_indices,
            post_indices,
            args.target_sm,
            args.target_post,
            args.margin,
        )
        val_metrics = evaluate_with_rule(
            model,
            val_loader,
            device,
            args.lambda_rule,
            sm_indices,
            post_indices,
            args.target_sm,
            args.target_post,
            args.margin,
        )
        lr = optimizer.param_groups[0]["lr"]
        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ce": train_ce,
            "train_rule": train_rule,
            "train_acc": train_acc,
            "val_loss": val_metrics["val_loss"],
            "val_ce": val_metrics["val_ce"],
            "val_rule": val_metrics["val_rule"],
            "val_acc": val_metrics["val_acc"],
            "lr": lr,
            "r_sm": val_metrics["r_sm"],
            "r_post": val_metrics["r_post"],
            "r_diff": val_metrics["r_diff"],
        }
        history.append(history_row)
        append_history_row(history_row, history_path)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_ce={train_ce:.4f} train_rule={train_rule:.4f} "
            f"train_acc={train_acc:.4f} val_loss={val_metrics['val_loss']:.4f} "
            f"val_ce={val_metrics['val_ce']:.4f} val_rule={val_metrics['val_rule']:.4f} "
            f"val_acc={val_metrics['val_acc']:.4f} R_SM={val_metrics['r_sm']:.4f} "
            f"R_POST={val_metrics['r_post']:.4f} margin={val_metrics['r_diff']:.4f} lr={lr:.6g}"
        )

        if val_metrics["val_acc"] > best_val_acc:
            best_val_acc = val_metrics["val_acc"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_best_path,
                args,
                model,
                optimizer,
                epoch,
                best_val_acc,
                best_epoch,
                channel_order,
                sm_indices,
                post_indices,
            )
            if args.checkpoint is not None:
                save_checkpoint(
                    args.checkpoint,
                    args,
                    model,
                    optimizer,
                    epoch,
                    best_val_acc,
                    best_epoch,
                    channel_order,
                    sm_indices,
                    post_indices,
                )
            print(f"saved best channel-rule checkpoint: {checkpoint_best_path}")
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            checkpoint_last_path,
            args,
            model,
            optimizer,
            epoch,
            best_val_acc,
            best_epoch,
            channel_order,
            sm_indices,
            post_indices,
        )
        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            periodic_path = run_dir / "checkpoints" / f"checkpoint_epoch_{epoch:03d}.pt"
            save_checkpoint(
                periodic_path,
                args,
                model,
                optimizer,
                epoch,
                best_val_acc,
                best_epoch,
                channel_order,
                sm_indices,
                post_indices,
            )
            print(f"saved periodic checkpoint: {periodic_path}")

        gate_metrics = compute_gate_metrics(
            model,
            sm_indices,
            post_indices,
            args.target_sm,
            args.target_post,
            args.margin,
        )
        write_summary(
            args,
            run_dir,
            checkpoint_best_path,
            checkpoint_last_path,
            history_path,
            args_path,
            figure_path,
            channel_gates_csv,
            channel_gates_figure,
            best_val_acc,
            best_epoch,
            len(history),
            stopped_early,
            status,
            channel_order,
            gate_metrics,
            history,
        )
        upsert_summary_csv(
            summary_row(
                args,
                run_dir,
                checkpoint_best_path,
                checkpoint_last_path,
                history_path,
                args_path,
                figure_path,
                channel_gates_csv,
                channel_gates_figure,
                best_val_acc,
                best_epoch,
                history,
                status,
                gate_metrics,
            )
        )

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            stopped_early = True
            status = "early_stopped"
            print(
                "early stopping: "
                f"no validation improvement for {epochs_without_improvement} epochs"
            )
            break

    if status == "running":
        status = "completed"

    gate_metrics = compute_gate_metrics(
        model,
        sm_indices,
        post_indices,
        args.target_sm,
        args.target_post,
        args.margin,
    )
    write_channel_gates_csv(channel_gates_csv, channel_order, model, sm_indices, post_indices)
    plot_history(history, figure_path)
    plot_channel_gates(channel_gates_csv, channel_gates_figure)
    write_summary(
        args,
        run_dir,
        checkpoint_best_path,
        checkpoint_last_path,
        history_path,
        args_path,
        figure_path,
        channel_gates_csv,
        channel_gates_figure,
        best_val_acc,
        best_epoch,
        len(history),
        stopped_early,
        status,
        channel_order,
        gate_metrics,
        history,
    )
    upsert_summary_csv(
        summary_row(
            args,
            run_dir,
            checkpoint_best_path,
            checkpoint_last_path,
            history_path,
            args_path,
            figure_path,
            channel_gates_csv,
            channel_gates_figure,
            best_val_acc,
            best_epoch,
            history,
            status,
            gate_metrics,
        )
    )

    print(f"run_dir={run_dir}")
    print(f"best_val_acc={best_val_acc:.4f}")
    print(f"best_epoch={best_epoch}")
    print(f"training_curves={figure_path}")
    print(f"channel_gates_csv={channel_gates_csv}")
    print(f"channel_gates_figure={channel_gates_figure}")
    print(f"summary_csv={SUMMARY_CSV}")


if __name__ == "__main__":
    main()
