#!/usr/bin/env python
"""Train EEGNet + channel gates with subject-specific cross-session protocol."""

from __future__ import annotations

import argparse
import csv
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, f1_score
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from mi_ltn_common import (
    BCI_IV_2A_CHANNELS,
    CHUNK_SIZE,
    DEFAULT_DATA_ROOT,
    NUM_CLASSES,
    NUM_ELECTRODES,
    RUNS_DIR,
    build_dataset,
    get_device,
    load_eegnet_class,
    seed_everything,
)
from subject_session_utils import save_subject_session_split, split_subject_session_indices, subject_ids, write_json


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_ROOT = RUNS_DIR / "eegnet_logic_subject_session"
DEFAULT_RUN_ROOT = RUN_ROOT / "logic_runs"
DEFAULT_SELECTION_METRIC = "val_acc"
SENSORIMOTOR_CHANNELS = ["FC3", "FC4", "FCz", "C3", "C4", "Cz", "CP3", "CP4", "CPz"]
POSTERIOR_CHANNELS = ["P1", "Pz", "P2", "POz"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--config-name", default="weight_decay")
    parser.add_argument("--phase-tag", default="logic_sweep")
    parser.add_argument(
        "--model-type",
        choices=("eegnet_baseline", "eegnet_gates_no_rule", "eegnet_gates_rule"),
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=9e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=("none", "plateau"), default="none")
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--val-mode", choices=("stratified_trialwise", "run_holdout"), default="stratified_trialwise")
    parser.add_argument("--run-holdout-index", type=int, default=None)
    parser.add_argument("--selection-metric", choices=("val_acc", "val_kappa"), default=DEFAULT_SELECTION_METRIC)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--subject-ids", nargs="*", default=None)
    parser.add_argument("--channel-order-file", type=Path, default=SCRIPT_DIR / "bciciv2a_channel_order.txt")
    parser.add_argument("--lambda-rule", type=float, default=0.0)
    parser.add_argument("--target-sm-mean", type=float, default=1.0)
    parser.add_argument("--target-post-mean", type=float, default=1.0)
    parser.add_argument("--margin-mean", type=float, default=0.05)
    return parser.parse_args()


class ChannelGate(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.raw_gate = nn.Parameter(torch.zeros(num_channels))

    def raw_values(self) -> torch.Tensor:
        return self.raw_gate

    def effective_values(self) -> torch.Tensor:
        return 2.0 * torch.sigmoid(self.raw_gate)

    def normalized_values(self) -> torch.Tensor:
        effective = self.effective_values()
        return effective / effective.sum().clamp_min(torch.finfo(effective.dtype).eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input [B, 1, C, T], got {tuple(x.shape)}")
        return x * self.effective_values().view(1, 1, -1, 1)


class EEGNetChannelGate(nn.Module):
    def __init__(self):
        super().__init__()
        EEGNet = load_eegnet_class()
        self.channel_gate = ChannelGate(NUM_ELECTRODES)
        self.backbone = EEGNet(
            chunk_size=CHUNK_SIZE,
            num_electrodes=NUM_ELECTRODES,
            num_classes=NUM_CLASSES,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.channel_gate(x))

    def raw_gate_values(self) -> torch.Tensor:
        return self.channel_gate.raw_values()

    def effective_gate_values(self) -> torch.Tensor:
        return self.channel_gate.effective_values()

    def normalized_gate_values(self) -> torch.Tensor:
        return self.channel_gate.normalized_values()


def safe_kappa(labels: list[int], preds: list[int]) -> float | None:
    if not labels:
        return None
    try:
        return float(cohen_kappa_score(labels, preds))
    except Exception:
        return None


def safe_macro_f1(labels: list[int], preds: list[int]) -> float | None:
    if not labels:
        return None
    try:
        return float(f1_score(labels, preds, average="macro"))
    except Exception:
        return None


def resolve_split_seed(args: argparse.Namespace) -> int:
    return args.split_seed if args.split_seed is not None else args.seed


def validate_args(args: argparse.Namespace) -> None:
    if args.model_type == "eegnet_baseline" and args.lambda_rule != 0.0:
        raise ValueError("eegnet_baseline must use lambda_rule=0.0")
    if args.model_type == "eegnet_gates_no_rule" and args.lambda_rule != 0.0:
        raise ValueError("eegnet_gates_no_rule must use lambda_rule=0.0")
    if args.model_type == "eegnet_gates_rule" and args.lambda_rule <= 0.0:
        raise ValueError("eegnet_gates_rule must use lambda_rule > 0")


def load_channel_order(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Channel-order file not found: {path.resolve()}")
    order = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if order != BCI_IV_2A_CHANNELS:
        raise ValueError(
            "Channel-order file does not match the canonical BCI IV 2a order.\n"
            f"Expected: {BCI_IV_2A_CHANNELS}\n"
            f"Found:    {order}"
        )
    return order


def channel_indices(order: list[str], names: list[str], group_name: str) -> list[int]:
    missing = [name for name in names if name not in order]
    if missing:
        raise ValueError(f"Missing {group_name} channels: {missing}")
    return [order.index(name) for name in names]


def make_run_dir(args: argparse.Namespace) -> Path:
    args.run_root.mkdir(parents=True, exist_ok=True)
    if args.run_id:
        base_run_id = args.run_id
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        lambda_tag = str(args.lambda_rule).replace(".", "p")
        base_run_id = f"{args.model_type}_lam{lambda_tag}_seed{args.seed}_split{resolve_split_seed(args)}_{timestamp}"
    run_dir = args.run_root / base_run_id
    if run_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = args.run_root / f"{base_run_id}_{timestamp}"
    run_dir.mkdir(parents=True)
    (run_dir / "subjects").mkdir()
    return run_dir


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


def build_model(model_type: str) -> nn.Module:
    if model_type == "eegnet_baseline":
        EEGNet = load_eegnet_class()
        return EEGNet(chunk_size=CHUNK_SIZE, num_electrodes=NUM_ELECTRODES, num_classes=NUM_CLASSES)
    return EEGNetChannelGate()


def neutral_gate_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = torch.zeros(NUM_ELECTRODES)
    effective = torch.ones(NUM_ELECTRODES)
    normalized = effective / effective.sum()
    return raw, effective, normalized


def final_gate_state(model: nn.Module) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(model, EEGNetChannelGate):
        return (
            model.raw_gate_values().detach().cpu(),
            model.effective_gate_values().detach().cpu(),
            model.normalized_gate_values().detach().cpu(),
        )
    return neutral_gate_tensors()


def metric_triplet(values: torch.Tensor, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    subset = values[indices]
    return subset.sum(), subset.mean()


def gate_group_metrics(values: torch.Tensor, sm_idx: list[int], post_idx: list[int], other_idx: list[int]) -> dict[str, float]:
    sm_sum, sm_mean = metric_triplet(values, sm_idx)
    post_sum, post_mean = metric_triplet(values, post_idx)
    other_sum, other_mean = metric_triplet(values, other_idx)
    return {
        "R_SM_sum": float(sm_sum.item()),
        "R_POST_sum": float(post_sum.item()),
        "R_OTHER_sum": float(other_sum.item()),
        "R_SM_mean": float(sm_mean.item()),
        "R_POST_mean": float(post_mean.item()),
        "R_OTHER_mean": float(other_mean.item()),
        "R_SM_mean_minus_POST_mean": float((sm_mean - post_mean).item()),
    }


def rule_terms(
    effective_gates: torch.Tensor,
    sm_indices: list[int],
    post_indices: list[int],
    target_sm_mean: float,
    target_post_mean: float,
    margin_mean: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    other_indices = [idx for idx in range(len(effective_gates)) if idx not in set(sm_indices) | set(post_indices)]
    metrics = gate_group_metrics(effective_gates, sm_indices, post_indices, other_indices)
    r_sm_mean = effective_gates[sm_indices].mean()
    r_post_mean = effective_gates[post_indices].mean()
    rule_loss = (
        torch.relu(effective_gates.new_tensor(target_sm_mean) - r_sm_mean)
        + torch.relu(r_post_mean - effective_gates.new_tensor(target_post_mean))
        + torch.relu(effective_gates.new_tensor(margin_mean) - (r_sm_mean - r_post_mean))
    )
    return rule_loss, metrics


def compute_all_gate_metrics(
    raw_gates: torch.Tensor,
    effective_gates: torch.Tensor,
    normalized_gates: torch.Tensor,
    sm_indices: list[int],
    post_indices: list[int],
    other_indices: list[int],
    target_sm_mean: float,
    target_post_mean: float,
    margin_mean: float,
) -> dict[str, float]:
    metrics = gate_group_metrics(effective_gates, sm_indices, post_indices, other_indices)
    metrics.update(
        {
            f"normalized_{key}": value
            for key, value in gate_group_metrics(normalized_gates, sm_indices, post_indices, other_indices).items()
        }
    )
    rule_loss, _ = rule_terms(
        effective_gates,
        sm_indices,
        post_indices,
        target_sm_mean,
        target_post_mean,
        margin_mean,
    )
    metrics["rule_loss"] = float(rule_loss.item())
    metrics["raw_gate_mean"] = float(raw_gates.mean().item())
    metrics["effective_gate_mean"] = float(effective_gates.mean().item())
    metrics["normalized_gate_mean"] = float(normalized_gates.mean().item())
    return metrics


def selection_value(metrics: dict[str, object], selection_metric: str) -> float:
    if selection_metric == "val_acc":
        return float(metrics["acc"])
    if metrics["kappa"] is None:
        return -math.inf
    return float(metrics["kappa"])


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    sm_indices: list[int],
    post_indices: list[int],
) -> dict[str, float]:
    model.train()
    total_examples = 0
    total_total_loss = 0.0
    total_cls_loss = 0.0
    total_rule_loss = 0.0
    preds: list[int] = []
    labels: list[int] = []
    non_blocking = device.type == "cuda"

    for x, y in loader:
        x = x.to(device, non_blocking=non_blocking)
        y = y.long().to(device, non_blocking=non_blocking)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        classification_loss = criterion(logits, y)

        if isinstance(model, EEGNetChannelGate):
            effective_gates = model.effective_gate_values()
            rule_loss, _ = rule_terms(
                effective_gates,
                sm_indices,
                post_indices,
                args.target_sm_mean,
                args.target_post_mean,
                args.margin_mean,
            )
        else:
            rule_loss = classification_loss.new_tensor(0.0)

        total_loss = classification_loss + args.lambda_rule * rule_loss
        total_loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        batch_preds = logits.argmax(dim=1)
        total_examples += batch_size
        total_total_loss += total_loss.item() * batch_size
        total_cls_loss += classification_loss.item() * batch_size
        total_rule_loss += rule_loss.item() * batch_size
        preds.extend(batch_preds.detach().cpu().tolist())
        labels.extend(y.detach().cpu().tolist())

    return {
        "total_loss": total_total_loss / max(total_examples, 1),
        "classification_loss": total_cls_loss / max(total_examples, 1),
        "rule_loss": total_rule_loss / max(total_examples, 1),
        "acc": float(np.mean(np.equal(preds, labels))) if total_examples else 0.0,
        "kappa": safe_kappa(labels, preds),
        "macro_f1": safe_macro_f1(labels, preds),
    }


def evaluate_subset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    subset: Subset,
    args: argparse.Namespace,
    sm_indices: list[int],
    post_indices: list[int],
) -> dict[str, object]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_examples = 0
    total_total_loss = 0.0
    total_cls_loss = 0.0
    total_rule_loss = 0.0
    preds: list[int] = []
    labels: list[int] = []
    non_blocking = device.type == "cuda"

    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=non_blocking)
            y = y.long().to(device, non_blocking=non_blocking)
            logits = model(x)
            classification_loss = criterion(logits, y)

            if isinstance(model, EEGNetChannelGate):
                effective_gates = model.effective_gate_values()
                rule_loss, _ = rule_terms(
                    effective_gates,
                    sm_indices,
                    post_indices,
                    args.target_sm_mean,
                    args.target_post_mean,
                    args.margin_mean,
                )
            else:
                rule_loss = classification_loss.new_tensor(0.0)

            total_loss = classification_loss + args.lambda_rule * rule_loss
            batch_size = y.size(0)
            batch_preds = logits.argmax(dim=1)
            total_examples += batch_size
            total_total_loss += total_loss.item() * batch_size
            total_cls_loss += classification_loss.item() * batch_size
            total_rule_loss += rule_loss.item() * batch_size
            preds.extend(batch_preds.detach().cpu().tolist())
            labels.extend(y.detach().cpu().tolist())

    return {
        "total_loss": total_total_loss / max(total_examples, 1),
        "classification_loss": total_cls_loss / max(total_examples, 1),
        "rule_loss": total_rule_loss / max(total_examples, 1),
        "acc": float(np.mean(np.equal(preds, labels))) if total_examples else 0.0,
        "kappa": safe_kappa(labels, preds),
        "macro_f1": safe_macro_f1(labels, preds),
        "num_samples": total_examples,
        "preds": preds,
        "labels": labels,
        "subset_size": len(subset),
    }


def init_history(history_path: Path) -> None:
    with history_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_total_loss",
                "train_classification_loss",
                "train_rule_loss",
                "train_acc",
                "train_kappa",
                "val_total_loss",
                "val_classification_loss",
                "val_rule_loss",
                "val_acc",
                "val_kappa",
                "R_SM_sum",
                "R_POST_sum",
                "R_OTHER_sum",
                "R_SM_mean",
                "R_POST_mean",
                "R_OTHER_mean",
                "R_SM_mean_minus_POST_mean",
                "lr",
            ],
        )
        writer.writeheader()


def append_history_row(row: dict[str, object], history_path: Path) -> None:
    with history_path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_total_loss",
                "train_classification_loss",
                "train_rule_loss",
                "train_acc",
                "train_kappa",
                "val_total_loss",
                "val_classification_loss",
                "val_rule_loss",
                "val_acc",
                "val_kappa",
                "R_SM_sum",
                "R_POST_sum",
                "R_OTHER_sum",
                "R_SM_mean",
                "R_POST_mean",
                "R_OTHER_mean",
                "R_SM_mean_minus_POST_mean",
                "lr",
            ],
        )
        writer.writerow(row)


def checkpoint_payload(
    args: argparse.Namespace,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau | None,
    epoch: int,
    best_value: float,
    best_epoch: int,
    subject_summary: dict[str, object],
    split_manifest_path: Path,
) -> dict[str, object]:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_selection_value": best_value,
        "best_epoch": best_epoch,
        "epoch": epoch,
        "args": vars(args),
        "subject_summary": subject_summary,
        "split_manifest_json": str(split_manifest_path),
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    return payload


def save_checkpoint(
    path: Path,
    args: argparse.Namespace,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau | None,
    epoch: int,
    best_value: float,
    best_epoch: int,
    subject_summary: dict[str, object],
    split_manifest_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        checkpoint_payload(
            args=args,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_value=best_value,
            best_epoch=best_epoch,
            subject_summary=subject_summary,
            split_manifest_path=split_manifest_path,
        ),
        path,
    )


def plot_history(history: list[dict[str, object]], output_path: Path, subject_id: str, model_type: str, lambda_rule: float) -> None:
    mpl_config_dir = output_path.parent.parent / ".matplotlib"
    mpl_config_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0, 0].plot(epochs, [float(row["train_total_loss"]) for row in history], label="train total")
    axes[0, 0].plot(epochs, [float(row["val_total_loss"]) for row in history], label="val total")
    axes[0, 0].plot(epochs, [float(row["train_classification_loss"]) for row in history], label="train cls")
    axes[0, 0].plot(epochs, [float(row["val_classification_loss"]) for row in history], label="val cls")
    axes[0, 0].set_title("Loss")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(epochs, [float(row["train_rule_loss"]) for row in history], label="train rule")
    axes[0, 1].plot(epochs, [float(row["val_rule_loss"]) for row in history], label="val rule")
    axes[0, 1].set_title("Rule loss")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(epochs, [float(row["train_acc"]) for row in history], label="train")
    axes[1, 0].plot(epochs, [float(row["val_acc"]) for row in history], label="validation")
    axes[1, 0].set_title("Accuracy")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(epochs, [float(row["R_SM_mean"]) for row in history], label="R_SM_mean")
    axes[1, 1].plot(epochs, [float(row["R_POST_mean"]) for row in history], label="R_POST_mean")
    axes[1, 1].plot(epochs, [float(row["R_OTHER_mean"]) for row in history], label="R_OTHER_mean")
    axes[1, 1].plot(epochs, [float(row["R_SM_mean_minus_POST_mean"]) for row in history], label="SM-POST")
    axes[1, 1].set_title("Gate means")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend(frameon=False)

    fig.suptitle(f"{subject_id} | {model_type} | lambda={lambda_rule}")
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

    rows = list(csv.DictReader(csv_path.open()))
    labels = [row["channel_name"] for row in rows]
    values = [float(row["effective_gate"]) for row in rows]
    colors = []
    for row in rows:
        if row["group"] == "sensorimotor":
            colors.append("#2a9d8f")
        elif row["group"] == "posterior":
            colors.append("#e76f51")
        else:
            colors.append("#8d99ae")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(len(labels)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("Effective gate")
    ax.set_title("Final channel gates")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def subject_run_dir(run_dir: Path, subject_id: str) -> Path:
    path = run_dir / "subjects" / str(subject_id)
    path.mkdir(parents=True, exist_ok=True)
    (path / "figures").mkdir(exist_ok=True)
    (path / "checkpoints").mkdir(exist_ok=True)
    return path


def write_subject_args(args: argparse.Namespace, path: Path, subject_id: str) -> Path:
    payload = dict(vars(args))
    payload["subject_id"] = subject_id
    payload["train_session"] = "T"
    payload["test_session"] = "E"
    payload["channel_groups"] = {
        "sensorimotor": SENSORIMOTOR_CHANNELS,
        "posterior": POSTERIOR_CHANNELS,
        "other": [channel for channel in BCI_IV_2A_CHANNELS if channel not in set(SENSORIMOTOR_CHANNELS) | set(POSTERIOR_CHANNELS)],
    }
    write_json(path, payload)
    return path


def write_gate_csvs(
    subject_dir: Path,
    channel_order: list[str],
    raw_gates: torch.Tensor,
    effective_gates: torch.Tensor,
    normalized_gates: torch.Tensor,
    sm_indices: list[int],
    post_indices: list[int],
) -> dict[str, Path]:
    sm_set = set(sm_indices)
    post_set = set(post_indices)
    rows: list[dict[str, object]] = []
    for idx, channel_name in enumerate(channel_order):
        if idx in sm_set:
            group = "sensorimotor"
        elif idx in post_set:
            group = "posterior"
        else:
            group = "other"
        rows.append(
            {
                "channel_index": idx,
                "channel_name": channel_name,
                "group": group,
                "raw_gate": float(raw_gates[idx].item()),
                "effective_gate": float(effective_gates[idx].item()),
                "normalized_gate": float(normalized_gates[idx].item()),
            }
        )

    outputs = {
        "channel_gates_csv": subject_dir / "channel_gates.csv",
        "raw_channel_gates_csv": subject_dir / "raw_channel_gates.csv",
        "effective_channel_gates_csv": subject_dir / "effective_channel_gates.csv",
        "normalized_channel_gates_csv": subject_dir / "normalized_channel_gates.csv",
    }

    with outputs["channel_gates_csv"].open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["channel_index", "channel_name", "group", "raw_gate", "effective_gate", "normalized_gate"],
        )
        writer.writeheader()
        writer.writerows(rows)

    for key, value_field in [
        ("raw_channel_gates_csv", "raw_gate"),
        ("effective_channel_gates_csv", "effective_gate"),
        ("normalized_channel_gates_csv", "normalized_gate"),
    ]:
        with outputs[key].open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["channel_index", "channel_name", "group", value_field])
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "channel_index": row["channel_index"],
                        "channel_name": row["channel_name"],
                        "group": row["group"],
                        value_field: row[value_field],
                    }
                )
    return outputs


def train_subject(
    dataset,
    args: argparse.Namespace,
    run_dir: Path,
    subject_id: str,
    device: torch.device,
    channel_order: list[str],
    sm_indices: list[int],
    post_indices: list[int],
    other_indices: list[int],
) -> dict[str, object]:
    subject_dir = subject_run_dir(run_dir, subject_id)
    history_path = subject_dir / "history.csv"
    checkpoint_best_path = subject_dir / "checkpoint_best.pt"
    checkpoint_last_path = subject_dir / "checkpoint_last.pt"
    figure_path = subject_dir / "figures" / "training_curves.png"
    gates_figure_path = subject_dir / "figures" / "channel_gates.png"
    summary_path = subject_dir / "summary.json"
    args_path = write_subject_args(args, subject_dir / "args.json", subject_id)
    init_history(history_path)

    split_artifacts = split_subject_session_indices(
        dataset=dataset,
        subject_id=subject_id,
        train_session="T",
        test_session="E",
        val_ratio=args.val_ratio,
        split_seed=resolve_split_seed(args),
        val_mode=args.val_mode,
        run_holdout_index=args.run_holdout_index,
    )
    split_files = save_subject_session_split(output_dir=subject_dir, split_artifacts=split_artifacts, val_ratio=args.val_ratio)
    split_manifest_path = split_files["split_manifest_json"]

    train_set = split_artifacts["train_subset"]
    val_set = split_artifacts["validation_subset"]
    test_set = split_artifacts["test_subset"]
    train_loader = make_loader(train_set, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_set, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_set, args.batch_size, False, args.num_workers)

    model = build_model(args.model_type).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
        )

    history: list[dict[str, object]] = []
    best_epoch = 0
    best_selection_score = -math.inf
    best_val_snapshot: dict[str, object] | None = None
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, args, sm_indices, post_indices)
        val_metrics = evaluate_subset(model, val_loader, device, val_set, args, sm_indices, post_indices)
        raw_gates, effective_gates, normalized_gates = final_gate_state(model)
        gate_metrics = compute_all_gate_metrics(
            raw_gates,
            effective_gates,
            normalized_gates,
            sm_indices,
            post_indices,
            other_indices,
            args.target_sm_mean,
            args.target_post_mean,
            args.margin_mean,
        )

        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_total_loss": train_metrics["total_loss"],
            "train_classification_loss": train_metrics["classification_loss"],
            "train_rule_loss": train_metrics["rule_loss"],
            "train_acc": train_metrics["acc"],
            "train_kappa": train_metrics["kappa"],
            "val_total_loss": val_metrics["total_loss"],
            "val_classification_loss": val_metrics["classification_loss"],
            "val_rule_loss": val_metrics["rule_loss"],
            "val_acc": val_metrics["acc"],
            "val_kappa": val_metrics["kappa"],
            "R_SM_sum": gate_metrics["R_SM_sum"],
            "R_POST_sum": gate_metrics["R_POST_sum"],
            "R_OTHER_sum": gate_metrics["R_OTHER_sum"],
            "R_SM_mean": gate_metrics["R_SM_mean"],
            "R_POST_mean": gate_metrics["R_POST_mean"],
            "R_OTHER_mean": gate_metrics["R_OTHER_mean"],
            "R_SM_mean_minus_POST_mean": gate_metrics["R_SM_mean_minus_POST_mean"],
            "lr": lr,
        }
        history.append(row)
        append_history_row(row, history_path)

        current_score = selection_value(val_metrics, args.selection_metric)
        if current_score > best_selection_score:
            best_selection_score = current_score
            best_epoch = epoch
            best_val_snapshot = {
                "best_val_acc": float(val_metrics["acc"]),
                "best_val_kappa": None if val_metrics["kappa"] is None else float(val_metrics["kappa"]),
                "best_val_macro_f1": None if val_metrics["macro_f1"] is None else float(val_metrics["macro_f1"]),
            }
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_best_path,
                args,
                model,
                optimizer,
                scheduler,
                epoch,
                best_selection_score,
                best_epoch,
                {"subject_id": subject_id},
                split_manifest_path,
            )
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            scheduler.step(float(val_metrics["acc"]))

        save_checkpoint(
            checkpoint_last_path,
            args,
            model,
            optimizer,
            scheduler,
            epoch,
            best_selection_score,
            best_epoch,
            {"subject_id": subject_id},
            split_manifest_path,
        )
        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            save_checkpoint(
                subject_dir / "checkpoints" / f"checkpoint_epoch_{epoch:03d}.pt",
                args,
                model,
                optimizer,
                scheduler,
                epoch,
                best_selection_score,
                best_epoch,
                {"subject_id": subject_id},
                split_manifest_path,
            )

        print(
            f"subject={subject_id} epoch={epoch:03d} train_acc={train_metrics['acc']:.4f} "
            f"val_acc={float(val_metrics['acc']):.4f} val_rule={float(val_metrics['rule_loss']):.4f} "
            f"sm-post={gate_metrics['R_SM_mean_minus_POST_mean']:.4f} lr={lr:.6g}"
        )

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            stopped_early = True
            break

    plot_history(history, figure_path, subject_id, args.model_type, args.lambda_rule)

    best_checkpoint = torch.load(checkpoint_best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    best_train_metrics = evaluate_subset(model, train_loader, device, train_set, args, sm_indices, post_indices)
    best_val_metrics = evaluate_subset(model, val_loader, device, val_set, args, sm_indices, post_indices)
    test_metrics = evaluate_subset(model, test_loader, device, test_set, args, sm_indices, post_indices)
    raw_gates, effective_gates, normalized_gates = final_gate_state(model)
    gate_metrics = compute_all_gate_metrics(
        raw_gates,
        effective_gates,
        normalized_gates,
        sm_indices,
        post_indices,
        other_indices,
        args.target_sm_mean,
        args.target_post_mean,
        args.margin_mean,
    )
    gate_paths = write_gate_csvs(
        subject_dir=subject_dir,
        channel_order=channel_order,
        raw_gates=raw_gates,
        effective_gates=effective_gates,
        normalized_gates=normalized_gates,
        sm_indices=sm_indices,
        post_indices=post_indices,
    )
    plot_channel_gates(gate_paths["channel_gates_csv"], gates_figure_path)

    final_row = history[-1]
    summary = {
        "model": "EEGNet",
        "protocol": "subject_specific_cross_session",
        "subject_id": subject_id,
        "train_session": "T",
        "test_session": "E",
        "config_name": args.config_name,
        "phase_tag": args.phase_tag,
        "model_type": args.model_type,
        "seed": args.seed,
        "split_seed": resolve_split_seed(args),
        "selection_metric": args.selection_metric,
        "val_mode": args.val_mode,
        "run_holdout_index": args.run_holdout_index,
        "device": str(device),
        "optimizer": "Adam",
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "plateau_factor": args.plateau_factor,
        "plateau_patience": args.plateau_patience,
        "early_stopping_patience": args.early_stopping_patience,
        "checkpoint_every": args.checkpoint_every,
        "val_ratio": args.val_ratio,
        "lambda_rule": args.lambda_rule,
        "target_sm_mean": args.target_sm_mean,
        "target_post_mean": args.target_post_mean,
        "margin_mean": args.margin_mean,
        "best_epoch": best_epoch,
        "best_val_acc": None if best_val_snapshot is None else best_val_snapshot["best_val_acc"],
        "best_val_kappa": None if best_val_snapshot is None else best_val_snapshot["best_val_kappa"],
        "best_val_macro_f1": None if best_val_snapshot is None else best_val_snapshot["best_val_macro_f1"],
        "final_train_acc": float(final_row["train_acc"]),
        "final_train_kappa": final_row["train_kappa"],
        "final_val_acc": float(final_row["val_acc"]),
        "final_val_kappa": final_row["val_kappa"],
        "best_checkpoint_train_acc": float(best_train_metrics["acc"]),
        "best_checkpoint_val_acc": float(best_val_metrics["acc"]),
        "test_acc": float(test_metrics["acc"]),
        "test_loss": float(test_metrics["total_loss"]),
        "test_kappa": test_metrics["kappa"],
        "macro_f1": test_metrics["macro_f1"],
        "train_size": len(train_set),
        "validation_size": len(val_set),
        "test_size": len(test_set),
        "history_csv": str(history_path),
        "args_json": str(args_path),
        "summary_json": str(summary_path),
        "checkpoint_best_path": str(checkpoint_best_path),
        "checkpoint_last_path": str(checkpoint_last_path),
        "figures_path": str(figure_path),
        "split_manifest_json": str(split_manifest_path),
        "stopped_early": stopped_early,
        "status": "completed",
    }
    summary.update(gate_metrics)
    summary.update({name: str(path) for name, path in gate_paths.items()})
    write_json(summary_path, summary)
    return summary


def aggregate_subject_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    def collect(name: str) -> list[float]:
        values = []
        for row in rows:
            value = row.get(name)
            if value is not None:
                values.append(float(value))
        return values

    def mean_or_none(name: str) -> float | None:
        values = collect(name)
        return None if not values else float(np.mean(values))

    def std_or_none(name: str) -> float | None:
        values = collect(name)
        return None if not values else float(np.std(values))

    return {
        "num_subjects": len(rows),
        "mean_best_val_acc": mean_or_none("best_val_acc"),
        "std_best_val_acc": std_or_none("best_val_acc"),
        "mean_best_val_kappa": mean_or_none("best_val_kappa"),
        "mean_final_val_acc": mean_or_none("final_val_acc"),
        "mean_test_acc": mean_or_none("test_acc"),
        "std_test_acc": std_or_none("test_acc"),
        "mean_test_kappa": mean_or_none("test_kappa"),
        "mean_macro_f1": mean_or_none("macro_f1"),
        "mean_R_SM_mean": mean_or_none("R_SM_mean"),
        "mean_R_POST_mean": mean_or_none("R_POST_mean"),
        "mean_R_OTHER_mean": mean_or_none("R_OTHER_mean"),
        "mean_R_SM_mean_minus_POST_mean": mean_or_none("R_SM_mean_minus_POST_mean"),
    }


def write_subject_metrics_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "subject_id",
        "train_session",
        "test_session",
        "config_name",
        "model_type",
        "lambda_rule",
        "seed",
        "split_seed",
        "val_mode",
        "selection_metric",
        "train_size",
        "validation_size",
        "test_size",
        "best_val_acc",
        "best_val_kappa",
        "best_epoch",
        "final_train_acc",
        "final_val_acc",
        "test_acc",
        "test_loss",
        "test_kappa",
        "macro_f1",
        "R_SM_sum",
        "R_POST_sum",
        "R_OTHER_sum",
        "R_SM_mean",
        "R_POST_mean",
        "R_OTHER_mean",
        "R_SM_mean_minus_POST_mean",
        "raw_channel_gates_csv",
        "effective_channel_gates_csv",
        "normalized_channel_gates_csv",
        "channel_gates_csv",
        "summary_json",
        "checkpoint_best_path",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not args.data_root.exists():
        raise FileNotFoundError(f"Dataset not found at {args.data_root.resolve()}")

    channel_order = load_channel_order(args.channel_order_file)
    sm_indices = channel_indices(channel_order, SENSORIMOTOR_CHANNELS, "sensorimotor")
    post_indices = channel_indices(channel_order, POSTERIOR_CHANNELS, "posterior")
    other_indices = [idx for idx in range(len(channel_order)) if idx not in set(sm_indices) | set(post_indices)]

    seed_everything(args.seed)
    device = get_device(args.device)
    run_dir = make_run_dir(args)
    args_path = run_dir / "args.json"
    write_json(args_path, vars(args))

    dataset = build_dataset(args.data_root, verbose=True)
    sample_x, _ = dataset[0]
    if tuple(sample_x.shape) != (1, NUM_ELECTRODES, CHUNK_SIZE):
        raise ValueError(f"Unexpected dataset sample shape: {tuple(sample_x.shape)}")

    available_subjects = subject_ids(dataset)
    selected_subjects = args.subject_ids if args.subject_ids else available_subjects
    unknown_subjects = sorted(set(selected_subjects).difference(available_subjects))
    if unknown_subjects:
        raise ValueError(f"Unknown subject ids: {unknown_subjects}")

    subject_rows: list[dict[str, object]] = []
    for subject_id in selected_subjects:
        print(
            f"=== Subject {subject_id} | model={args.model_type} "
            f"lambda={args.lambda_rule} seed={args.seed} ==="
        )
        subject_rows.append(
            train_subject(
                dataset=dataset,
                args=args,
                run_dir=run_dir,
                subject_id=subject_id,
                device=device,
                channel_order=channel_order,
                sm_indices=sm_indices,
                post_indices=post_indices,
                other_indices=other_indices,
            )
        )

    subject_metrics_path = run_dir / "subject_metrics.csv"
    write_subject_metrics_csv(subject_metrics_path, subject_rows)
    aggregate = aggregate_subject_metrics(subject_rows)

    summary = {
        "model": "EEGNet",
        "protocol": "subject_specific_cross_session",
        "block": "eegnet_logic_subject_session",
        "config_name": args.config_name,
        "phase_tag": args.phase_tag,
        "model_type": args.model_type,
        "seed": args.seed,
        "split_seed": resolve_split_seed(args),
        "device": str(device),
        "selection_metric": args.selection_metric,
        "val_mode": args.val_mode,
        "run_holdout_index": args.run_holdout_index,
        "train_session": "T",
        "test_session": "E",
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "plateau_factor": args.plateau_factor,
        "plateau_patience": args.plateau_patience,
        "early_stopping_patience": args.early_stopping_patience,
        "checkpoint_every": args.checkpoint_every,
        "val_ratio": args.val_ratio,
        "lambda_rule": args.lambda_rule,
        "target_sm_mean": args.target_sm_mean,
        "target_post_mean": args.target_post_mean,
        "margin_mean": args.margin_mean,
        "channel_order_file": str(args.channel_order_file),
        "subject_ids": selected_subjects,
        "subject_metrics_csv": str(subject_metrics_path),
        "args_json": str(args_path),
        "summary_json": str(run_dir / "summary.json"),
        "run_dir": str(run_dir),
        "status": "completed",
        "uses_channel_gates": args.model_type != "eegnet_baseline",
    }
    summary.update(aggregate)
    write_json(run_dir / "summary.json", summary)
    print(f"run_summary_json={run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
