#!/usr/bin/env python
"""Train EEGNet baseline with subject-specific cross-session protocol."""

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
    DEFAULT_DATA_ROOT,
    RUNS_DIR,
    build_dataset,
    build_model,
    env_flag,
    get_device,
    seed_everything,
    str2bool,
)
from subject_session_utils import save_subject_session_split, split_subject_session_indices, subject_ids, write_json


RUN_ROOT = RUNS_DIR / "eegnet_baseline_subject_session"
DEFAULT_RUN_ROOT = RUN_ROOT / "config_runs"
DEFAULT_SELECTION_METRIC = "val_acc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--config-name", default="unnamed_config")
    parser.add_argument("--phase-tag", default="config_sweep")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=9e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
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
    parser.add_argument(
        "--skip-trial-with-artifacts",
        type=str2bool,
        default=env_flag("SKIP_ARTIFACTS", False),
        help="Exclude trials flagged as artifacts in the BCICIV 2a metadata.",
    )
    return parser.parse_args()


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


def make_run_dir(args: argparse.Namespace) -> Path:
    args.run_root.mkdir(parents=True, exist_ok=True)
    if args.run_id:
        base_run_id = args.run_id
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_run_id = f"{args.config_name}_seed{args.seed}_split{resolve_split_seed(args)}_{timestamp}"
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


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    preds: list[int] = []
    labels: list[int] = []
    non_blocking = device.type == "cuda"

    for x, y in loader:
        x = x.to(device, non_blocking=non_blocking)
        y = y.long().to(device, non_blocking=non_blocking)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        batch_preds = logits.argmax(dim=1)
        total_loss += loss.item() * batch_size
        total_correct += (batch_preds == y).sum().item()
        total_examples += batch_size
        preds.extend(batch_preds.detach().cpu().tolist())
        labels.extend(y.detach().cpu().tolist())

    return {
        "loss": total_loss / max(total_examples, 1),
        "acc": total_correct / max(total_examples, 1),
        "kappa": safe_kappa(labels, preds),
        "macro_f1": safe_macro_f1(labels, preds),
    }


def evaluate_subset(model: nn.Module, loader: DataLoader, device: torch.device, subset: Subset) -> dict[str, object]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    preds: list[int] = []
    labels: list[int] = []
    non_blocking = device.type == "cuda"

    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=non_blocking)
            y = y.long().to(device, non_blocking=non_blocking)
            logits = model(x)
            loss = criterion(logits, y)

            batch_size = y.size(0)
            batch_preds = logits.argmax(dim=1)
            total_loss += loss.item() * batch_size
            total_correct += (batch_preds == y).sum().item()
            total_examples += batch_size
            preds.extend(batch_preds.detach().cpu().tolist())
            labels.extend(y.detach().cpu().tolist())

    return {
        "loss": total_loss / max(total_examples, 1),
        "acc": total_correct / max(total_examples, 1),
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
                "train_loss",
                "train_acc",
                "train_kappa",
                "train_macro_f1",
                "val_loss",
                "val_acc",
                "val_kappa",
                "val_macro_f1",
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
                "train_loss",
                "train_acc",
                "train_kappa",
                "train_macro_f1",
                "val_loss",
                "val_acc",
                "val_kappa",
                "val_macro_f1",
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


def plot_history(history: list[dict[str, object]], output_path: Path, subject_id: str, config_name: str) -> None:
    mpl_config_dir = output_path.parent.parent / ".matplotlib"
    mpl_config_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    axes[0].plot(epochs, [float(row["train_loss"]) for row in history], label="train")
    axes[0].plot(epochs, [float(row["val_loss"]) for row in history], label="validation")
    axes[0].set_title("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, [float(row["train_acc"]) for row in history], label="train")
    axes[1].plot(epochs, [float(row["val_acc"]) for row in history], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend(frameon=False)

    axes[2].plot(epochs, [0.0 if row["train_kappa"] is None else float(row["train_kappa"]) for row in history], label="train")
    axes[2].plot(epochs, [0.0 if row["val_kappa"] is None else float(row["val_kappa"]) for row in history], label="validation")
    axes[2].set_title("Kappa")
    axes[2].grid(alpha=0.3)
    axes[2].legend(frameon=False)

    fig.suptitle(f"{config_name} - {subject_id}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def selection_value(metrics: dict[str, object], selection_metric: str) -> float:
    if selection_metric == "val_acc":
        return float(metrics["acc"])
    if metrics["kappa"] is None:
        return -math.inf
    return float(metrics["kappa"])


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
    payload["skip_trial_with_artifacts"] = bool(args.skip_trial_with_artifacts)
    write_json(path, payload)
    return path


def train_subject(
    dataset,
    args: argparse.Namespace,
    run_dir: Path,
    subject_id: str,
    device: torch.device,
) -> dict[str, object]:
    subject_dir = subject_run_dir(run_dir, subject_id)
    history_path = subject_dir / "history.csv"
    checkpoint_best_path = subject_dir / "checkpoint_best.pt"
    checkpoint_last_path = subject_dir / "checkpoint_last.pt"
    figure_path = subject_dir / "figures" / "training_curves.png"
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

    split_files = save_subject_session_split(
        output_dir=subject_dir,
        split_artifacts=split_artifacts,
        val_ratio=args.val_ratio,
    )
    split_manifest_path = split_files["split_manifest_json"]

    train_set = split_artifacts["train_subset"]
    val_set = split_artifacts["validation_subset"]
    test_set = split_artifacts["test_subset"]
    train_loader = make_loader(train_set, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_set, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_set, args.batch_size, False, args.num_workers)

    model = build_model().to(device)
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
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_subset(model, val_loader, device, val_set)
        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_kappa": train_metrics["kappa"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_kappa": val_metrics["kappa"],
            "val_macro_f1": val_metrics["macro_f1"],
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

        val_kappa_text = "n/a" if val_metrics["kappa"] is None else f"{float(val_metrics['kappa']):.4f}"
        print(
            f"subject={subject_id} epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['acc']:.4f} val_loss={float(val_metrics['loss']):.4f} "
            f"val_acc={float(val_metrics['acc']):.4f} val_kappa={val_kappa_text} lr={lr:.6g}"
        )

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            stopped_early = True
            break

    plot_history(history, figure_path, subject_id, args.config_name)

    best_checkpoint = torch.load(checkpoint_best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    best_train_metrics = evaluate_subset(model, train_loader, device, train_set)
    best_val_metrics = evaluate_subset(model, val_loader, device, val_set)
    test_metrics = evaluate_subset(model, test_loader, device, test_set)

    final_row = history[-1]
    summary = {
        "model": "EEGNet",
        "protocol": "subject_specific_cross_session",
        "subject_id": subject_id,
        "train_session": "T",
        "test_session": "E",
        "config_name": args.config_name,
        "phase_tag": args.phase_tag,
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
        "skip_trial_with_artifacts": bool(args.skip_trial_with_artifacts),
        "best_epoch": best_epoch,
        "best_val_acc": None if best_val_snapshot is None else best_val_snapshot["best_val_acc"],
        "best_val_kappa": None if best_val_snapshot is None else best_val_snapshot["best_val_kappa"],
        "best_val_macro_f1": None if best_val_snapshot is None else best_val_snapshot["best_val_macro_f1"],
        "final_train_acc": float(final_row["train_acc"]),
        "final_train_kappa": final_row["train_kappa"],
        "final_train_macro_f1": final_row["train_macro_f1"],
        "final_val_acc": float(final_row["val_acc"]),
        "final_val_kappa": final_row["val_kappa"],
        "final_val_macro_f1": final_row["val_macro_f1"],
        "best_checkpoint_train_acc": float(best_train_metrics["acc"]),
        "best_checkpoint_val_acc": float(best_val_metrics["acc"]),
        "test_acc": float(test_metrics["acc"]),
        "test_loss": float(test_metrics["loss"]),
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
    }


def write_subject_metrics_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "subject_id",
        "train_session",
        "test_session",
        "config_name",
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
    if not args.data_root.exists():
        raise FileNotFoundError(f"Dataset not found at {args.data_root.resolve()}")

    seed_everything(args.seed)
    device = get_device(args.device)
    run_dir = make_run_dir(args)
    args_path = run_dir / "args.json"
    write_json(args_path, vars(args))

    dataset = build_dataset(
        args.data_root,
        verbose=True,
        skip_trial_with_artifacts=args.skip_trial_with_artifacts,
    )
    available_subjects = subject_ids(dataset)
    selected_subjects = args.subject_ids if args.subject_ids else available_subjects
    unknown_subjects = sorted(set(selected_subjects).difference(available_subjects))
    if unknown_subjects:
        raise ValueError(f"Unknown subject ids: {unknown_subjects}")

    subject_rows: list[dict[str, object]] = []
    for subject_id in selected_subjects:
        print(f"=== Subject {subject_id} | config={args.config_name} seed={args.seed} ===")
        subject_rows.append(train_subject(dataset, args, run_dir, subject_id, device))

    subject_metrics_path = run_dir / "subject_metrics.csv"
    write_subject_metrics_csv(subject_metrics_path, subject_rows)
    aggregate = aggregate_subject_metrics(subject_rows)

    summary = {
        "model": "EEGNet",
        "protocol": "subject_specific_cross_session",
        "block": "eegnet_baseline_subject_session",
        "config_name": args.config_name,
        "phase_tag": args.phase_tag,
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
        "skip_trial_with_artifacts": bool(args.skip_trial_with_artifacts),
        "subject_ids": selected_subjects,
        "subject_metrics_csv": str(subject_metrics_path),
        "args_json": str(args_path),
        "summary_json": str(run_dir / "summary.json"),
        "run_dir": str(run_dir),
        "status": "completed",
    }
    summary.update(aggregate)
    write_json(run_dir / "summary.json", summary)
    print(f"run_summary_json={run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
