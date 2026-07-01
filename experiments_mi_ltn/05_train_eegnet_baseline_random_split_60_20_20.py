#!/usr/bin/env python
"""Train EEGNet baseline with a reproducible random train/validation/test split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Subset

from mi_ltn_common import DEFAULT_DATA_ROOT, RUNS_DIR, build_dataset, build_model, get_device, seed_everything


RUN_ROOT = RUNS_DIR / "eegnet_baseline_random_split_60_20_20"
DEFAULT_RUN_ROOT = RUN_ROOT / "config_runs"
DEFAULT_EXPERIMENT_ROOT = RUN_ROOT
CONFIG_PRESETS: dict[str, dict[str, object]] = {
    "baseline_short": {
        "epochs": 100,
        "learning_rate": 5e-4,
        "batch_size": 64,
        "weight_decay": 0.0,
        "scheduler": "none",
        "plateau_factor": 0.5,
        "plateau_patience": 10,
        "early_stopping_patience": 50,
    },
    "long_training_300": {
        "epochs": 300,
        "learning_rate": 9e-4,
        "batch_size": 64,
        "weight_decay": 0.0,
        "scheduler": "none",
        "plateau_factor": 0.5,
        "plateau_patience": 10,
        "early_stopping_patience": 50,
    },
    "long_training_500": {
        "epochs": 500,
        "learning_rate": 9e-4,
        "batch_size": 64,
        "weight_decay": 0.0,
        "scheduler": "none",
        "plateau_factor": 0.5,
        "plateau_patience": 10,
        "early_stopping_patience": 75,
    },
    "lr_low": {
        "epochs": 300,
        "learning_rate": 5e-4,
        "batch_size": 64,
        "weight_decay": 0.0,
        "scheduler": "none",
        "plateau_factor": 0.5,
        "plateau_patience": 10,
        "early_stopping_patience": 50,
    },
    "lr_high": {
        "epochs": 300,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "weight_decay": 0.0,
        "scheduler": "none",
        "plateau_factor": 0.5,
        "plateau_patience": 10,
        "early_stopping_patience": 50,
    },
    "weight_decay": {
        "epochs": 300,
        "learning_rate": 9e-4,
        "batch_size": 64,
        "weight_decay": 1e-4,
        "scheduler": "none",
        "plateau_factor": 0.5,
        "plateau_patience": 10,
        "early_stopping_patience": 50,
    },
    "scheduler_plateau": {
        "epochs": 300,
        "learning_rate": 9e-4,
        "batch_size": 64,
        "weight_decay": 0.0,
        "scheduler": "plateau",
        "plateau_factor": 0.5,
        "plateau_patience": 10,
        "early_stopping_patience": 50,
    },
}
SUMMARY_FIELDS = [
    "model",
    "block",
    "protocol",
    "config_name",
    "phase_tag",
    "seed",
    "split_seed",
    "device",
    "optimizer",
    "epochs_requested",
    "epochs_completed",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "scheduler",
    "plateau_factor",
    "plateau_patience",
    "early_stopping_patience",
    "checkpoint_every",
    "train_ratio",
    "val_ratio",
    "test_ratio",
    "best_val_acc",
    "best_val_kappa",
    "best_epoch",
    "final_train_acc",
    "final_train_kappa",
    "final_train_macro_f1",
    "final_val_acc",
    "final_val_kappa",
    "final_val_macro_f1",
    "test_acc",
    "test_kappa",
    "test_macro_f1",
    "train_loss",
    "val_loss",
    "test_loss",
    "stopped_early",
    "mixes_sessions",
    "shared_split_manifest_json",
    "shared_split_summary_csv",
    "run_split_manifest_json",
    "run_split_summary_csv",
    "train_indices_csv",
    "validation_indices_csv",
    "test_indices_csv",
    "subject_test_metrics_csv",
    "test_predictions_csv",
    "test_confusion_matrix_csv",
    "history_csv",
    "summary_json",
    "args_json",
    "figures_path",
    "run_dir",
    "status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--config-name", default="unnamed_config")
    parser.add_argument("--phase-tag", default="config_sweep")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=("none", "plateau"), default="none")
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def resolve_split_seed(args: argparse.Namespace) -> int:
    return args.split_seed if args.split_seed is not None else args.seed


def resolve_train_ratio(args: argparse.Namespace) -> float:
    return args.train_ratio


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    for name, value in [("train", train_ratio), ("validation", val_ratio), ("test", test_ratio)]:
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} ratio must be in (0, 1), got {value}")


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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_dir(run_root: Path, run_id: str, overwrite: bool) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    run_dir = run_root / run_id
    if run_dir.exists() and not overwrite:
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    if run_dir.exists() and overwrite:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = run_root / f"{run_id}_previous_{timestamp}"
        run_dir.rename(backup_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir()
    (run_dir / "checkpoints").mkdir()
    return run_dir


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
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


def write_args(args: argparse.Namespace, run_dir: Path) -> Path:
    path = run_dir / "args.json"
    with path.open("w") as handle:
        json.dump(vars(args), handle, indent=2, default=json_default)
    return path


def init_history(history_path: Path) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "val_kappa",
        "val_macro_f1",
        "lr",
    ]
    with history_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def append_history_row(row: dict[str, object], history_path: Path) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "val_kappa",
        "val_macro_f1",
        "lr",
    ]
    with history_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerow(row)


def limit_dataset_indices(dataset_size: int, limit_samples: int | None) -> list[int]:
    if limit_samples is None:
        return list(range(dataset_size))
    return list(range(min(limit_samples, dataset_size)))


def per_group_split_counts(group_size: int, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if group_size < 3:
        raise ValueError(f"Expected at least 3 samples per (subject, class) group, got {group_size}")
    val_count = int(round(group_size * val_ratio))
    test_count = int(round(group_size * test_ratio))
    val_count = max(1, val_count)
    test_count = max(1, test_count)
    while val_count + test_count >= group_size:
        if val_count >= test_count and val_count > 1:
            val_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            break
    train_count = group_size - val_count - test_count
    if train_count < 1:
        raise ValueError(
            f"Could not allocate a non-empty train split for group_size={group_size}, "
            f"val_count={val_count}, test_count={test_count}"
        )
    return train_count, val_count, test_count


def split_session_counts(info) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    if "session" not in info.columns:
        return counts
    grouped = info.groupby(["split", "session"]).size()
    for (split_name, session_name), value in grouped.items():
        counts.setdefault(str(split_name), {})[str(session_name)] = int(value)
    return counts


def write_indices_csv(path: Path, indices: list[int]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset_index"])
        writer.writeheader()
        for index in indices:
            writer.writerow({"dataset_index": index})


def load_indices_csv(path: Path) -> list[int]:
    with path.open() as handle:
        return [int(row["dataset_index"]) for row in csv.DictReader(handle)]


def materialize_shared_split(
    dataset: Dataset,
    experiment_root: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    split_seed: int,
    limit_samples: int | None,
) -> dict[str, object]:
    validate_ratios(train_ratio, val_ratio, test_ratio)
    shared_root = experiment_root / "shared_trial_splits"
    limit_tag = f"limit{limit_samples}" if limit_samples is not None else "limitall"
    split_name = f"subject_stratified_seed{split_seed}_val{val_ratio:0.4f}".replace(".", "p")
    split_name += f"_test{test_ratio:0.4f}".replace(".", "p")
    split_name += f"_{limit_tag}"
    split_dir = shared_root / split_name
    manifest_path = split_dir / "split_manifest.json"
    summary_path = split_dir / "split_summary.csv"
    train_csv = split_dir / "train_indices.csv"
    val_csv = split_dir / "validation_indices.csv"
    test_csv = split_dir / "test_indices.csv"

    if manifest_path.exists() and summary_path.exists() and train_csv.exists() and val_csv.exists() and test_csv.exists():
        train_indices = load_indices_csv(train_csv)
        val_indices = load_indices_csv(val_csv)
        test_indices = load_indices_csv(test_csv)
        manifest = json.load(manifest_path.open())
    else:
        split_dir.mkdir(parents=True, exist_ok=True)
        info = dataset.info.reset_index(drop=True).copy()
        selected_indices = limit_dataset_indices(len(info), limit_samples)
        selected_info = info.iloc[selected_indices].copy()
        selected_info["dataset_index"] = selected_indices
        rng = np.random.default_rng(split_seed)

        train_indices: list[int] = []
        val_indices: list[int] = []
        test_indices: list[int] = []
        summary_rows: list[dict[str, object]] = []

        grouped = selected_info.groupby(["subject_id", "label"], sort=True)
        for (subject_id, class_label), group in grouped:
            group_indices = group["dataset_index"].tolist()
            shuffled = rng.permutation(group_indices).tolist()
            train_count, val_count, test_count = per_group_split_counts(len(shuffled), val_ratio, test_ratio)
            train_group = shuffled[:train_count]
            val_group = shuffled[train_count : train_count + val_count]
            test_group = shuffled[train_count + val_count : train_count + val_count + test_count]
            train_indices.extend(train_group)
            val_indices.extend(val_group)
            test_indices.extend(test_group)
            summary_rows.extend(
                [
                    {
                        "subject_id": str(subject_id),
                        "class_label": int(class_label),
                        "split": "train",
                        "num_trials": len(train_group),
                    },
                    {
                        "subject_id": str(subject_id),
                        "class_label": int(class_label),
                        "split": "validation",
                        "num_trials": len(val_group),
                    },
                    {
                        "subject_id": str(subject_id),
                        "class_label": int(class_label),
                        "split": "test",
                        "num_trials": len(test_group),
                    },
                ]
            )

        train_indices.sort()
        val_indices.sort()
        test_indices.sort()

        write_indices_csv(train_csv, train_indices)
        write_indices_csv(val_csv, val_indices)
        write_indices_csv(test_csv, test_indices)
        with summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["subject_id", "class_label", "split", "num_trials"])
            writer.writeheader()
            writer.writerows(summary_rows)

        split_assignments = []
        for split_name_local, split_indices in [
            ("train", train_indices),
            ("validation", val_indices),
            ("test", test_indices),
        ]:
            subset_info = info.iloc[split_indices].copy()
            subset_info["split"] = split_name_local
            split_assignments.append(subset_info)
        split_info = pd.concat(split_assignments, ignore_index=True)
        session_counts = split_session_counts(split_info)
        mixes_sessions = any(len(counts) > 1 for counts in session_counts.values())

        manifest = {
            "protocol": "random_train_validation_test_60_20_20_preliminary",
            "split_strategy": "subject_aware_class_stratified_within_subject",
            "split_seed": split_seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "limit_samples": limit_samples,
            "dataset_size": len(info),
            "num_selected_samples": len(selected_indices),
            "train_size": len(train_indices),
            "validation_size": len(val_indices),
            "test_size": len(test_indices),
            "mixes_sessions": mixes_sessions,
            "session_counts_by_split": session_counts,
            "train_indices_csv": str(train_csv.resolve()),
            "validation_indices_csv": str(val_csv.resolve()),
            "test_indices_csv": str(test_csv.resolve()),
            "split_summary_csv": str(summary_path.resolve()),
            "created_at": utc_now_iso(),
        }
        with manifest_path.open("w") as handle:
            json.dump(manifest, handle, indent=2, default=json_default)

    split_indices_root = experiment_root / "split_indices"
    split_indices_root.mkdir(parents=True, exist_ok=True)
    split_seed_json = split_indices_root / f"split_seed_{split_seed}.json"
    split_seed_payload = {
        "protocol": "random_train_validation_test_60_20_20_preliminary",
        "mixes_sessions": manifest.get("mixes_sessions"),
        "split_seed": split_seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "limit_samples": limit_samples,
        "shared_split_manifest_json": str(manifest_path.resolve()),
        "shared_split_summary_csv": str(summary_path.resolve()),
        "train_indices_csv": str(train_csv.resolve()),
        "validation_indices_csv": str(val_csv.resolve()),
        "test_indices_csv": str(test_csv.resolve()),
        "session_counts_by_split": manifest.get("session_counts_by_split", {}),
        "updated_at": utc_now_iso(),
    }
    with split_seed_json.open("w") as handle:
        json.dump(split_seed_payload, handle, indent=2, default=json_default)

    return {
        "manifest": manifest,
        "manifest_json": manifest_path,
        "summary_csv": summary_path,
        "train_indices_csv": train_csv,
        "validation_indices_csv": val_csv,
        "test_indices_csv": test_csv,
        "train_subset": Subset(dataset, train_indices),
        "validation_subset": Subset(dataset, val_indices),
        "test_subset": Subset(dataset, test_indices),
        "canonical_split_json": split_seed_json,
    }


def write_local_split_manifest(run_dir: Path, shared_split: dict[str, object]) -> tuple[Path, Path]:
    manifest_path = run_dir / "split_manifest.json"
    summary_path = run_dir / "split_summary.csv"
    shared_manifest = dict(shared_split["manifest"])
    shared_manifest["shared_split_manifest_json"] = str(shared_split["manifest_json"])
    shared_manifest["shared_split_summary_csv"] = str(shared_split["summary_csv"])
    shared_manifest["canonical_split_json"] = str(shared_split["canonical_split_json"])
    with manifest_path.open("w") as handle:
        json.dump(shared_manifest, handle, indent=2, default=json_default)
    summary_path.write_text(Path(shared_split["summary_csv"]).read_text())
    return manifest_path, summary_path


def subset_subject_rows(
    subset: Subset,
    labels: list[int],
    preds: list[int],
) -> list[dict[str, object]]:
    if not hasattr(subset.dataset, "info"):
        return []
    info = subset.dataset.info.iloc[list(subset.indices)]
    subject_ids = info["subject_id"].astype(str).tolist()
    by_subject: dict[str, dict[str, list[int]]] = {}
    for subject_id, label, pred in zip(subject_ids, labels, preds):
        bucket = by_subject.setdefault(subject_id, {"labels": [], "preds": []})
        bucket["labels"].append(label)
        bucket["preds"].append(pred)
    rows: list[dict[str, object]] = []
    for subject_id in sorted(by_subject):
        subject_labels = by_subject[subject_id]["labels"]
        subject_preds = by_subject[subject_id]["preds"]
        rows.append(
            {
                "subject_id": subject_id,
                "num_samples": len(subject_labels),
                "acc": float(np.mean(np.equal(subject_preds, subject_labels))),
                "kappa": safe_kappa(subject_labels, subject_preds),
                "macro_f1": safe_macro_f1(subject_labels, subject_preds),
            }
        )
    return rows


def evaluate_subset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    subset: Subset,
) -> dict[str, object]:
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
            batch_preds = logits.argmax(dim=1)

            batch_size = y.size(0)
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
        "preds": preds,
        "labels": labels,
        "num_samples": total_examples,
        "subject_rows": subset_subject_rows(subset, labels, preds),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
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
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_examples += batch_size

    return total_loss / max(total_examples, 1), total_correct / max(total_examples, 1)


def checkpoint_payload(
    args: argparse.Namespace,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau | None,
    epoch: int,
    best_val_acc: float,
    best_val_kappa: float | None,
    best_epoch: int,
    run_split_manifest_path: Path,
) -> dict[str, object]:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
        "best_val_kappa": best_val_kappa,
        "best_epoch": best_epoch,
        "epoch": epoch,
        "args": vars(args),
        "split_manifest_json": str(run_split_manifest_path),
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
    best_val_acc: float,
    best_val_kappa: float | None,
    best_epoch: int,
    run_split_manifest_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        checkpoint_payload(
            args,
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_acc,
            best_val_kappa,
            best_epoch,
            run_split_manifest_path,
        ),
        path,
    )


def write_subject_metrics_csv(path: Path, rows: list[dict[str, object]], seed: int) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["subject_id", "num_samples", "test_acc", "test_kappa", "macro_f1_test", "seed"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "subject_id": row["subject_id"],
                    "num_samples": row["num_samples"],
                    "test_acc": row["acc"],
                    "test_kappa": row["kappa"],
                    "macro_f1_test": row["macro_f1"],
                    "seed": seed,
                }
            )


def write_test_predictions_csv(path: Path, subset: Subset, labels: list[int], preds: list[int]) -> None:
    info = subset.dataset.info.iloc[list(subset.indices)].copy()
    columns = [column for column in ["clip_id", "subject_id", "session", "run", "trial_id", "label"] if column in info.columns]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset_index", *columns, "target", "prediction"],
        )
        writer.writeheader()
        for dataset_index, (_, row), target, prediction in zip(subset.indices, info.iterrows(), labels, preds):
            payload = {"dataset_index": dataset_index, "target": target, "prediction": prediction}
            for column in columns:
                payload[column] = row[column]
            writer.writerow(payload)


def write_confusion_matrix_csv(path: Path, labels: list[int], preds: list[int]) -> None:
    matrix = confusion_matrix(labels, preds, labels=list(range(4)))
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", 0, 1, 2, 3])
        for true_label, row in enumerate(matrix):
            writer.writerow([true_label, *row.tolist()])


def plot_history(history: list[dict[str, object]], output_path: Path, title: str) -> None:
    mpl_config_dir = output_path.parent.parent / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, [row["train_acc"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{title} accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def config_payload(args: argparse.Namespace) -> dict[str, object]:
    return {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "plateau_factor": args.plateau_factor,
        "plateau_patience": args.plateau_patience,
        "early_stopping_patience": args.early_stopping_patience,
    }


def apply_preset_defaults(args: argparse.Namespace) -> None:
    preset = CONFIG_PRESETS.get(args.config_name)
    if not preset:
        return
    parser_defaults = {
        "epochs": 300,
        "learning_rate": 5e-4,
        "batch_size": 64,
        "weight_decay": 0.0,
        "scheduler": "none",
        "plateau_factor": 0.5,
        "plateau_patience": 10,
        "early_stopping_patience": 50,
    }
    for key, value in preset.items():
        current = getattr(args, key)
        if current == parser_defaults.get(key):
            setattr(args, key, value)


def write_summary(path: Path, summary: dict[str, object]) -> None:
    with path.open("w") as handle:
        json.dump(summary, handle, indent=2, default=json_default)


def main() -> None:
    args = parse_args()
    apply_preset_defaults(args)
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    if not args.data_root.exists():
        raise FileNotFoundError(f"Dataset not found at {args.data_root.resolve()}")

    seed_everything(args.seed)
    device = get_device(args.device)
    split_seed = resolve_split_seed(args)
    args.experiment_root.mkdir(parents=True, exist_ok=True)
    if args.run_id:
        run_id = args.run_id
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{args.config_name}_seed{args.seed}_split{split_seed}_{timestamp}"

    run_dir = make_run_dir(args.run_root, run_id, args.overwrite)
    checkpoint_best_path = run_dir / "checkpoint_best.pt"
    checkpoint_last_path = run_dir / "checkpoint_last.pt"
    history_path = run_dir / "history.csv"
    figure_path = run_dir / "figures" / "training_curves.png"
    subject_test_metrics_path = run_dir / "subject_test_metrics.csv"
    test_predictions_path = run_dir / "test_predictions.csv"
    confusion_matrix_path = run_dir / "test_confusion_matrix.csv"
    args_path = write_args(args, run_dir)
    init_history(history_path)

    dataset = build_dataset(args.data_root, verbose=True)
    shared_split = materialize_shared_split(
        dataset=dataset,
        experiment_root=args.experiment_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=split_seed,
        limit_samples=args.limit_samples,
    )
    train_set = shared_split["train_subset"]
    val_set = shared_split["validation_subset"]
    test_set = shared_split["test_subset"]
    run_split_manifest_path, run_split_summary_path = write_local_split_manifest(run_dir, shared_split)

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

    best_val_acc = -math.inf
    best_val_kappa: float | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    status = "running"
    history: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_subset(model, val_loader, device, val_set)
        val_loss = float(val_metrics["loss"])
        val_acc = float(val_metrics["acc"])
        val_kappa = val_metrics["kappa"]
        val_macro_f1 = val_metrics["macro_f1"]
        lr = optimizer.param_groups[0]["lr"]
        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_kappa": val_kappa,
            "val_macro_f1": val_macro_f1,
            "lr": lr,
        }
        history.append(history_row)
        append_history_row(history_row, history_path)

        print(
            f"epoch={epoch:03d} config={args.config_name} seed={args.seed} split={split_seed} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={lr:.6g}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_kappa = None if val_kappa is None else float(val_kappa)
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_best_path,
                args,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_acc,
                best_val_kappa,
                best_epoch,
                run_split_manifest_path,
            )
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            scheduler.step(val_acc)

        save_checkpoint(
            checkpoint_last_path,
            args,
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_acc,
            best_val_kappa,
            best_epoch,
            run_split_manifest_path,
        )
        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            save_checkpoint(
                run_dir / "checkpoints" / f"checkpoint_epoch_{epoch:03d}.pt",
                args,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_acc,
                best_val_kappa,
                best_epoch,
                run_split_manifest_path,
            )

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            stopped_early = True
            status = "early_stopped"
            print(f"early stopping after {epochs_without_improvement} epochs without validation improvement")
            break

    if status == "running":
        status = "completed"

    best_checkpoint = torch.load(checkpoint_best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    final_train_metrics = evaluate_subset(model, train_loader, device, train_set)
    final_val_metrics = evaluate_subset(model, val_loader, device, val_set)
    test_metrics = evaluate_subset(model, test_loader, device, test_set)

    write_subject_metrics_csv(subject_test_metrics_path, test_metrics["subject_rows"], args.seed)
    write_test_predictions_csv(test_predictions_path, test_set, test_metrics["labels"], test_metrics["preds"])
    write_confusion_matrix_csv(confusion_matrix_path, test_metrics["labels"], test_metrics["preds"])
    plot_history(history, figure_path, f"{args.config_name} seed={args.seed}")

    summary = {
        "model": "EEGNet",
        "block": "eegnet_baseline_random_split_60_20_20",
        "protocol": "random_train_validation_test_60_20_20_preliminary",
        "config_name": args.config_name,
        "phase_tag": args.phase_tag,
        "seed": args.seed,
        "split_seed": split_seed,
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
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "best_val_acc": best_val_acc,
        "best_val_kappa": best_val_kappa,
        "best_epoch": best_epoch,
        "final_train_acc": final_train_metrics["acc"],
        "final_train_kappa": final_train_metrics["kappa"],
        "final_train_macro_f1": final_train_metrics["macro_f1"],
        "final_val_acc": final_val_metrics["acc"],
        "final_val_kappa": final_val_metrics["kappa"],
        "final_val_macro_f1": final_val_metrics["macro_f1"],
        "test_acc": test_metrics["acc"],
        "test_kappa": test_metrics["kappa"],
        "test_macro_f1": test_metrics["macro_f1"],
        "train_loss": final_train_metrics["loss"],
        "val_loss": final_val_metrics["loss"],
        "test_loss": test_metrics["loss"],
        "stopped_early": stopped_early,
        "mixes_sessions": shared_split["manifest"].get("mixes_sessions"),
        "shared_split_manifest_json": str(shared_split["manifest_json"]),
        "shared_split_summary_csv": str(shared_split["summary_csv"]),
        "run_split_manifest_json": str(run_split_manifest_path),
        "run_split_summary_csv": str(run_split_summary_path),
        "train_indices_csv": str(shared_split["train_indices_csv"]),
        "validation_indices_csv": str(shared_split["validation_indices_csv"]),
        "test_indices_csv": str(shared_split["test_indices_csv"]),
        "subject_test_metrics_csv": str(subject_test_metrics_path),
        "test_predictions_csv": str(test_predictions_path),
        "test_confusion_matrix_csv": str(confusion_matrix_path),
        "history_csv": str(history_path),
        "summary_json": str(run_dir / "summary.json"),
        "args_json": str(args_path),
        "figures_path": str(figure_path),
        "run_dir": str(run_dir),
        "status": status,
    }
    write_summary(run_dir / "summary.json", summary)

    print(f"run_dir={run_dir}")
    print(f"best_val_acc={best_val_acc:.4f}")
    print(f"best_epoch={best_epoch}")
    print(f"final_train_acc={float(final_train_metrics['acc']):.4f}")
    print(f"final_val_acc={float(final_val_metrics['acc']):.4f}")
    print(f"test_acc={float(test_metrics['acc']):.4f}")
    print(f"test_kappa={test_metrics['kappa']}")
    print(f"test_macro_f1={test_metrics['macro_f1']}")
    print(f"shared_split_manifest_json={shared_split['manifest_json']}")


if __name__ == "__main__":
    main()
