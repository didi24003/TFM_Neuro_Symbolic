#!/usr/bin/env python
"""Train EEGNet with a simple LTN-inspired logic loss."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from mi_ltn_common import (
    DEFAULT_DATA_ROOT,
    RUNS_DIR,
    build_dataset,
    build_model,
    evaluate,
    get_device,
    make_loaders,
    seed_everything,
)


SUMMARY_CSV = RUNS_DIR / "logic_loss_eegnet" / "logic_loss_runs_summary.csv"
SUMMARY_FIELDNAMES = [
    "run_id",
    "model",
    "seed",
    "epochs",
    "optimizer",
    "lr",
    "batch_size",
    "weight_decay",
    "scheduler",
    "early_stopping_patience",
    "checkpoint_every",
    "lambda_logic",
    "baseline_val_acc",
    "best_val_acc",
    "best_epoch",
    "final_train_acc",
    "final_val_acc",
    "checkpoint_best_path",
    "checkpoint_last_path",
    "history_csv",
    "summary_json",
    "args_json",
    "figures_path",
    "status",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lambda-logic", type=float, default=0.1)
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=RUNS_DIR / "best_eegnet_bciciv2a.pt",
        help=(
            "Reference baseline checkpoint used only to evaluate baseline_val_acc on the "
            "same validation split. It is not used to initialize or resume logic-loss training."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional extra best checkpoint path.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--run-root", type=Path, default=RUNS_DIR / "logic_loss_eegnet")
    parser.add_argument("--run-id", default=None, help="Optional explicit run folder name.")
    parser.add_argument("--scheduler", choices=("none", "plateau"), default="none")
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=0, help="0 disables early stopping.")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Save checkpoints/checkpoint_epoch_XXX.pt every N epochs. 0 disables periodic checkpoints.",
    )
    return parser.parse_args()


def logic_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    true_probs = probs.gather(1, labels.view(-1, 1)).squeeze(1)
    return 1.0 - true_probs.mean()


def train_one_epoch(model, loader, ce_loss, optimizer, device, lambda_logic):
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_logic = 0.0
    total_correct = 0
    total_examples = 0
    non_blocking = device.type == "cuda"

    for x, y in loader:
        x = x.to(device, non_blocking=non_blocking)
        y = y.long().to(device, non_blocking=non_blocking)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        ce = ce_loss(logits, y)
        logic = logic_loss_from_logits(logits, y)
        loss = ce + lambda_logic * logic
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_ce += ce.item() * batch_size
        total_logic += logic.item() * batch_size
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_examples += batch_size

    return (
        total_loss / total_examples,
        total_ce / total_examples,
        total_logic / total_examples,
        total_correct / total_examples,
    )


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


def load_baseline_accuracy(args, val_loader, device):
    if not args.baseline_checkpoint.exists():
        return None
    model = build_model().to(device)
    state_dict = load_checkpoint_state_dict(args.baseline_checkpoint, device)
    model.load_state_dict(state_dict)
    _, acc = evaluate(model, val_loader, device)
    return acc


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def make_run_dir(args: argparse.Namespace) -> Path:
    args.run_root.mkdir(parents=True, exist_ok=True)
    if args.run_id:
        run_id = args.run_id
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        lambda_tag = str(args.lambda_logic).replace(".", "p")
        run_id = f"logic_lam{lambda_tag}_seed{args.seed}_epochs{args.epochs}_{timestamp}"
    run_dir = args.run_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir.resolve()}")
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
        "train_logic",
        "train_acc",
        "val_loss",
        "val_acc",
        "lr",
    ]
    with history_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_history_row(row: dict[str, float | int], history_path: Path) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "train_ce",
        "train_logic",
        "train_acc",
        "val_loss",
        "val_acc",
        "lr",
    ]
    with history_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def final_metrics(history: list[dict[str, float | int]]) -> tuple[float | None, float | None]:
    if not history:
        return None, None
    last = history[-1]
    return float(last["train_acc"]), float(last["val_acc"])


def write_summary(
    args: argparse.Namespace,
    run_dir: Path,
    checkpoint_best_path: Path,
    checkpoint_last_path: Path,
    history_path: Path,
    args_path: Path,
    figure_path: Path,
    baseline_acc: float | None,
    best_val_acc: float,
    best_epoch: int,
    epochs_completed: int,
    stopped_early: bool,
    status: str,
) -> None:
    summary = {
        "model": "EEGNet",
        "block": "logic_loss_eegnet",
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
        "lambda_logic": args.lambda_logic,
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "baseline_val_acc": baseline_acc,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "checkpoint_best_path": str(checkpoint_best_path),
        "checkpoint_last_path": str(checkpoint_last_path),
        "checkpoint_path": str(checkpoint_best_path),
        "history_csv": str(history_path),
        "summary_json": str(run_dir / "summary.json"),
        "args_json": str(args_path),
        "figures_path": str(figure_path),
        "run_dir": str(run_dir),
        "stopped_early": stopped_early,
        "status": status,
    }
    with (run_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


def checkpoint_payload(
    args: argparse.Namespace,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau | None,
    epoch: int,
    best_val_acc: float,
    best_epoch: int,
    baseline_acc: float | None,
) -> dict:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "epoch": epoch,
        "lambda_logic": args.lambda_logic,
        "baseline_val_acc": baseline_acc,
        "args": vars(args),
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
    best_epoch: int,
    baseline_acc: float | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        checkpoint_payload(
            args, model, optimizer, scheduler, epoch, best_val_acc, best_epoch, baseline_acc
        ),
        path,
    )


def summary_row(
    args: argparse.Namespace,
    run_dir: Path,
    checkpoint_best_path: Path,
    checkpoint_last_path: Path,
    history_path: Path,
    args_path: Path,
    figure_path: Path,
    baseline_acc: float | None,
    best_val_acc: float,
    best_epoch: int,
    history: list[dict[str, float | int]],
    status: str,
    notes: str = "",
) -> dict:
    final_train_acc, final_val_acc = final_metrics(history)
    return {
        "run_id": run_dir.name,
        "model": "EEGNet",
        "seed": args.seed,
        "epochs": len(history),
        "optimizer": "Adam",
        "lr": args.learning_rate,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "early_stopping_patience": args.early_stopping_patience,
        "checkpoint_every": args.checkpoint_every,
        "lambda_logic": args.lambda_logic,
        "baseline_val_acc": "" if baseline_acc is None else baseline_acc,
        "best_val_acc": best_val_acc if best_val_acc >= 0 else "",
        "best_epoch": best_epoch if best_epoch > 0 else "",
        "final_train_acc": "" if final_train_acc is None else final_train_acc,
        "final_val_acc": "" if final_val_acc is None else final_val_acc,
        "checkpoint_best_path": str(checkpoint_best_path),
        "checkpoint_last_path": str(checkpoint_last_path),
        "history_csv": str(history_path),
        "summary_json": str(run_dir / "summary.json"),
        "args_json": str(args_path),
        "figures_path": str(figure_path),
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


def plot_history(history: list[dict[str, float | int]], output_path: Path) -> None:
    mpl_config_dir = output_path.parent.parent / ".matplotlib"
    mpl_config_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train total")
    axes[0].plot(epochs, [row["train_ce"] for row in history], label="train CE")
    axes[0].plot(epochs, [row["train_logic"] for row in history], label="train logic")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation CE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("EEGNet logic loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, [row["train_acc"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("EEGNet logic accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend(frameon=False)

    axes[2].plot(epochs, [row["lr"] for row in history], label="lr")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning rate")
    axes[2].set_title("Learning rate")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.data_root.exists():
        raise FileNotFoundError(f"Dataset not found at {args.data_root.resolve()}")

    seed_everything(args.seed)
    device = get_device(args.device)
    run_dir = make_run_dir(args)
    checkpoint_best_path = run_dir / "checkpoint_best.pt"
    checkpoint_last_path = run_dir / "checkpoint_last.pt"
    history_path = run_dir / "history.csv"
    figure_path = run_dir / "figures" / "training_curves.png"
    args_path = write_args(args, run_dir)
    init_history(history_path)

    dataset = build_dataset(args.data_root, verbose=True)
    train_loader, val_loader = make_loaders(
        dataset,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        limit_samples=args.limit_samples,
    )

    baseline_acc = load_baseline_accuracy(args, val_loader, device)
    if baseline_acc is not None:
        print(f"baseline_val_acc={baseline_acc:.4f}")
    else:
        print(f"baseline checkpoint not found: {args.baseline_checkpoint}")

    print("logic model initialization: fresh EEGNet weights from seed, no baseline checkpoint loaded")
    model = build_model().to(device)
    ce_loss = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
        )

    best_val_acc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    history = []
    status = "running"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_ce, train_logic, train_acc = train_one_epoch(
            model, train_loader, ce_loss, optimizer, device, args.lambda_logic
        )
        val_loss, val_acc = evaluate(model, val_loader, device)
        lr = optimizer.param_groups[0]["lr"]
        delta = "" if baseline_acc is None else f" delta_vs_baseline={val_acc - baseline_acc:+.4f}"
        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ce": train_ce,
            "train_logic": train_logic,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": lr,
        }
        history.append(history_row)
        append_history_row(history_row, history_path)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_ce={train_ce:.4f} "
            f"train_logic={train_logic:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={lr:.6g}{delta}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
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
                best_epoch,
                baseline_acc,
            )
            if args.checkpoint is not None:
                save_checkpoint(
                    args.checkpoint,
                    args,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_val_acc,
                    best_epoch,
                    baseline_acc,
                )
            print(f"saved best logic checkpoint: {checkpoint_best_path}")
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
            best_epoch,
            baseline_acc,
        )
        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            periodic_path = run_dir / "checkpoints" / f"checkpoint_epoch_{epoch:03d}.pt"
            save_checkpoint(
                periodic_path,
                args,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_acc,
                best_epoch,
                baseline_acc,
            )
            print(f"saved periodic checkpoint: {periodic_path}")

        write_summary(
            args,
            run_dir,
            checkpoint_best_path,
            checkpoint_last_path,
            history_path,
            args_path,
            figure_path,
            baseline_acc,
            best_val_acc,
            best_epoch,
            len(history),
            stopped_early,
            status,
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
                baseline_acc,
                best_val_acc,
                best_epoch,
                history,
                status,
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
    write_summary(
        args,
        run_dir,
        checkpoint_best_path,
        checkpoint_last_path,
        history_path,
        args_path,
        figure_path,
        baseline_acc,
        best_val_acc,
        best_epoch,
        len(history),
        stopped_early,
        status,
    )
    plot_history(history, figure_path)
    upsert_summary_csv(
        summary_row(
            args,
            run_dir,
            checkpoint_best_path,
            checkpoint_last_path,
            history_path,
            args_path,
            figure_path,
            baseline_acc,
            best_val_acc,
            best_epoch,
            history,
            status,
        )
    )

    print(f"best_logic_val_acc={best_val_acc:.4f}")
    print(f"best_epoch={best_epoch}")
    print(f"run_dir={run_dir}")
    print(f"history_csv={history_path}")
    print(f"training_curves={figure_path}")
    print(f"summary_csv={SUMMARY_CSV}")


if __name__ == "__main__":
    main()
