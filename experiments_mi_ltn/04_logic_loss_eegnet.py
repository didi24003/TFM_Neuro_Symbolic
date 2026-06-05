#!/usr/bin/env python
"""Train EEGNet with a simple LTN-inspired logic loss."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lambda-logic", type=float, default=0.1)
    parser.add_argument("--baseline-checkpoint", type=Path, default=RUNS_DIR / "best_eegnet_bciciv2a.pt")
    parser.add_argument("--checkpoint", type=Path, default=RUNS_DIR / "best_eegnet_logic_bciciv2a.pt")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-samples", type=int, default=None)
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

    for x, y in loader:
        x = x.to(device)
        y = y.long().to(device)

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


def main() -> None:
    args = parse_args()
    if not args.data_root.exists():
        raise FileNotFoundError(f"Dataset not found at {args.data_root.resolve()}")

    seed_everything(args.seed)
    device = get_device(args.device)
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

    model = build_model().to(device)
    ce_loss = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    best_val_acc = -1.0
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_ce, train_logic, train_acc = train_one_epoch(
            model, train_loader, ce_loss, optimizer, device, args.lambda_logic
        )
        val_loss, val_acc = evaluate(model, val_loader, device)
        delta = "" if baseline_acc is None else f" delta_vs_baseline={val_acc - baseline_acc:+.4f}"

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_ce={train_ce:.4f} "
            f"train_logic={train_logic:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}{delta}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_acc": best_val_acc,
                    "epoch": epoch,
                    "lambda_logic": args.lambda_logic,
                    "baseline_val_acc": baseline_acc,
                    "args": vars(args),
                },
                args.checkpoint,
            )
            print(f"saved best logic checkpoint: {args.checkpoint}")

    print(f"best_logic_val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()
