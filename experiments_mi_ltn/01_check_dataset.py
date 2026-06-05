#!/usr/bin/env python
"""Check BCI IV 2a loading and a single EEGNet forward pass."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mi_ltn_common import DEFAULT_DATA_ROOT, build_dataset, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="Only check paths/imports/model construction.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = build_model()
    print(model.__class__.__name__, "OK")

    if args.dry_run:
        print(f"Dry run OK. Expected dataset path: {args.data_root.resolve()}")
        return

    if not args.data_root.exists():
        raise FileNotFoundError(
            f"Dataset not found at {args.data_root.resolve()}. "
            "Run with --dry-run until BCI IV 2a .mat files are available."
        )

    dataset = build_dataset(args.data_root, verbose=True)
    x, y = dataset[0]
    with torch.no_grad():
        output = model(x.unsqueeze(0))

    print(f"x shape: {tuple(x.shape)}")
    print(f"label: {y}")
    print(f"model output shape: {tuple(output.shape)}")


if __name__ == "__main__":
    main()
