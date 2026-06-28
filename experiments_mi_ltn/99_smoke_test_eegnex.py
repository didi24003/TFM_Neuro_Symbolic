#!/usr/bin/env python
"""Smoke test for EEGNeX on the real BCI IV 2a batch shape."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mi_ltn_common import DEFAULT_DATA_ROOT, build_dataset
from model_factory import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def instantiate_eegnex(n_times: int):
    preferred_kwargs = {"kernel_block_1_2": 32}
    try:
        model = build_model(
            model_name="eegnex",
            n_chans=22,
            n_outputs=4,
            n_times=n_times,
            model_kwargs=preferred_kwargs,
        )
        return model, preferred_kwargs, None
    except Exception as exc:
        fallback_kwargs = {}
        model = build_model(
            model_name="eegnex",
            n_chans=22,
            n_outputs=4,
            n_times=n_times,
            model_kwargs=fallback_kwargs,
        )
        return model, fallback_kwargs, exc


def main() -> None:
    args = parse_args()
    if not args.data_root.exists():
        raise FileNotFoundError(f"Dataset not found at {args.data_root.resolve()}")

    dataset = build_dataset(args.data_root, verbose=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    batch_x, batch_y = next(iter(loader))
    model_input = batch_x

    if batch_x.ndim == 4 and batch_x.shape[1] == 1:
        model_input = batch_x.squeeze(1)

    n_times = model_input.shape[-1]
    model, used_kwargs, preferred_error = instantiate_eegnex(n_times=n_times)
    model.eval()

    with torch.inference_mode():
        logits = model(model_input)

    if logits.ndim != 2 or logits.shape[0] != model_input.shape[0] or logits.shape[1] != 4:
        raise RuntimeError(
            f"Unexpected EEGNeX output shape {tuple(logits.shape)} for input {tuple(model_input.shape)}"
        )

    print("EEGNeX smoke test OK")
    print(f"dataset_path: {args.data_root.resolve()}")
    print(f"raw_batch_x_shape: {tuple(batch_x.shape)}")
    print(f"model_input_shape: {tuple(model_input.shape)}")
    print(f"batch_y_shape: {tuple(batch_y.shape)}")
    print(f"batch_dtype: {batch_x.dtype}")
    print(f"labels_dtype: {batch_y.dtype}")
    print(f"n_times: {n_times}")
    print(f"model_class: {model.__class__.__name__}")
    print(f"model_kwargs_used: {used_kwargs}")
    if preferred_error is not None:
        print(
            "kernel_block_1_2=32 fallback:"
            f" {type(preferred_error).__name__}: {preferred_error}"
        )
    print(f"logits_shape: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
