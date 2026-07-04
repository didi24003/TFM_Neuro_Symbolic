"""Shared helpers for the BCI IV 2a EEGNet experiments.

The helpers live inside the experiment folder so the TorchEEG core package is
left untouched.
"""

from __future__ import annotations

import importlib.util
import random
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from torcheeg import transforms
from torcheeg.datasets import BCICIV2aDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "BCICIV_2a_mat"
DEFAULT_IO_ROOT = EXPERIMENT_ROOT / "io" / "bciciv2a"
RUNS_DIR = EXPERIMENT_ROOT / "runs"

CHUNK_SIZE = 1750
NUM_ELECTRODES = 22
NUM_CLASSES = 4

BCI_IV_2A_CHANNELS = [
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_eegnet_class():
    """Load EEGNet without importing torcheeg.models globally.

    In this checkout, `from torcheeg.models import EEGNet` imports optional GNN
    modules and fails when torch_scatter is not installed. Loading the CNN file
    directly keeps the baseline independent of that optional dependency.
    """

    eegnet_path = REPO_ROOT / "torcheeg" / "models" / "cnn" / "eegnet.py"
    spec = importlib.util.spec_from_file_location("torcheeg_eegnet_direct", eegnet_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load EEGNet from {eegnet_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EEGNet


def build_model() -> torch.nn.Module:
    EEGNet = load_eegnet_class()
    return EEGNet(
        chunk_size=CHUNK_SIZE,
        num_electrodes=NUM_ELECTRODES,
        num_classes=NUM_CLASSES,
    )


def build_dataset(
    data_root: Path | str = DEFAULT_DATA_ROOT,
    io_path: Optional[Path | str] = DEFAULT_IO_ROOT,
    verbose: bool = False,
    skip_trial_with_artifacts: bool = False,
) -> BCICIV2aDataset:
    data_root = Path(data_root).expanduser().resolve()
    if io_path is not None:
        io_path = Path(io_path).expanduser().resolve()
        if skip_trial_with_artifacts and io_path == DEFAULT_IO_ROOT.resolve():
            io_path = io_path.parent / f"{io_path.name}_skip_artifacts"
        io_path.parent.mkdir(parents=True, exist_ok=True)

    return BCICIV2aDataset(
        root_path=str(data_root),
        io_path=str(io_path) if io_path is not None else None,
        chunk_size=CHUNK_SIZE,
        num_channel=NUM_ELECTRODES,
        skip_trial_with_artifacts=skip_trial_with_artifacts,
        online_transform=transforms.Compose(
            [
                transforms.To2d(),
                transforms.ToTensor(),
            ]
        ),
        label_transform=transforms.Compose(
            [
                transforms.Select("label"),
                transforms.Lambda(lambda x: int(x) - 1),
            ]
        ),
        verbose=verbose,
    )


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    return value in {"1", "true", "yes", "on"}


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def split_dataset(
    dataset: Dataset,
    val_ratio: float = 0.2,
    seed: int = 42,
    limit_samples: Optional[int] = None,
) -> Tuple[Dataset, Dataset]:
    if limit_samples is not None:
        n = min(limit_samples, len(dataset))
        dataset = Subset(dataset, list(range(n)))

    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    if train_size < 1:
        raise ValueError("Dataset too small to create train/validation splits.")

    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def make_loaders(
    dataset: Dataset,
    batch_size: int,
    val_ratio: float,
    seed: int,
    num_workers: int,
    limit_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    train_set, val_set = split_dataset(dataset, val_ratio, seed, limit_samples)
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> int:
    return (logits.argmax(dim=1) == labels).sum().item()


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    non_blocking = device.type == "cuda"

    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=non_blocking)
            y = y.long().to(device, non_blocking=non_blocking)
            logits = model(x)
            loss = criterion(logits, y)

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_correct += accuracy_from_logits(logits, y)
            total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples
