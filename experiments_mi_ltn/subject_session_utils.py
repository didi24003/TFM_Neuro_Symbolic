"""Helpers for subject/session-specific experiment protocols."""

from __future__ import annotations

import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from torch.utils.data import Dataset, Subset


EXPERIMENT_ROOT = Path(__file__).resolve().parent
SPLITS_DIR = EXPERIMENT_ROOT / "runs" / "shared_trial_splits"


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
        f.write("\n")
    os.replace(tmp_path, path)


def ensure_info_columns(dataset: Dataset, required_columns: set[str]) -> None:
    if not hasattr(dataset, "info"):
        raise AttributeError(f"Dataset must expose an info table with columns: {sorted(required_columns)}")
    info = dataset.info
    missing = sorted(required_columns.difference(info.columns))
    if missing:
        raise ValueError(f"Dataset info is missing required columns: {missing}")


def subject_ids(dataset: Dataset) -> list[str]:
    ensure_info_columns(dataset, {"subject_id"})
    return sorted(dataset.info["subject_id"].astype(str).unique().tolist())


def subject_sessions(dataset: Dataset, subject_id: str) -> list[str]:
    ensure_info_columns(dataset, {"subject_id", "session"})
    info = dataset.info
    mask = info["subject_id"].astype(str) == str(subject_id)
    return sorted(info.loc[mask, "session"].astype(str).unique().tolist())


def _sort_indices_by_trial_metadata(dataset: Dataset, indices: list[int]) -> list[int]:
    if not indices:
        return []
    ensure_info_columns(dataset, {"subject_id", "session", "run", "trial_id"})
    info = dataset.info.iloc[indices].copy()
    info["dataset_index"] = indices
    info["run"] = info["run"].astype(int)
    info["trial_id"] = info["trial_id"].astype(int)
    ordered = info.sort_values(["subject_id", "session", "run", "trial_id", "dataset_index"])
    return [int(index) for index in ordered["dataset_index"].tolist()]


def dataset_indices_by_subject_session(
    dataset: Dataset,
    subject_id: str,
    session: str,
) -> list[int]:
    ensure_info_columns(dataset, {"subject_id", "session"})
    info = dataset.info
    mask = (
        info["subject_id"].astype(str) == str(subject_id)
    ) & (
        info["session"].astype(str) == str(session)
    )
    indices = [int(index) for index in info.index[mask].tolist()]
    return _sort_indices_by_trial_metadata(dataset, indices)


def subset_from_indices(dataset: Dataset, indices: list[int]) -> Subset:
    return Subset(dataset, list(indices))


def split_subject_session_indices(
    dataset: Dataset,
    subject_id: str,
    train_session: str = "T",
    test_session: str = "E",
    val_ratio: float = 0.2,
    split_seed: int = 42,
    val_mode: str = "stratified_trialwise",
    run_holdout_index: Optional[int] = None,
) -> dict[str, Any]:
    ensure_info_columns(dataset, {"subject_id", "session", "label", "run", "trial_id"})
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    sessions = subject_sessions(dataset, subject_id)
    missing_sessions = [session for session in (train_session, test_session) if session not in sessions]
    if missing_sessions:
        raise ValueError(
            f"Subject {subject_id} is missing required sessions {missing_sessions}. Available sessions: {sessions}"
        )

    train_session_indices = dataset_indices_by_subject_session(dataset, subject_id, train_session)
    test_indices = dataset_indices_by_subject_session(dataset, subject_id, test_session)

    info = dataset.info
    train_info = info.iloc[train_session_indices].copy()
    train_info["dataset_index"] = train_session_indices

    if val_mode == "stratified_trialwise":
        by_class: dict[int, list[int]] = defaultdict(list)
        for row in train_info.itertuples(index=False):
            by_class[int(row.label)].append(int(row.dataset_index))

        rng = random.Random(split_seed)
        train_indices: list[int] = []
        val_indices: list[int] = []
        for class_label in sorted(by_class):
            bucket = list(by_class[class_label])
            rng.shuffle(bucket)
            if len(bucket) < 2:
                raise ValueError(
                    f"Subject {subject_id} session {train_session} class {class_label} "
                    "does not have enough samples for train/validation split."
                )
            val_count = max(1, int(round(len(bucket) * val_ratio)))
            val_count = min(val_count, len(bucket) - 1)
            train_indices.extend(bucket[val_count:])
            val_indices.extend(bucket[:val_count])
    elif val_mode == "run_holdout":
        unique_runs = sorted(train_info["run"].astype(int).unique().tolist())
        if len(unique_runs) < 2:
            raise ValueError(
                f"Subject {subject_id} session {train_session} needs at least 2 runs for run_holdout validation."
            )
        selected_position = (
            split_seed % len(unique_runs)
            if run_holdout_index is None
            else max(0, min(int(run_holdout_index), len(unique_runs) - 1))
        )
        val_run = unique_runs[selected_position]
        val_mask = train_info["run"].astype(int) == val_run
        val_indices = [int(index) for index in train_info.loc[val_mask, "dataset_index"].tolist()]
        train_indices = [int(index) for index in train_info.loc[~val_mask, "dataset_index"].tolist()]
        if not train_indices or not val_indices:
            raise ValueError(
                f"Subject {subject_id} session {train_session} produced empty train/validation split in run_holdout mode."
            )
    else:
        raise ValueError(f"Unsupported val_mode: {val_mode}")

    train_indices = _sort_indices_by_trial_metadata(dataset, train_indices)
    val_indices = _sort_indices_by_trial_metadata(dataset, val_indices)
    test_indices = _sort_indices_by_trial_metadata(dataset, test_indices)

    return {
        "subject_id": str(subject_id),
        "available_sessions": sessions,
        "train_session": str(train_session),
        "test_session": str(test_session),
        "val_mode": val_mode,
        "split_seed": int(split_seed),
        "run_holdout_index": run_holdout_index,
        "train_indices": train_indices,
        "validation_indices": val_indices,
        "test_indices": test_indices,
        "train_subset": subset_from_indices(dataset, train_indices),
        "validation_subset": subset_from_indices(dataset, val_indices),
        "test_subset": subset_from_indices(dataset, test_indices),
    }


def _write_index_csv(path: Path, indices: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset_index"])
        writer.writeheader()
        for index in indices:
            writer.writerow({"dataset_index": int(index)})


def save_subject_session_split(
    output_dir: Path,
    split_artifacts: dict[str, Any],
    val_ratio: float,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_indices_path = output_dir / "train_indices.csv"
    validation_indices_path = output_dir / "validation_indices.csv"
    test_indices_path = output_dir / "test_indices.csv"
    manifest_path = output_dir / "split_manifest.json"

    _write_index_csv(train_indices_path, split_artifacts["train_indices"])
    _write_index_csv(validation_indices_path, split_artifacts["validation_indices"])
    _write_index_csv(test_indices_path, split_artifacts["test_indices"])

    manifest = {
        "protocol": "subject_specific_cross_session",
        "subject_id": split_artifacts["subject_id"],
        "available_sessions": split_artifacts["available_sessions"],
        "train_session": split_artifacts["train_session"],
        "test_session": split_artifacts["test_session"],
        "val_mode": split_artifacts["val_mode"],
        "val_ratio": val_ratio,
        "split_seed": split_artifacts["split_seed"],
        "run_holdout_index": split_artifacts["run_holdout_index"],
        "train_indices": split_artifacts["train_indices"],
        "validation_indices": split_artifacts["validation_indices"],
        "test_indices": split_artifacts["test_indices"],
        "train_size": len(split_artifacts["train_indices"]),
        "validation_size": len(split_artifacts["validation_indices"]),
        "test_size": len(split_artifacts["test_indices"]),
        "train_indices_csv": train_indices_path,
        "validation_indices_csv": validation_indices_path,
        "test_indices_csv": test_indices_path,
    }
    write_json(manifest_path, manifest)
    return {
        "split_manifest_json": manifest_path,
        "train_indices_csv": train_indices_path,
        "validation_indices_csv": validation_indices_path,
        "test_indices_csv": test_indices_path,
    }
