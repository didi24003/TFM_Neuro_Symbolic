#!/usr/bin/env python
"""Verify the artifact-free BCICIV 2a configuration."""

from __future__ import annotations

from pathlib import Path

from experiments_mi_ltn.mi_ltn_common import DEFAULT_DATA_ROOT, NUM_ELECTRODES, build_dataset


EXPECTED_FULL = 5184
EXPECTED_SKIP = 4696
EXPECTED_REMOVED = 488


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_root = (repo_root / DEFAULT_DATA_ROOT).resolve() if not DEFAULT_DATA_ROOT.is_absolute() else DEFAULT_DATA_ROOT.resolve()
    io_root = repo_root / "experiments_mi_ltn" / "io"

    full_ds = build_dataset(
        data_root=data_root,
        io_path=io_root / "bciciv2a_verify_full",
        verbose=False,
        skip_trial_with_artifacts=False,
    )
    skip_ds = build_dataset(
        data_root=data_root,
        io_path=io_root / "bciciv2a_verify_skip",
        verbose=False,
        skip_trial_with_artifacts=True,
    )

    full_len = len(full_ds)
    skip_len = len(skip_ds)
    removed = full_len - skip_len
    sessions_full = full_ds.info.groupby("session").size().to_dict()
    sessions_skip = skip_ds.info.groupby("session").size().to_dict()
    labels_full = sorted(full_ds.info["label"].astype(int).unique().tolist())
    subjects_full = sorted(full_ds.info["subject_id"].astype(str).unique().tolist())

    print("Artifact-free verification")
    print(f"data_root={data_root}")
    print(f"full_trials={full_len}")
    print(f"skip_trials={skip_len}")
    print(f"removed_trials={removed}")
    print(f"num_channel={NUM_ELECTRODES}")
    print("eog_enters_model=no")
    print(f"sessions_full={sessions_full}")
    print(f"sessions_skip={sessions_skip}")
    print(f"subjects={subjects_full}")
    print(f"labels={labels_full}")

    if full_len != EXPECTED_FULL:
        raise SystemExit(f"Unexpected full dataset size: {full_len} != {EXPECTED_FULL}")
    if skip_len != EXPECTED_SKIP:
        raise SystemExit(f"Unexpected artifact-free dataset size: {skip_len} != {EXPECTED_SKIP}")
    if removed != EXPECTED_REMOVED:
        raise SystemExit(f"Unexpected removed trial count: {removed} != {EXPECTED_REMOVED}")
    if NUM_ELECTRODES != 22:
        raise SystemExit(f"Unexpected EEG channel count: {NUM_ELECTRODES}")
    if set(sessions_full) != {"T", "E"} or set(sessions_skip) != {"T", "E"}:
        raise SystemExit("Sessions T/E are not both present after filtering.")

    print("verification_status=OK")


if __name__ == "__main__":
    main()
