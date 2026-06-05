#!/usr/bin/env python
"""Compute simple sensorimotor coherence scores from channel importance CSVs."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mi_ltn_common import RUNS_DIR


SENSORIMOTOR_CHANNELS = {"FC3", "FC4", "FCz", "C3", "C4", "Cz", "CP3", "CP4", "CPz"}
POSTERIOR_CHANNELS = {"P1", "Pz", "P2", "POz"}
REQUIRED_COLUMNS = {"channel_name", "importance"}
PLOT_COLUMNS = ["sensorimotor_ratio", "posterior_ratio", "sensorimotor_minus_posterior"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=RUNS_DIR / "channel_importance_baseline_30ep.csv",
        help="Input channel-importance CSV for the baseline model.",
    )
    parser.add_argument(
        "--logic-csv",
        type=Path,
        default=RUNS_DIR / "channel_importance_logic_30ep.csv",
        help="Input channel-importance CSV for the logic-loss model.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=RUNS_DIR / "coherence_scores.csv",
        help="Output CSV for coherence scores.",
    )
    parser.add_argument(
        "--output-fig",
        type=Path,
        default=RUNS_DIR / "coherence_scores.png",
        help="Output PNG figure.",
    )
    return parser.parse_args()


def validate_input_csv(path: Path, model_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{model_name} CSV not found at {path.resolve()}")

    df = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{model_name} CSV is missing required columns: {missing}")

    df = df.copy()
    df["channel_name"] = df["channel_name"].astype(str)
    df["importance"] = pd.to_numeric(df["importance"], errors="raise")
    warn_missing_channels(df, model_name, SENSORIMOTOR_CHANNELS, "sensorimotor")
    warn_missing_channels(df, model_name, POSTERIOR_CHANNELS, "posterior")
    return df


def warn_missing_channels(
    df: pd.DataFrame,
    model_name: str,
    expected_channels: set[str],
    channel_group: str,
) -> None:
    present_channels = set(df["channel_name"])
    missing_channels = sorted(expected_channels - present_channels)
    if missing_channels:
        warnings.warn(
            f"{model_name}: missing expected {channel_group} channels: "
            f"{', '.join(missing_channels)}",
            stacklevel=2,
        )


def ratio(numerator: float, denominator: float, label: str, model_name: str) -> float:
    if denominator == 0:
        warnings.warn(
            f"{model_name}: total importance is zero; {label} set to 0.0.",
            stacklevel=2,
        )
        return 0.0
    return numerator / denominator


def compute_scores(model_name: str, df: pd.DataFrame) -> dict[str, float | str]:
    total_importance_sum = df["importance"].sum()
    sensorimotor_importance_sum = df.loc[
        df["channel_name"].isin(SENSORIMOTOR_CHANNELS), "importance"
    ].sum()
    posterior_importance_sum = df.loc[
        df["channel_name"].isin(POSTERIOR_CHANNELS), "importance"
    ].sum()

    sensorimotor_ratio = ratio(
        sensorimotor_importance_sum,
        total_importance_sum,
        "sensorimotor_ratio",
        model_name,
    )
    posterior_ratio = ratio(
        posterior_importance_sum,
        total_importance_sum,
        "posterior_ratio",
        model_name,
    )

    return {
        "model_name": model_name,
        "total_importance_sum": total_importance_sum,
        "sensorimotor_importance_sum": sensorimotor_importance_sum,
        "sensorimotor_ratio": sensorimotor_ratio,
        "posterior_importance_sum": posterior_importance_sum,
        "posterior_ratio": posterior_ratio,
        "sensorimotor_minus_posterior": sensorimotor_ratio - posterior_ratio,
    }


def plot_scores(scores_df: pd.DataFrame, output_fig: Path) -> None:
    plot_df = scores_df.set_index("model_name")[PLOT_COLUMNS]
    ax = plot_df.plot(kind="bar", figsize=(9, 5), rot=0)
    ax.set_title("Sensorimotor coherence score")
    ax.set_ylabel("Ratio / difference")
    ax.set_xlabel("Model")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.legend(
        [
            "Sensorimotor ratio",
            "Posterior ratio",
            "Sensorimotor - posterior",
        ],
        frameon=False,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()

    output_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_fig, dpi=200)
    plt.close()


def print_interpretation(scores_df: pd.DataFrame) -> None:
    baseline = scores_df.loc[scores_df["model_name"] == "baseline"].iloc[0]
    logic = scores_df.loc[scores_df["model_name"] == "logic"].iloc[0]

    print("\nInterpretation:")
    if logic["sensorimotor_ratio"] > baseline["sensorimotor_ratio"]:
        print(
            "- The logic loss increases the proportion of importance assigned "
            "to sensorimotor channels."
        )
    else:
        print(
            "- The logic loss does not increase the sensorimotor importance "
            "proportion in this run; interpret this cautiously."
        )

    if logic["posterior_ratio"] < baseline["posterior_ratio"]:
        print("- The logic loss reduces dependence on posterior channels.")
    else:
        print(
            "- The logic loss does not reduce the posterior-channel proportion "
            "in this run; interpret this cautiously."
        )


def main() -> None:
    args = parse_args()
    inputs = {
        "baseline": args.baseline_csv,
        "logic": args.logic_csv,
    }

    rows = []
    for model_name, path in inputs.items():
        df = validate_input_csv(path, model_name)
        rows.append(compute_scores(model_name, df))

    scores_df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(args.output_csv, index=False)
    plot_scores(scores_df, args.output_fig)

    print(scores_df.to_string(index=False))
    print_interpretation(scores_df)
    print(f"\nsaved CSV: {args.output_csv}")
    print(f"saved figure: {args.output_fig}")


if __name__ == "__main__":
    main()
