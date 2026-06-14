#!/usr/bin/env python
"""Build final comparison, coherence metrics, LaTeX table, and JSON summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mi_ltn_common import BCI_IV_2A_CHANNELS


SENSORIMOTOR_CHANNELS = ["FC3", "FC4", "FCz", "C3", "C4", "Cz", "CP3", "CP4", "CPz"]
POSTERIOR_CHANNELS = ["P1", "Pz", "P2", "POz"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--logic-csv", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--logic-summary", type=Path, required=True)
    parser.add_argument("--comparison-csv", type=Path, required=True)
    parser.add_argument("--delta-csv", type=Path, required=True)
    parser.add_argument("--delta-png", type=Path, required=True)
    parser.add_argument("--coherence-csv", type=Path, required=True)
    parser.add_argument("--coherence-png", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--latex-table", type=Path, required=True)
    return parser.parse_args()


def load_importance_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} CSV not found at {path.resolve()}")
    df = pd.read_csv(path).copy()
    required = {"channel_index", "channel_name", "importance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{label} CSV is missing required columns: {', '.join(sorted(missing))}"
        )
    df["channel_index"] = pd.to_numeric(df["channel_index"], errors="raise").astype(int)
    df["channel_name"] = df["channel_name"].astype(str)
    df["importance"] = pd.to_numeric(df["importance"], errors="raise")
    return df


def load_summary(path: Path, label: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{label} summary not found at {path.resolve()}")
    return json.loads(path.read_text())


def build_delta_table(baseline_df: pd.DataFrame, logic_df: pd.DataFrame) -> pd.DataFrame:
    baseline = baseline_df.rename(columns={"importance": "baseline_importance"})
    logic = logic_df.rename(columns={"importance": "logic_importance"})
    merged = baseline.merge(
        logic,
        on=["channel_index", "channel_name"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("No common channels found between baseline and logic CSVs.")

    merged["delta_logic_minus_baseline"] = (
        merged["logic_importance"] - merged["baseline_importance"]
    )
    merged["abs_delta"] = merged["delta_logic_minus_baseline"].abs()
    merged["channel"] = merged["channel_name"]

    order = {name: idx for idx, name in enumerate(BCI_IV_2A_CHANNELS)}
    merged["sort_order"] = merged["channel_name"].map(order).fillna(10_000)
    merged = merged.sort_values(["sort_order", "channel_index"]).reset_index(drop=True)
    return merged[
        [
            "channel",
            "baseline_importance",
            "logic_importance",
            "delta_logic_minus_baseline",
            "abs_delta",
        ]
    ]


def ratio(df: pd.DataFrame, channel_group: list[str]) -> float:
    total = float(df["importance"].sum())
    if total == 0.0:
        return 0.0
    subset = float(df.loc[df["channel_name"].isin(channel_group), "importance"].sum())
    return subset / total


def build_coherence_table(
    baseline_df: pd.DataFrame,
    logic_df: pd.DataFrame,
    baseline_summary: dict,
    logic_summary: dict,
) -> pd.DataFrame:
    baseline_sensorimotor = ratio(baseline_df, SENSORIMOTOR_CHANNELS)
    logic_sensorimotor = ratio(logic_df, SENSORIMOTOR_CHANNELS)
    baseline_posterior = ratio(baseline_df, POSTERIOR_CHANNELS)
    logic_posterior = ratio(logic_df, POSTERIOR_CHANNELS)

    rows = [
        {
            "model": "EEGNet baseline",
            "best_val_acc": baseline_summary["best_val_acc"],
            "sensorimotor_ratio": baseline_sensorimotor,
            "posterior_ratio": baseline_posterior,
        },
        {
            "model": "EEGNet + logic loss (lambda_logic=1.0)",
            "best_val_acc": logic_summary["best_val_acc"],
            "sensorimotor_ratio": logic_sensorimotor,
            "posterior_ratio": logic_posterior,
        },
        {
            "model": "Difference (logic - baseline)",
            "best_val_acc": logic_summary["best_val_acc"] - baseline_summary["best_val_acc"],
            "sensorimotor_ratio": logic_sensorimotor - baseline_sensorimotor,
            "posterior_ratio": logic_posterior - baseline_posterior,
        },
    ]
    return pd.DataFrame(rows)


def plot_delta(delta_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = delta_df.copy()
    colors = [
        "#54a24b" if value >= 0 else "#e45756"
        for value in plot_df["delta_logic_minus_baseline"]
    ]
    fig_height = max(6.0, len(plot_df) * 0.35)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.barh(plot_df["channel"], plot_df["delta_logic_minus_baseline"], color=colors)
    ax.axvline(0.0, color="#222222", linewidth=1.0)
    ax.invert_yaxis()
    ax.set_xlabel("Delta importance (logic - baseline)")
    ax.set_ylabel("Channel")
    ax.set_title("Delta de importancia por canal")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_coherence(coherence_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = coherence_df.iloc[:2].copy()
    x = range(len(plot_df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        [pos - width / 2 for pos in x],
        plot_df["sensorimotor_ratio"],
        width=width,
        color="#4c78a8",
        label="sensorimotor_ratio",
    )
    ax.bar(
        [pos + width / 2 for pos in x],
        plot_df["posterior_ratio"],
        width=width,
        color="#f58518",
        label="posterior_ratio",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["model"], rotation=0)
    ax.set_ylabel("Ratio")
    ax.set_title("Coherence scores")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_latex_table(coherence_df: pd.DataFrame, output_path: Path) -> None:
    table_df = coherence_df.iloc[:2].copy()
    model_labels = [
        "EEGNet baseline",
        r"EEGNet + logic loss ($\lambda_{logic}=1.0$)",
    ]
    lines = [
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Modelo & Best val acc & Sensorimotor ratio & Posterior ratio \\",
        r"\hline",
    ]
    for label, (_, row) in zip(model_labels, table_df.iterrows()):
        lines.append(
            f"{label} & {row['best_val_acc']:.4f} & "
            f"{row['sensorimotor_ratio']:.4f} & {row['posterior_ratio']:.4f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def write_json_summary(
    baseline_summary: dict,
    logic_summary: dict,
    coherence_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    output_path: Path,
) -> None:
    top_positive = delta_df.sort_values(
        "delta_logic_minus_baseline", ascending=False
    ).head(5)
    top_negative = delta_df.sort_values(
        "delta_logic_minus_baseline", ascending=True
    ).head(5)

    data = {
        "baseline": {
            "run_dir": baseline_summary.get("run_dir"),
            "best_val_acc": baseline_summary.get("best_val_acc"),
            "best_epoch": baseline_summary.get("best_epoch"),
            "checkpoint_best_path": baseline_summary.get("checkpoint_best_path"),
        },
        "logic": {
            "run_dir": logic_summary.get("run_dir"),
            "lambda_logic": logic_summary.get("lambda_logic"),
            "best_val_acc": logic_summary.get("best_val_acc"),
            "best_epoch": logic_summary.get("best_epoch"),
            "checkpoint_best_path": logic_summary.get("checkpoint_best_path"),
        },
        "coherence": coherence_df.to_dict(orient="records"),
        "top_delta_increase": top_positive.to_dict(orient="records"),
        "top_delta_decrease": top_negative.to_dict(orient="records"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    baseline_df = load_importance_csv(args.baseline_csv, "Baseline")
    logic_df = load_importance_csv(args.logic_csv, "Logic")
    baseline_summary = load_summary(args.baseline_summary, "Baseline")
    logic_summary = load_summary(args.logic_summary, "Logic")

    comparison_df = baseline_df.rename(columns={"importance": "importance_baseline"}).merge(
        logic_df.rename(columns={"importance": "importance_logic"}),
        on=["channel_index", "channel_name"],
        how="inner",
        validate="one_to_one",
    )
    comparison_df["delta_importance"] = (
        comparison_df["importance_logic"] - comparison_df["importance_baseline"]
    )
    comparison_df = comparison_df[
        [
            "channel_index",
            "channel_name",
            "importance_baseline",
            "importance_logic",
            "delta_importance",
        ]
    ]

    delta_df = build_delta_table(baseline_df, logic_df)
    coherence_df = build_coherence_table(
        baseline_df, logic_df, baseline_summary, logic_summary
    )

    args.comparison_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(args.comparison_csv, index=False)
    delta_df.to_csv(args.delta_csv, index=False)
    coherence_df.to_csv(args.coherence_csv, index=False)
    plot_delta(delta_df, args.delta_png)
    plot_coherence(coherence_df, args.coherence_png)
    write_latex_table(coherence_df, args.latex_table)
    write_json_summary(
        baseline_summary,
        logic_summary,
        coherence_df,
        delta_df,
        args.summary_json,
    )

    print(f"saved comparison CSV: {args.comparison_csv}")
    print(f"saved delta CSV: {args.delta_csv}")
    print(f"saved delta PNG: {args.delta_png}")
    print(f"saved coherence CSV: {args.coherence_csv}")
    print(f"saved coherence PNG: {args.coherence_png}")
    print(f"saved LaTeX table: {args.latex_table}")
    print(f"saved JSON summary: {args.summary_json}")


if __name__ == "__main__":
    main()
