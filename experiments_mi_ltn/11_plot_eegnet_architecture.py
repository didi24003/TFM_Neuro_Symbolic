#!/usr/bin/env python
"""Create a conceptual EEGNet architecture figure for the TFM."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR / "runs"
MPLCONFIG_DIR = RUNS_DIR / ".matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUNS_DIR,
        help="Directory where the PDF and PNG figures will be written.",
    )
    parser.add_argument(
        "--basename",
        default="eegnet_architecture_conceptual",
        help="Output filename without extension.",
    )
    return parser.parse_args()


def draw_block(ax: plt.Axes, x: float, y: float, width: float, height: float, text: str) -> None:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.035",
        linewidth=1.15,
        edgecolor="#333333",
        facecolor="#f4f4f4",
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=11,
        color="#222222",
        linespacing=1.25,
    )


def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.1,
        color="#444444",
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arrow)


def create_figure(output_pdf: Path, output_png: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(10.5, 2.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y = 0.36
    width = 0.18
    height = 0.28
    xs = [0.05, 0.30, 0.55, 0.80]
    labels = [
        "Entrada EEG\nmulticanal",
        "Bloque\ntemporal",
        "Bloque espacial\ndepthwise",
        "Bloque separable\n+ clasificador",
    ]

    for x, label in zip(xs, labels):
        draw_block(ax, x, y, width, height, label)

    center_y = y + height / 2
    for left_x, right_x in zip(xs[:-1], xs[1:]):
        draw_arrow(ax, (left_x + width, center_y), (right_x, center_y))

    ax.text(
        0.5,
        0.83,
        "Esquema conceptual de EEGNet",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color="#222222",
    )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_pdf = args.output_dir / f"{args.basename}.pdf"
    output_png = args.output_dir / f"{args.basename}.png"
    create_figure(output_pdf, output_png)
    print(f"saved PDF: {output_pdf}")
    print(f"saved PNG: {output_png}")


if __name__ == "__main__":
    main()
