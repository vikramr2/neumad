#!/usr/bin/env python3
"""
eval/plot_cc_overview.py

5 horizontal boxplots — one per system — showing the distribution of
final creativity scores across the 7 questions.
Saves to eval/plots/cc_overview.{png,pdf}
"""

import csv
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def _darken(hex_color: str, factor: float = 0.65) -> tuple:
    r, g, b = mcolors.to_rgb(hex_color)
    return (r * factor, g * factor, b * factor)

ARTIFACTS = Path("../artifacts")

SYSTEMS  = ["synthesis", "adversarial", "choreographed", "neukrag", "neukrag_inter"]
LABELS   = ["Synthesis", "Adversarial", "Choreographed", "NeuKRAG", "NeuKRAG-inter"]
SUFFIXES = {
    "synthesis":     "synthesis",
    "adversarial":   "adversarial_r3_tft2",
    "choreographed": "choreographed",
    "neukrag":       "neukrag",
    "neukrag_inter": "neukrag_inter",
}
COLORS = ["#1f6aa5", "#c0392b", "#2d8a4e", "#8b4513", "#8e44ad"]


def get_final_metric(artifact_dir: Path, metric: str) -> float:
    csv_path = artifact_dir / "cc_scores.csv"
    if not csv_path.exists():
        return 0.0
    with open(csv_path) as f:
        for row in reversed(list(csv.DictReader(f))):
            if row["round"] == "final":
                return float(row[metric])
    return 0.0


def make_boxplot(metric: str, title: str, xlabel: str, filename: str, out_dir: Path):
    data = [
        [get_final_metric(ARTIFACTS / f"ashish_q{q}_{SUFFIXES[s]}", metric) for q in range(1, 8)]
        for s in SYSTEMS
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    positions = list(range(len(SYSTEMS), 0, -1))

    bp = ax.boxplot(
        data,
        positions=positions,
        vert=False,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="black", linewidth=3.5),
        whiskerprops=dict(linewidth=1.4),
        capprops=dict(linewidth=1.4),
        flierprops=dict(marker="o", markersize=5, linestyle="none"),
    )

    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    for flier, color in zip(bp["fliers"], COLORS):
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)

    # Overlay individual points (jittered vertically) so collapsed boxes are visible
    rng = np.random.default_rng(42)
    for si, (vals, pos) in enumerate(zip(data, positions)):
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(vals, [pos + j for j in jitter],
                   color=_darken(COLORS[si]), edgecolors="none",
                   s=22, zorder=5, alpha=0.95)

    ax.set_yticks(positions)
    ax.set_yticklabels(LABELS, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = out_dir / f"{filename}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


out_dir = Path("plots")
out_dir.mkdir(exist_ok=True)

make_boxplot("creativity", "Combinatorial Creativity across Questions (n=7)",
             "Final Creativity Score",  "cc_overview",  out_dir)
make_boxplot("novelty",   "Novelty across Questions (n=7)",
             "Final Novelty Score",    "nov_overview", out_dir)
make_boxplot("utility",   "Utility across Questions (n=7)",
             "Final Utility Score",    "util_overview", out_dir)
