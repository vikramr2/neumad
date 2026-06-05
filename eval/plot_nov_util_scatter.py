#!/usr/bin/env python3
"""
eval/plot_nov_util_scatter.py

Novelty vs. utility scatterplot across all CC results (all systems, all questions,
all rounds/agents).  Points are coloured by agent type.

Saves: eval/plots/nov_util_scatter.{png,pdf}
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ARTIFACTS = Path("../artifacts")

SUFFIXES = [
    "synthesis",
    "adversarial_r3_tft2",
    "choreographed",
    "neukrag",
    "neukrag_inter",
]

# Agent-type colour palette
AGENT_COLORS = {
    "neuroscience": "#2196F3",   # blue
    "neuromorphic":  "#4CAF50",  # green
    "aiml":          "#FF5722",  # orange-red
    "neukrag":       "#8b4513",  # brown
    "neukrag_inter": "#9C27B0",  # purple
    "mediator":      "#607D8B",  # blue-grey  (synthesis rows mapped here too)
}

AGENT_LABELS = {
    "neuroscience": "Neuroscience",
    "neuromorphic":  "Neuromorphic",
    "aiml":          "AI/ML",
    "neukrag":       "NeuKRAG",
    "neukrag_inter": "NeuKRAG-inter",
    "mediator":      "Mediator / Synthesis",
}

AGENT_MARKERS = {
    "neuroscience": "o",
    "neuromorphic":  "s",
    "aiml":          "^",
    "neukrag":       "D",
    "neukrag_inter": "P",
    "mediator":      "*",
}

AGENT_SIZES = {
    "neuroscience": 55,
    "neuromorphic":  55,
    "aiml":          55,
    "neukrag":       55,
    "neukrag_inter": 55,
    "mediator":      110,   # star looks small at 55
}


def normalise_agent(name: str) -> str:
    if name in ("synthesis",):
        return "mediator"
    return name


def load_points() -> dict[str, list[tuple[float, float]]]:
    """Returns {agent_type: [(novelty, utility), ...]}"""
    points: dict[str, list[tuple[float, float]]] = {k: [] for k in AGENT_COLORS}

    for q in range(1, 8):
        for suffix in SUFFIXES:
            csv_path = ARTIFACTS / f"ashish_q{q}_{suffix}" / "cc_scores.csv"
            if not csv_path.exists():
                continue
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    agent = normalise_agent(row["agent"])
                    if agent not in points:
                        continue
                    nov = float(row["novelty"])
                    util = float(row["utility"])
                    points[agent].append((nov, util))

    return points


def main():
    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)

    points = load_points()

    fig, ax = plt.subplots(figsize=(8, 6))

    for agent, pts in points.items():
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.scatter(
            xs, ys,
            color=AGENT_COLORS[agent],
            marker=AGENT_MARKERS[agent],
            s=AGENT_SIZES[agent],
            label=AGENT_LABELS[agent],
            alpha=0.80,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )

    ax.set_xlabel("Novelty", fontsize=12)
    ax.set_ylabel("Utility", fontsize=12)
    ax.set_title("Novelty vs. Utility — all CC results", fontsize=13, fontweight="bold")
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    ax.legend(
        title="Agent",
        title_fontsize=10,
        fontsize=9,
        framealpha=0.9,
        loc="upper left",
    )

    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = out_dir / f"nov_util_scatter.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
