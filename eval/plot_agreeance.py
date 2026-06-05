#!/usr/bin/env python3
"""
Usage: python plot_agreeance.py <artifact_folder>

Reads round_XX_{agent}.json files from <artifact_folder>/responses/,
skips mediator rounds, and saves a 3 × num_rounds agreeance heatmap
to <artifact_folder>/plots/agreeance_matrix.png.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

AGENTS = ["neuroscience", "aiml", "neuromorphic"]
AGENT_LABELS = ["Neuroscience", "AI/ML", "Neuromorphic"]

# agreed value → numeric: None=0.5 (yellow), False=0 (red), True=1 (green)
def agreed_to_val(agreed):
    if agreed is True:
        return 1.0
    if agreed is False:
        return 0.0
    return 0.5  # None


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python plot_agreeance.py <artifact_folder>")

    artifact_dir = Path(sys.argv[1])
    responses_dir = artifact_dir / "responses"
    if not responses_dir.exists():
        sys.exit(f"No responses/ folder found in {artifact_dir}")

    # Collect agent files; mediator rounds are marked separately
    round_data: dict[int, dict[str, float]] = {}
    mediator_rounds: set[int] = set()
    for f in sorted(responses_dir.glob("round_*.json")):
        name = f.stem  # e.g. "round_02_aiml" or "round_04_mediator"
        parts = name.split("_", 2)
        if len(parts) < 3:
            continue
        agent = parts[2]
        round_num = int(parts[1])
        if agent == "mediator":
            mediator_rounds.add(round_num)
            # All three agent rows shown as yellow for mediator rounds
            round_data.setdefault(round_num, {})
            for a in AGENTS:
                round_data[round_num][a] = 0.5
        elif agent in AGENTS:
            record = json.loads(f.read_text())
            round_data.setdefault(round_num, {})[agent] = agreed_to_val(record.get("agreed"))

    if not round_data:
        sys.exit("No agent round files found.")

    sorted_rounds = sorted(round_data)
    num_rounds = len(sorted_rounds)

    # Build 3 × num_rounds matrix
    matrix = np.full((3, num_rounds), 0.5)
    for col, rnum in enumerate(sorted_rounds):
        for row, agent in enumerate(AGENTS):
            if agent in round_data[rnum]:
                matrix[row, col] = round_data[rnum][agent]

    # Colormap: 0=red, 0.5=yellow, 1=green
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rygcmap", ["#d32f2f", "#f9a825", "#388e3c"]
    )

    fig, ax = plt.subplots(figsize=(max(4, num_rounds * 1.1), 3.2))
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(num_rounds))
    ax.set_xticklabels(
        [f"R{r}\n(med)" if r in mediator_rounds else f"R{r}" for r in sorted_rounds],
        fontsize=10,
    )
    ax.set_yticks(range(3))
    ax.set_yticklabels(AGENT_LABELS, fontsize=10)
    ax.set_xlabel("Round", fontsize=11)
    ax.set_title("Agent Agreement per Round", fontsize=12, fontweight="bold")

    # Annotate cells
    labels = {0.0: "✗", 0.5: "–", 1.0: "✓"}
    for row in range(3):
        for col in range(num_rounds):
            val = matrix[row, col]
            text_color = "white" if val != 0.5 else "#333333"
            ax.text(col, row, labels.get(val, ""), ha="center", va="center",
                    fontsize=13, color=text_color, fontweight="bold")

    # Legend
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, color="#d32f2f", label="Disagree"),
        plt.Rectangle((0, 0), 1, 1, color="#f9a825", label="No judgement"),
        plt.Rectangle((0, 0), 1, 1, color="#388e3c", label="Agree"),
    ]
    ax.legend(handles=legend_patches, loc="upper center",
              bbox_to_anchor=(0.5, -0.30), ncol=3, fontsize=9, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)

    out_dir = artifact_dir / "plots"
    out_dir.mkdir(exist_ok=True)

    for ext in ("png", "pdf"):
        out_path = out_dir / f"agreeance_matrix.{ext}"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")

    csv_path = out_dir / "agreeance_matrix.csv"
    val_to_label = {0.0: "disagree", 0.5: "none", 1.0: "agree"}
    with csv_path.open("w") as f:
        f.write("agent," + ",".join(f"R{r}" for r in sorted_rounds) + "\n")
        for row, agent in enumerate(AGENT_LABELS):
            vals = ",".join(val_to_label[matrix[row, col]] for col in range(num_rounds))
            f.write(f"{agent},{vals}\n")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
