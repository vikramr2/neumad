#!/usr/bin/env python3
"""
Usage: python plot_similarity.py <artifact_folder>

For each debate round with 3 agents in <artifact_folder>/responses/, computes
3x3 pairwise similarity matrices using SPECTER2, SciBERT, SciSpacy, and BM25,
then saves heatmaps and CSVs to <artifact_folder>/plots/.

Output files:
    round{N}_specter2_similarity.{png,pdf,csv}
    round{N}_scibert_similarity.{png,pdf,csv}
    round{N}_scispacy_similarity.{png,pdf,csv}
    round{N}_bm25_similarity.{png,pdf,csv}
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

AGENTS       = ["neuroscience", "aiml", "neuromorphic"]
AGENT_LABELS = ["Neuroscience", "AI/ML", "Neuromorphic"]

# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------

def embed_specter2(texts: list[str]) -> np.ndarray:
    from transformers import AutoTokenizer, AutoModel
    import torch
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base", use_fast=False)
    model     = AutoModel.from_pretrained("allenai/specter2_base")
    model.eval()
    vecs = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=512, padding=True)
            out = model(**inputs)
            # CLS token
            vecs.append(out.last_hidden_state[:, 0, :].squeeze().numpy())
    return np.array(vecs)


def embed_scibert(texts: list[str]) -> np.ndarray:
    from transformers import AutoTokenizer, AutoModel
    import torch
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    model     = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
    model.eval()
    vecs = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=512, padding=True)
            out = model(**inputs)
            vecs.append(out.last_hidden_state[:, 0, :].squeeze().numpy())
    return np.array(vecs)


def embed_scispacy(texts: list[str]) -> np.ndarray:
    import spacy
    nlp = spacy.load("en_core_sci_lg")
    return np.array([nlp(t).vector for t in texts])


def bm25_similarity(texts: list[str]) -> np.ndarray:
    from rank_bm25 import BM25Okapi
    tokenized = [t.lower().split() for t in texts]
    n = len(texts)
    mat = np.zeros((n, n))
    for i in range(n):
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(tokenized[i])
        mat[i] = scores
    # Make symmetric and normalise to [0, 1]
    mat = (mat + mat.T) / 2
    dmax = mat.max()
    if dmax > 0:
        mat /= dmax
    np.fill_diagonal(mat, 1.0)
    return mat


EMBEDDERS = {
    "specter2": embed_specter2,
    "scibert":  embed_scibert,
    "scispacy": embed_scispacy,
}

# ---------------------------------------------------------------------------
# Plot + save helpers
# ---------------------------------------------------------------------------

def compute_cosine(vecs: np.ndarray) -> np.ndarray:
    return cosine_similarity(vecs)


_RYG = mcolors.LinearSegmentedColormap.from_list(
    "ryg", ["#d32f2f", "#f9a825", "#388e3c"]
)


def save_heatmap(matrix: np.ndarray, labels: list[str],
                 title: str, out_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    im = ax.imshow(matrix, cmap=_RYG, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="black" if matrix[i, j] < 0.7 else "white")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_csv(matrix: np.ndarray, labels: list[str], out_stem: Path) -> None:
    with open(f"{out_stem}.csv", "w") as f:
        f.write("," + ",".join(labels) + "\n")
        for i, label in enumerate(labels):
            row = ",".join(f"{matrix[i, j]:.4f}" for j in range(len(labels)))
            f.write(f"{label},{row}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python plot_similarity.py <artifact_folder>")

    artifact_dir  = Path(sys.argv[1])
    responses_dir = artifact_dir / "responses"
    if not responses_dir.exists():
        sys.exit(f"No responses/ folder in {artifact_dir}")

    out_dir = artifact_dir / "plots"
    out_dir.mkdir(exist_ok=True)

    # Collect rounds that have all 3 agents
    round_statements: dict[int, dict[str, str]] = {}
    for f in sorted(responses_dir.glob("round_*.json")):
        parts = f.stem.split("_", 2)
        if len(parts) < 3:
            continue
        agent = parts[2]
        if agent not in AGENTS:
            continue
        rnum = int(parts[1])
        record = json.loads(f.read_text())
        round_statements.setdefault(rnum, {})[agent] = record.get("statement", "")

    complete_rounds = {r: s for r, s in round_statements.items()
                       if all(a in s for a in AGENTS)}

    if not complete_rounds:
        sys.exit("No complete agent rounds found.")

    print(f"Found {len(complete_rounds)} complete round(s): {sorted(complete_rounds)}")

    for rnum in sorted(complete_rounds):
        texts  = [complete_rounds[rnum][a] for a in AGENTS]
        prefix = f"round{rnum:02d}"

        # --- Embedding-based methods ---
        for name, embed_fn in EMBEDDERS.items():
            print(f"  R{rnum} {name}...", end=" ", flush=True)
            vecs   = embed_fn(texts)
            matrix = compute_cosine(vecs)
            stem   = out_dir / f"{prefix}_{name}_similarity"
            save_heatmap(matrix, AGENT_LABELS,
                         f"R{rnum} — {name} cosine similarity", stem)
            save_csv(matrix, AGENT_LABELS, stem)
            print("done")

        # --- BM25 ---
        print(f"  R{rnum} bm25...", end=" ", flush=True)
        matrix = bm25_similarity(texts)
        stem   = out_dir / f"{prefix}_bm25_similarity"
        save_heatmap(matrix, AGENT_LABELS,
                     f"R{rnum} — BM25 similarity", stem)
        save_csv(matrix, AGENT_LABELS, stem)
        print("done")

    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
