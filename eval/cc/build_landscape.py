#!/usr/bin/env python3
"""
eval/cc/build_landscape.py  —  Assemble the Creative Landscape (Algorithm 1)

CLI usage:
    python build_landscape.py [--config ../../config.toml] [--cache-dir cache/]

This script loads kg_all from config.toml, builds the full graph, and caches it
to disk so that compute_cc.py can run quickly.

It also exports:
    load_kg_all(config_path, cache_dir)  -> nx.DiGraph
    assemble_creative_landscape(triples, G) -> (P_dict, pi)

where:
    triples  — list of {"h", "r", "t", "document_id"} dicts
    G        — full kg_all networkx DiGraph
    P_dict   — dict[(i,j)] -> float, row-normalised transition probabilities
    pi       — list of node names forming the assembled walk
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
from collections import defaultdict
from pathlib import Path

import networkx as nx

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graph loading + caching
# ---------------------------------------------------------------------------

def _load_jsonl_graph(path: Path) -> nx.DiGraph:
    G = nx.DiGraph()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            h, r, tail = str(t.get("h", "")), t.get("r", ""), str(t.get("t", ""))
            doc_id = t.get("document_id")
            if h and tail:
                G.add_edge(h, tail, relation=r, document_id=doc_id)
    return G


def load_kg_all(config_path: Path, cache_dir: Path) -> nx.DiGraph:
    """Load kg_all graph, using a pickle cache for speed on subsequent runs."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "kg_all_graph.pkl"

    if cache_file.exists():
        log.info(f"Loading cached kg_all graph from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    import tomllib
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    kg_all_path = Path(cfg["kg_paths"]["all_kg"]).expanduser() / "merged" / "deduped.jsonl"
    if not kg_all_path.exists():
        raise FileNotFoundError(f"kg_all not found: {kg_all_path}")

    log.info(f"Building kg_all graph from {kg_all_path} ...")
    G = _load_jsonl_graph(kg_all_path)
    log.info(f"  {len(G.nodes)} nodes, {len(G.edges)} edges")

    with open(cache_file, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info(f"  Cached to {cache_file}")
    return G


# ---------------------------------------------------------------------------
# Adamic-Adar
# ---------------------------------------------------------------------------

def _undirected_neighbors(G: nx.DiGraph, u: str) -> set:
    return set(G.predecessors(u)) | set(G.successors(u))


def _adamic_adar(G: nx.DiGraph, i: str, j: str,
                 neighbor_cache: dict[str, set]) -> float:
    Ni = neighbor_cache.get(i) or _undirected_neighbors(G, i)
    Nj = neighbor_cache.get(j) or _undirected_neighbors(G, j)
    common = Ni & Nj
    score = 0.0
    for u in common:
        Nu = neighbor_cache.get(u) or _undirected_neighbors(G, u)
        deg = len(Nu)
        if deg > 1:
            score += 1.0 / math.log(deg)
    return score


# ---------------------------------------------------------------------------
# Algorithm 1: Assemble Creative Landscape
# ---------------------------------------------------------------------------

def assemble_creative_landscape(
    triples: list[dict],
    G: nx.DiGraph,
) -> tuple[dict, list[str]]:
    """
    Implements Algorithm 1 from the paper.

    Returns:
        P_dict  — {(i, j): float} row-normalised transition probability for i,j in V
        pi      — ordered list of node names (the assembled walk)
    """
    # Step 1: V = union of all h,t nodes
    V = []
    seen = set()
    for t in triples:
        for node in (str(t.get("h", "")), str(t.get("t", ""))):
            if node and node not in seen:
                V.append(node)
                seen.add(node)

    if len(V) == 0:
        return {}, []
    if len(V) == 1:
        return {(V[0], V[0]): 1.0}, V

    # Pre-build neighbor cache for all nodes in V and their neighbors
    neighbor_cache: dict[str, set] = {}
    for node in V:
        neighbor_cache[node] = _undirected_neighbors(G, node)

    edge_set = set(G.edges())

    # Step 2–7: Build weight matrix W over V×V
    W: dict[tuple, float] = {}
    for i in V:
        for j in V:
            if i == j:
                continue
            if (i, j) in edge_set:
                W[(i, j)] = 1.0
            else:
                aa = _adamic_adar(G, i, j, neighbor_cache)
                W[(i, j)] = aa if aa > 0 else 1e-10  # avoid zero weights

    # Step 9: Row-normalise → transition matrix P
    P_dict: dict[tuple, float] = {}
    for i in V:
        total = sum(W.get((i, j), 0.0) for j in V if j != i)
        for j in V:
            if j == i:
                continue
            P_dict[(i, j)] = W.get((i, j), 0.0) / total if total > 0 else 1e-10

    # Step 10: Blossom min-cost perfect matching over V
    # cost(i,j) = -log(P_ij)  →  min-cost = max-likelihood
    pi = _chain_matching(V, P_dict)

    return P_dict, pi


def _chain_matching(V: list[str], P_dict: dict) -> list[str]:
    """
    Build min-cost perfect matching on V and chain pairs into a walk π.
    Handles odd |V| by duplicating the last node.
    """
    nodes = list(V)
    dummy = None
    if len(nodes) % 2 == 1:
        dummy = "__dummy__"
        nodes.append(dummy)

    # Build complete graph with edge cost = -log(P_ij)
    MG = nx.Graph()
    MG.add_nodes_from(nodes)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            if u == dummy or v == dummy:
                cost = 0.0
            else:
                p_ij = P_dict.get((u, v), 1e-10)
                p_ji = P_dict.get((v, u), 1e-10)
                p = (p_ij + p_ji) / 2  # symmetrise
                cost = -math.log(max(p, 1e-10))
            MG.add_edge(u, v, weight=cost)

    matching = nx.min_weight_matching(MG, weight="weight")

    # Step 11: ChainMatching — concatenate pairs into a single walk
    pairs = [
        (u, v) if u != dummy and v != dummy else (v, u)
        for u, v in matching
        if not (u == dummy and v == dummy)
    ]
    # Greedy chain: connect pairs end-to-end by highest P[tail → next_head]
    if not pairs:
        return [n for n in V if n != dummy]

    remaining = list(pairs)
    chain_a, chain_b = remaining.pop(0)
    walk = [chain_a, chain_b] if chain_a != dummy else [chain_b]

    while remaining:
        tail = walk[-1]
        # Pick the pair whose start (or end) connects best to tail
        best_idx, best_score, best_flip = 0, -1.0, False
        for idx, (a, b) in enumerate(remaining):
            s_ab = P_dict.get((tail, a), 1e-10)
            s_ba = P_dict.get((tail, b), 1e-10)
            if s_ab >= s_ba and s_ab > best_score:
                best_score, best_idx, best_flip = s_ab, idx, False
            elif s_ba > best_score:
                best_score, best_idx, best_flip = s_ba, idx, True
        a, b = remaining.pop(best_idx)
        if best_flip:
            a, b = b, a
        if a != dummy:
            walk.append(a)
        if b != dummy:
            walk.append(b)

    return walk


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build and cache kg_all creative landscape graph")
    parser.add_argument("--config",    type=Path, default=Path("../../config.toml"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/"))
    args = parser.parse_args()

    G = load_kg_all(args.config, args.cache_dir)
    log.info(f"kg_all graph ready: {len(G.nodes)} nodes, {len(G.edges)} edges")


if __name__ == "__main__":
    main()
