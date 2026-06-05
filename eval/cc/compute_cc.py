#!/usr/bin/env python3
"""
eval/cc/compute_cc.py  —  Compute Combinatorial Creativity for a NeuMAD artifact

Usage:
    python compute_cc.py <artifact_folder>
                         [--config ../../config.toml]
                         [--cache-dir cache/]
                         [--alpha-h 0.5] [--alpha-r 0.5] [--alpha-R 1.0]

Output:
    <artifact_folder>/cc_scores.csv
    Columns: round, agent, novelty, utility, creativity

Accumulation rules (per the paper):
    - Individual agents: accumulate their own triples across rounds 1..R
    - Mediator / synthesis rows: accumulate ALL agents' triples across all
      rounds up to and including the current one
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

AGENTS = ["neuroscience", "aiml", "neuromorphic"]

# ---------------------------------------------------------------------------
# Config / metadata helpers
# ---------------------------------------------------------------------------

def load_toml(path: Path) -> dict:
    import tomllib
    with open(path, "rb") as f:
        return tomllib.load(f)


def build_doi_index(cfg: dict) -> dict[str, tuple[int, str]]:
    """
    Returns {doi -> (document_id, domain_kg_jsonl_path_str)}.
    Domain KG path is used later to get the paper's subgraph nodes.
    Searches all three metadata CSVs; DOIs are unique across domains.
    """
    import csv as _csv
    meta_cfg = cfg.get("metadata_paths", {})
    kg_cfg   = cfg.get("kg_paths", {})

    domain_info = [
        ("neuroscience", "neuroscience_metadata", "neuroscience_kg"),
        ("aiml",         "aiml_metadata",         "aiml_kg"),
        ("neuromorphic", "neuromorphic_metadata",  "neuromorphic_kg"),
    ]

    doi_index: dict[str, tuple[int, str]] = {}
    for domain, meta_key, kg_key in domain_info:
        meta_path = Path(meta_cfg.get(meta_key, "")).expanduser()
        kg_jsonl   = Path(kg_cfg.get(kg_key, "")).expanduser() / "merged" / "deduped.jsonl"
        if not meta_path.exists():
            continue
        with open(meta_path, encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                doi = (row.get("doi") or "").strip().lower()
                doc_id = row.get("id") or row.get("doc_id") or ""
                if doi and doc_id:
                    doi_index[doi] = (int(doc_id), str(kg_jsonl))
    return doi_index


def build_doc_node_map(kg_jsonl_path: str) -> dict[int, set[str]]:
    """
    For a given domain KG JSONL, returns {document_id -> set of nodes (h ∪ t)}.
    Cached in memory per path.
    """
    path = Path(kg_jsonl_path)
    if not path.exists():
        return {}
    doc_nodes: dict[int, set[str]] = defaultdict(set)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            doc_id = t.get("document_id")
            if doc_id is None:
                continue
            doc_nodes[int(doc_id)].add(str(t.get("h", "")))
            doc_nodes[int(doc_id)].add(str(t.get("t", "")))
    return dict(doc_nodes)


_doc_node_cache: dict[str, dict[int, set[str]]] = {}

def paper_subgraph_nodes(doi: str, doi_index: dict, doc_node_cache_store: dict) -> set[str]:
    """Return the set of KG nodes belonging to the paper identified by doi."""
    entry = doi_index.get(doi.lower())
    if entry is None:
        return set()
    doc_id, kg_jsonl = entry
    if kg_jsonl not in doc_node_cache_store:
        doc_node_cache_store[kg_jsonl] = build_doc_node_map(kg_jsonl)
    return doc_node_cache_store[kg_jsonl].get(doc_id, set())


# ---------------------------------------------------------------------------
# DOI extraction from reference strings
# ---------------------------------------------------------------------------

_DOI_RE = re.compile(r'\b(10\.\d{4,}/\S+)', re.IGNORECASE)

def extract_dois(references: str) -> list[str]:
    dois = _DOI_RE.findall(references or "")
    # Strip trailing punctuation
    return [d.rstrip(".,;)\"'") for d in dois]


# ---------------------------------------------------------------------------
# Artifact loading helpers
# ---------------------------------------------------------------------------

def load_triples(csv_path: Path) -> list[dict]:
    import csv as _csv
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            rows.append(row)
    return rows


def load_response(json_path: Path) -> dict:
    if not json_path.exists():
        return {}
    return json.loads(json_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Algorithm 2: Compute Utility
# ---------------------------------------------------------------------------

def compute_utility(
    P: set[str],           # nodes touched by fetched triples
    I: set[str],           # inclusionary constraint nodes (may be empty)
    X: set[str],           # exclusionary constraint nodes (may be empty)
    R: list[str],          # DOIs cited in response
    doi_index: dict,
    doc_node_cache: dict,
    alpha_I: float = 0.0,
    alpha_X: float = 0.0,
    alpha_R: float = 1.0,
) -> float:
    # U_C: constraint satisfaction
    if len(I) + len(X) == 0:
        U_C = 1.0
    else:
        satisfied = (I & P) | (X - P)
        sat = len(satisfied) / (len(I) + len(X))
        U_C = sat * (1 + alpha_I * len(I)) * (1 + alpha_X * len(X))

    # U_R: retrieval satisfaction
    if not R:
        U_R = 1.0
    else:
        used = sum(
            1 for doi in R
            if paper_subgraph_nodes(doi, doi_index, doc_node_cache) & P
        )
        U_R = (used / len(R)) * (1 + alpha_R * len(R))

    return U_C * U_R


# ---------------------------------------------------------------------------
# Algorithm 3: Compute Novelty
# ---------------------------------------------------------------------------

def compute_novelty(
    pi: list[str],
    P_dict: dict,
    alpha_h: float = 0.5,
    alpha_r: float = 0.5,
) -> float:
    h = len(pi) - 1
    if h <= 0:
        return 0.0

    S = 0.0
    for i in range(h):
        p = P_dict.get((pi[i], pi[i + 1]), 1e-10)
        S += -math.log(max(p, 1e-10))
    S /= h

    return alpha_h * h + alpha_r * S


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute combinatorial creativity for a NeuMAD artifact")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--config",    type=Path,  default=Path("../../config.toml"))
    parser.add_argument("--cache-dir", type=Path,  default=Path("cache/"))
    parser.add_argument("--alpha-h",   type=float, default=0.5)
    parser.add_argument("--alpha-r",   type=float, default=0.5)
    parser.add_argument("--alpha-R",   type=float, default=1.0)
    args = parser.parse_args()

    artifact_dir  = args.artifact
    responses_dir = artifact_dir / "responses"
    triples_dir   = artifact_dir / "kg_triples"

    if not responses_dir.exists():
        sys.exit(f"No responses/ in {artifact_dir}")

    # Load landscape graph
    sys.path.insert(0, str(Path(__file__).parent))
    from build_landscape import load_kg_all, assemble_creative_landscape

    log.info("Loading kg_all graph …")
    G = load_kg_all(args.config, args.cache_dir)

    # Build DOI lookup and doc-node cache store
    cfg = load_toml(args.config)
    doi_index       = build_doi_index(cfg)
    doc_node_cache  = {}

    # Accumulation state: per-agent triple lists, growing across rounds
    agent_triples:  dict[str, list[dict]] = defaultdict(list)
    agent_refs:     dict[str, list[str]]  = defaultdict(list)

    rows: list[tuple] = []

    # Enumerate all round files, sorted by round number
    # Collect (round_num, agent, triples_path, response_path) entries
    round_entries: list[tuple[int, str, Path | None, Path | None]] = []

    for f in sorted(responses_dir.glob("round_*.json")):
        parts = f.stem.split("_", 2)
        if len(parts) < 3:
            continue
        rnum  = int(parts[1])
        agent = parts[2]
        triples_path  = triples_dir / f"round_{parts[1]}_{agent}.csv"
        round_entries.append((rnum, agent, triples_path, f))

    # Process round entries in order
    for rnum, agent, triples_path, resp_path in sorted(round_entries, key=lambda x: (x[0], x[1])):
        triples  = load_triples(triples_path)
        response = load_response(resp_path)
        refs     = extract_dois(response.get("references", ""))

        if agent != "mediator":
            # Accumulate this agent's own triples and references
            agent_triples[agent].extend(triples)
            agent_refs[agent].extend(refs)

            T_agent = agent_triples[agent]
            R_agent = list(set(agent_refs[agent]))
            P_nodes = set(str(t.get("h", "")) for t in T_agent) | \
                      set(str(t.get("t", "")) for t in T_agent)

            log.info(f"R{rnum} {agent}: {len(T_agent)} triples, {len(R_agent)} cited DOIs")
            P_dict, pi = assemble_creative_landscape(T_agent, G)

            novelty  = compute_novelty(pi, P_dict, args.alpha_h, args.alpha_r)
            utility  = compute_utility(P_nodes, set(), set(), R_agent,
                                       doi_index, doc_node_cache,
                                       alpha_R=args.alpha_R)
            rows.append((rnum, agent, novelty, utility, novelty * utility))

        elif agent == "mediator":
            # Accumulate ALL agents' triples up to and including this round
            T_all = [t for ts in agent_triples.values() for t in ts]
            R_all = list(set(r for rs in agent_refs.values() for r in rs))
            P_nodes = set(str(t.get("h", "")) for t in T_all) | \
                      set(str(t.get("t", "")) for t in T_all)

            log.info(f"R{rnum} mediator: {len(T_all)} accumulated triples")
            P_dict, pi = assemble_creative_landscape(T_all, G)

            novelty  = compute_novelty(pi, P_dict, args.alpha_h, args.alpha_r)
            utility  = compute_utility(P_nodes, set(), set(), R_all,
                                       doi_index, doc_node_cache,
                                       alpha_R=args.alpha_R)
            rows.append((rnum, "mediator", novelty, utility, novelty * utility))

    # Final synthesis
    final_path = responses_dir / "final_synthesis.json"
    if final_path.exists():
        T_all   = [t for ts in agent_triples.values() for t in ts]
        R_all   = list(set(r for rs in agent_refs.values() for r in rs))
        P_nodes = set(str(t.get("h", "")) for t in T_all) | \
                  set(str(t.get("t", "")) for t in T_all)

        log.info(f"final synthesis: {len(T_all)} accumulated triples")
        P_dict, pi = assemble_creative_landscape(T_all, G)

        novelty  = compute_novelty(pi, P_dict, args.alpha_h, args.alpha_r)
        utility  = compute_utility(P_nodes, set(), set(), R_all,
                                   doi_index, doc_node_cache,
                                   alpha_R=args.alpha_R)
        rows.append(("final", "synthesis", novelty, utility, novelty * utility))

    # Write CSV
    out_path = artifact_dir / "cc_scores.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "agent", "novelty", "utility", "creativity"])
        for row in rows:
            writer.writerow([row[0], row[1], f"{row[2]:.4f}", f"{row[3]:.4f}", f"{row[4]:.4f}"])

    log.info(f"Saved: {out_path}")
    print(f"\nround,agent,novelty,utility,creativity")
    for row in rows:
        print(f"{row[0]},{row[1]},{row[2]:.4f},{row[3]:.4f},{row[4]:.4f}")


if __name__ == "__main__":
    main()
