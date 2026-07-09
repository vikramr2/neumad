#!/usr/bin/env python3
"""
orchestration.py  —  NeuKRAG Multi-Agent Debate orchestrator

Wires three KG-grounded specialist agents into either a cooperative synthesis
or a MAD-style adversarial debate, with a baseline LLM mediator.

Runtime configuration is read from mad/environment.json (defaults) and may
be overridden by shell environment variables of the same name:
    NEUKRAG_MODE          synthesis | adversarial | choreographed
    NEUKRAG_DEBATE_ROUNDS max debate rounds (adversarial only)
    NEUKRAG_DEBATE_LEVEL  0-3 tit-for-tat intensity (adversarial only)

Usage (via run.sh or directly):
    python orchestration.py "<query>"
    python orchestration.py              # runs default paper queries
    python orchestration.py "<query>" --out results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import dspy

# ---------------------------------------------------------------------------
# Path setup — allow imports from neukrag/ and mad/
# ---------------------------------------------------------------------------

_ROOT     = Path(__file__).parent.parent          # /home/vr9/neumad/
_NEUKRAG  = _ROOT / "neukrag"                     # contains run_neukrag.py
_MAD      = Path(__file__).parent                 # mad/ (agents/ lives here)
_ARGORA   = _ROOT / "argora-public"               # ARGORA graph utilities

sys.path.insert(0, str(_NEUKRAG))
sys.path.insert(0, str(_MAD))
sys.path.insert(0, str(_ARGORA))

from run_neukrag import (                          # noqa: E402
    DEFAULT_QUERIES,
    LLM_MODEL,
    OLLAMA_API_KEY,
    OLLAMA_BASE_URL,
    EntityExtractor,
    HypothesisGenerator,
    format_context,
    keyword_entry_points,
    bfs_subgraph,
    load_graph,
)
from agents.neuroscience import LABEL as NEURO_LABEL
from agents.aiml         import LABEL as AIML_LABEL
from agents.neuromorphic import LABEL as NEURO_M_LABEL
from agents.specialist   import SpecialistAgent, format_apa
from agents.mediator     import Mediator

# ---------------------------------------------------------------------------
# Chamber imports — run_* functions live in their own modules.
# Placed here (after all classes/helpers are defined) so the chambers can
# import back from this module without triggering a circular-import error.
# ---------------------------------------------------------------------------

from chambers.synthesis     import run_synthesis      # noqa: E402
from chambers.adversarial   import run_adversarial    # noqa: E402
from chambers.rotation      import run_rotation       # noqa: E402
from chambers.choreographed import (                  # noqa: E402
    CHOREOGRAPHED_ROUND_LABELS,
    CHOREOGRAPHED_COVARIANCE,
    run_choreographed,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ENV_PATH    = _MAD / "environment.json"
CONFIG_PATH = _ROOT / "config.toml"

# From MAD paper Table 11 — tit-for-tat levels
DEBATE_LEVEL_PROMPTS = {
    0: "All agents must reach full consensus on every point. Every claim must be agreed upon by all sides.",
    1: "Most points should show disagreement, but minor consensus on peripheral details is acceptable.",
    2: "It's not necessary to fully agree with each other's perspectives; the objective is to find the correct answer.",
    3: "All agents must disagree on every point. There should be no consensus whatsoever.",
}

_AGENT_LABELS = {
    "neuroscience": NEURO_LABEL,
    "aiml":         AIML_LABEL,
    "neuromorphic": NEURO_M_LABEL,
}


# ---------------------------------------------------------------------------
# Orchestration helpers
# ---------------------------------------------------------------------------

def format_debate_history(history: list[dict]) -> str:
    lines = []
    for entry in history:
        label  = _AGENT_LABELS.get(entry["agent"], entry["agent"].upper())
        prefix = "Initial Position" if entry["round"] == 0 else f"Round {entry['round']} Response"
        lines.append(f"[{label} — {prefix}]:\n{entry['statement']}\n")
    return "\n".join(lines)


def annotate_transitions(
    history: list[dict],
    round_num: int,
    query: str,
    agents: list,
    mediator,
    position_history: dict[str, str],
) -> None:
    """Classify each agent's round_num statement against its own and peers' prior
    main claims, writing 'transition_type' and 'adopted_peer' onto the round's
    history entries in place. Updates position_history with this round's claims.

    position_history must already hold every agent's claim from the round being
    compared against (seeded from round 0/1's build_local_arguments main_claim).
    """
    agent_by_name = {a.name: a for a in agents}
    round_entries = {
        e["agent"]: e for e in history
        if e["round"] == round_num and e["agent"] in agent_by_name
    }

    new_claims = {
        name: agent_by_name[name].extract_main_claim(query, entry["statement"])
        for name, entry in round_entries.items()
    }

    for name, entry in round_entries.items():
        own_prior = position_history.get(name)
        peer_claims = {
            p: claim for p, claim in position_history.items()
            if p != name and p in agent_by_name
        }
        if own_prior is None:
            entry["transition_type"] = None
            entry["adopted_peer"]    = None
            continue

        match = mediator.classify_transition(query, name, new_claims[name], own_prior, peer_claims)
        if match == "own":
            entry["transition_type"], entry["adopted_peer"] = "unchanged", None
        elif match == "novel":
            entry["transition_type"], entry["adopted_peer"] = "independent_revision", None
        else:
            entry["transition_type"], entry["adopted_peer"] = "peer_aligned", match

    position_history.update(new_claims)


def get_references_from_triples(triples: list[dict], metadata: dict[int, dict]) -> str:
    """APA citations for the top-5 most-cited source papers in a triple set."""
    from collections import Counter
    counts = Counter(t["document_id"] for t in triples if t.get("document_id") is not None)
    refs = []
    for doc_id, _ in counts.most_common(5):
        paper = metadata.get(doc_id)
        if paper:
            refs.append(format_apa(paper))
    return "\n".join(refs)


def run_neukrag_single(
    query: str,
    graph,
    metadata: dict[int, dict],
    entity_extractor: EntityExtractor,
    hyp_generator: HypothesisGenerator,
    k_hops: int = 2,
    max_triples: int = 40,
    mode: str = "neukrag",
    status_cb=None,
) -> dict:
    def _status(msg: str):
        log.info(msg)
        if status_cb:
            status_cb(msg)

    _status(f"Query ({mode}): {query}")
    _status("  Extracting entities…")
    entities = entity_extractor(query=query)
    _status(f"  Entities: {entities}")

    entry_nodes = keyword_entry_points(graph, entities)
    if not entry_nodes:
        log.warning("  No entry nodes — using entity names as fallback")
        entry_nodes = set(entities)
    _status(f"  Entry nodes: {len(entry_nodes)}")

    triples = bfs_subgraph(graph, entry_nodes, k_hops=k_hops, max_triples=max_triples)
    _status(f"  Subgraph: {len(triples)} triples")

    _status("  Generating hypothesis…")
    hypothesis = hyp_generator(query=query, graph_context=format_context(triples))
    refs = get_references_from_triples(triples, metadata)

    return {
        "query":            query,
        "mode":             mode,
        "triples":          triples,
        "references":       refs,
        "final_hypothesis": hypothesis,
    }


# TODO: follow ups should also be multi-agent
def run_followup(
    question: str,
    previous_synthesis: str,
    mediator: Mediator,
    status_cb=None,
) -> dict:
    if status_cb:
        status_cb("Mediator answering follow-up…")
    answer = mediator.answer_followup(previous_synthesis, question)
    return {
        "query":            question,
        "mode":             "followup",
        "final_hypothesis": answer,
    }


# ---------------------------------------------------------------------------
# Config / env helpers
# ---------------------------------------------------------------------------

def load_metadata(csv_path: Path) -> dict[int, dict]:
    """Load a paper-metadata CSV into a dict keyed by integer document id."""
    import csv as _csv
    metadata: dict[int, dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            try:
                doc_id = int(float(row["id"]))
                metadata[doc_id] = dict(row)
            except (KeyError, ValueError):
                continue
    return metadata


def load_toml(config_path: Path) -> dict:
    import tomllib
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def load_env(env_path: Path) -> dict:
    """Load environment.json and apply as defaults — os.environ takes precedence."""
    with open(env_path) as f:
        defaults = json.load(f)
    merged = {}
    for k, v in defaults.items():
        merged[k] = os.environ.get(k, str(v))
    return merged

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NeuKRAG Multi-Agent Debate")
    parser.add_argument("query", nargs="?", default=None,
                        help="Research query (omit to run default paper queries)")
    parser.add_argument("--k-hops",      type=int,  default=2,
                        help="BFS depth per agent (default: 2)")
    parser.add_argument("--max-triples", type=int,  default=40,
                        help="Max KG triples per agent context (default: 40)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Save results as JSON to this path")
    args = parser.parse_args()

    env = load_env(ENV_PATH)

    mode = env["NEUKRAG_MODE"].lower()
    valid_modes = ("synthesis", "adversarial", "choreographed", "rotation", "neukrag", "neukrag-inter")
    if mode not in valid_modes:
        sys.exit(f"ERROR: NEUKRAG_MODE must be one of {valid_modes}, got '{mode}'")

    max_rounds   = int(env["NEUKRAG_DEBATE_ROUNDS"])
    debate_level = int(env["NEUKRAG_DEBATE_LEVEL"])
    if debate_level not in DEBATE_LEVEL_PROMPTS:
        sys.exit(f"ERROR: NEUKRAG_DEBATE_LEVEL must be 0-3, got {debate_level}")

    neuromorphic_mediator = env["NEUKRAG_NEUROMORPHIC_MEDIATOR"].lower() in ("1", "true", "yes")

    n_rotations = int(env["NEUKRAG_ROTATIONS"])
    if n_rotations < 1:
        sys.exit(f"ERROR: NEUKRAG_ROTATIONS must be >= 1, got {n_rotations}")

    cfg      = load_toml(CONFIG_PATH)
    kg_cfg   = cfg.get("kg_paths", {})
    meta_cfg = cfg.get("metadata_paths", {})

    kg_paths = {
        "neuroscience": Path(kg_cfg.get("neuroscience_kg", "")).expanduser(),
        "aiml":         Path(kg_cfg.get("aiml_kg", "")).expanduser(),
        "neuromorphic": Path(kg_cfg.get("neuromorphic_kg", "")).expanduser(),
        "all":          Path(kg_cfg.get("all_kg", "")).expanduser(),
    }
    # Only validate paths that the chosen mode will actually use
    required = {"neukrag": ["neuromorphic"], "neukrag-inter": ["all"]}.get(
        mode, ["neuroscience", "aiml", "neuromorphic"]
    )
    for name in required:
        if not kg_paths[name].exists():
            sys.exit(f"ERROR: KG path for '{name}' not found: {kg_paths[name]}")

    metadata = {
        name: load_metadata(Path(meta_cfg[f"{name}_metadata"]).expanduser())
        for name in ("neuroscience", "aiml", "neuromorphic")
        if meta_cfg.get(f"{name}_metadata")
    }

    lm = dspy.LM(LLM_MODEL, api_base=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
    dspy.configure(lm=lm)
    log.info(f"DSPy configured with {LLM_MODEL} @ {OLLAMA_BASE_URL}")
    log.info(f"Mode: {mode}" + (
        f"  (debate_level={debate_level}, max_rounds={max_rounds}, "
        f"neuromorphic_mediator={neuromorphic_mediator})" if mode == "adversarial" else
        f"  (neuromorphic_mediator={neuromorphic_mediator})" if mode == "choreographed" else
        f"  (n_rotations={n_rotations})" if mode == "rotation" else ""
    ))

    queries = [args.query] if args.query else DEFAULT_QUERIES
    results = []

    if mode in ("neukrag", "neukrag-inter"):
        kg_name = "neuromorphic" if mode == "neukrag" else "all"
        graph = load_graph(kg_paths[kg_name])
        if mode == "neukrag":
            unified_meta = metadata.get("neuromorphic", {})
        else:
            unified_meta = {}
            for m in metadata.values():
                unified_meta.update(m)
        entity_extractor = EntityExtractor()
        hyp_generator    = HypothesisGenerator()

        for query in queries:
            result = run_neukrag_single(
                query, graph, unified_meta, entity_extractor, hyp_generator,
                k_hops=args.k_hops, max_triples=args.max_triples, mode=mode,
            )
            results.append(result)
            print(f"\n{'='*70}")
            print(f"QUERY:  {result['query']}")
            print(f"MODE:   {result['mode'].upper()}\n{'='*70}")
            print(result["final_hypothesis"])
    else:
        agents = [
            SpecialistAgent("neuroscience", kg_paths["neuroscience"], args.k_hops, args.max_triples,
                            metadata=metadata.get("neuroscience")),
            SpecialistAgent("aiml",         kg_paths["aiml"],         args.k_hops, args.max_triples,
                            metadata=metadata.get("aiml")),
            SpecialistAgent("neuromorphic", kg_paths["neuromorphic"], args.k_hops, args.max_triples,
                            metadata=metadata.get("neuromorphic")),
        ]
        mediator = Mediator()

        for query in queries:
            if mode == "synthesis":
                result = run_synthesis(query, agents, mediator)
            elif mode == "adversarial":
                result = run_adversarial(
                    query, agents, mediator, max_rounds, debate_level,
                    neuromorphic_mediator=neuromorphic_mediator,
                )
            elif mode == "rotation":
                result = run_rotation(query, agents, mediator, n_rotations=n_rotations)
            else:
                result = run_choreographed(
                    query, agents, mediator,
                    neuromorphic_mediator=neuromorphic_mediator,
                )

            results.append(result)
            print(f"\n{'='*70}")
            print(f"QUERY:  {result['query']}")
            print(f"MODE:   {result['mode'].upper()}", end="")
            if mode == "adversarial":
                print(f"  (level={result['debate_level']}, rounds={result['rounds_completed']})", end="")
            elif mode == "rotation":
                print(f"  (rotations={result['n_rotations']})", end="")
            if mode in ("adversarial", "choreographed") and result.get("neuromorphic_mediator"):
                print("  [neuromorphic-mediated]", end="")
            print(f"\n{'='*70}")
            print(result["final_hypothesis"])

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        log.info(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()
