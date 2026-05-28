#!/usr/bin/env python3
"""
run_neukrag.py  —  NeuKRAG hypothesis generation via DSPy

Loads the built KG (output from extract_kg.py + cli merge/dedupe/cluster),
traverses it around a query, and synthesizes a neuromorphic circuit hypothesis.

Usage:
    python run_neukrag.py "<query>"
    python run_neukrag.py "<query>" --output-dir output_triple --k-hops 2
    python run_neukrag.py  # runs default queries from the paper

Default queries (from the NeuKRAG paper §4.1):
    1. "Design a biologically plausible, scalable spiking neuron model for neuromorphic hardware"
    2. "Which device technologies are best suited for implementing dynamic, task-adaptive learning rules?"

Prerequisites:
    python extract_kg.py <papers/>
    python -m src.kg_builder.cli merge
    python -m src.kg_builder.cli postmerge
    python -m src.kg_builder.cli dedupe
    python -m src.kg_builder.cli cluster


"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import deque
from pathlib import Path
from typing import Optional

import dspy
import networkx as nx

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

LLM_MODEL       = "openai/openai/gpt-oss-120b"
OLLAMA_BASE_URL = "http://earlsinclair.ornl.gov:8200/v1"
OLLAMA_API_KEY  = "vllm"

DEFAULT_QUERIES = [
    "Design a biologically plausible, scalable spiking neuron model for neuromorphic hardware",
    "Which device technologies are best suited for implementing dynamic, task-adaptive learning rules?",
]

# ---------------------------------------------------------------------------
# DSPy signatures
# ---------------------------------------------------------------------------

class ExtractQueryEntities(dspy.Signature):
    """Extract key technical entities from a neuromorphic computing research query.

    Return a JSON array of entity strings — e.g. ["LIF neuron", "STDP", "memristor"].
    Output ONLY the JSON array, no prose.
    """
    query: str      = dspy.InputField(desc="Research query about neuromorphic computing")
    entities: str   = dspy.OutputField(desc='JSON array of entity strings, e.g. ["LIF neuron", "STDP"]')


class SynthesizeHypothesis(dspy.Signature):
    """Synthesize a novel neuromorphic circuit hypothesis from knowledge graph evidence.

    You are an expert in neuromorphic computing, spiking neural networks, and
    brain-inspired hardware. Using the graph relations extracted from the literature,
    produce a well-structured hypothesis that:
    - Identifies a specific biological mechanism or learning rule
    - Proposes how it maps to a hardware-implementable circuit or architecture
    - Names relevant device technologies and expected performance benefits
    - Is grounded in the provided graph evidence, not speculation

    Write 3-5 coherent paragraphs in scientific style.
    """
    query: str          = dspy.InputField(desc="The original research question")
    graph_context: str  = dspy.InputField(desc="Relevant (head, relation, tail) triples from the KG")
    hypothesis: str     = dspy.OutputField(desc="Scientific hypothesis grounded in the graph evidence")


# ---------------------------------------------------------------------------
# DSPy modules
# ---------------------------------------------------------------------------

class EntityExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ExtractQueryEntities)

    def forward(self, query: str) -> list[str]:
        result = self.predict(query=query)
        raw = result.entities.strip()
        # strip markdown fences
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            entities = json.loads(raw)
            if isinstance(entities, list):
                return [str(e).strip() for e in entities if str(e).strip()]
        except json.JSONDecodeError:
            pass
        # fallback: comma-split
        return [t.strip().strip('"') for t in raw.split(",") if len(t.strip()) > 2]


class HypothesisGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(SynthesizeHypothesis)

    def forward(self, query: str, graph_context: str) -> str:
        result = self.predict(query=query, graph_context=graph_context)
        return result.hypothesis.strip()


# ---------------------------------------------------------------------------
# KG loader
# ---------------------------------------------------------------------------

def load_graph(output_dir: Path) -> nx.DiGraph:
    """Load the most refined available graph."""
    candidates = [
        output_dir / "merged" / "refined_graph.graphml",
        output_dir / "merged" / "deduped.jsonl",
        output_dir / "merged" / "all_triples.jsonl",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".graphml":
            g = nx.read_graphml(path)
            g = g.to_directed() if not isinstance(g, nx.DiGraph) else g
            log.info(f"Loaded GraphML: {len(g.nodes)} nodes, {len(g.edges)} edges ({path})")
            return g
        # JSONL
        g = nx.DiGraph()
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                t = json.loads(line)
                h, r, tail = t.get("h", ""), t.get("r", ""), t.get("t", "")
                if h and r and tail:
                    g.add_edge(h, tail, relation=r, document_id=t.get("document_id"))
        log.info(f"Loaded JSONL graph: {len(g.nodes)} nodes, {len(g.edges)} edges ({path})")
        return g
    sys.exit(f"ERROR: No KG found in {output_dir}/merged/. Run extract_kg.py + cli pipeline first.")


# ---------------------------------------------------------------------------
# Graph traversal
# ---------------------------------------------------------------------------

def keyword_entry_points(graph: nx.DiGraph, entities: list[str]) -> set[str]:
    """Find graph nodes matching any extracted entity (case-insensitive substring)."""
    nodes = set()
    for node in graph.nodes():
        if not isinstance(node, str):
            continue
        for ent in entities:
            if ent.lower() in node.lower():
                nodes.add(node)
                break
    return nodes


def bfs_subgraph(graph: nx.DiGraph, entry_nodes: set[str], k_hops: int, max_triples: int) -> list[dict]:
    """BFS from entry nodes, collecting (h, r, t) triples up to k_hops deep."""
    visited: set[str] = set()
    triples: list[dict] = []
    queue: deque[tuple[str, int]] = deque()

    for n in entry_nodes:
        if n in graph:
            queue.append((n, 0))
            visited.add(n)

    while queue and len(triples) < max_triples:
        node, depth = queue.popleft()
        if depth >= k_hops:
            continue
        for neighbor in graph.successors(node):
            edge = graph.get_edge_data(node, neighbor) or {}
            triples.append({"h": node, "r": edge.get("relation", "relates_to"), "t": neighbor,
                            "document_id": edge.get("document_id")})
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

    return triples


def format_context(triples: list[dict]) -> str:
    return "\n".join(f"{t['h']}  --[{t['r']}]-->  {t['t']}" for t in triples)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_query(
    query: str,
    graph: nx.DiGraph,
    entity_extractor: EntityExtractor,
    hyp_generator: HypothesisGenerator,
    k_hops: int,
    max_triples: int,
) -> dict:
    log.info(f"Query: {query}")

    entities = entity_extractor(query=query)
    log.info(f"  Entities: {entities}")

    entry_nodes = keyword_entry_points(graph, entities)
    log.info(f"  Entry nodes: {len(entry_nodes)}")

    if not entry_nodes:
        log.warning("  No entry nodes found — using all entities as fallback nodes")
        entry_nodes = set(entities)

    triples = bfs_subgraph(graph, entry_nodes, k_hops=k_hops, max_triples=max_triples)
    log.info(f"  Subgraph: {len(triples)} triples")

    context = format_context(triples)
    hypothesis = hyp_generator(query=query, graph_context=context)

    return {
        "query": query,
        "entities": entities,
        "entry_nodes": list(entry_nodes),
        "triples_used": len(triples),
        "hypothesis": hypothesis,
    }


def main():
    parser = argparse.ArgumentParser(description="NeuKRAG hypothesis generation via DSPy")
    parser.add_argument("query", nargs="?", default=None,
                        help="Research query (omit to run default paper queries)")
    parser.add_argument("--output-dir", type=Path, default=Path("output_triple"),
                        help="KG output directory (default: output_triple)")
    parser.add_argument("--k-hops", type=int, default=2,
                        help="BFS depth for graph traversal (default: 2)")
    parser.add_argument("--max-triples", type=int, default=40,
                        help="Max triples to include in context (default: 40)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Save results as JSON to this file")
    args = parser.parse_args()

    lm = dspy.LM(LLM_MODEL, api_base=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
    dspy.configure(lm=lm)
    log.info(f"DSPy configured with {LLM_MODEL} @ {OLLAMA_BASE_URL}")

    graph = load_graph(args.output_dir)
    entity_extractor = EntityExtractor()
    hyp_generator    = HypothesisGenerator()

    queries = [args.query] if args.query else DEFAULT_QUERIES
    results = []

    for query in queries:
        result = run_query(
            query, graph, entity_extractor, hyp_generator,
            k_hops=args.k_hops, max_triples=args.max_triples,
        )
        results.append(result)
        print(f"\n{'='*70}")
        print(f"QUERY: {result['query']}")
        print(f"{'='*70}")
        print(result["hypothesis"])

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        log.info(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()
