#!/usr/bin/env python3
"""
run_mad_neukrag.py  —  Multi-Agent Debate NeuKRAG

Three KG-grounded specialist agents debate or synthesize neuromorphic hypotheses:
  neuroscience   (kg_neuroscience) — biological plausibility, brain mechanisms
  aiml           (kg_aiml)         — performance, accuracy metrics, learning efficiency
  neuromorphic   (kg_neuromorphic) — energy efficiency, memristor implementability

A baseline LLM mediator synthesizes (cooperative) or judges (adversarial) the outcome.

Usage:
    python run_mad_neukrag.py "<query>"
    python run_mad_neukrag.py              # runs default queries from the paper
    python run_mad_neukrag.py "<query>" --out results.json

Environment variables:
    NEUKRAG_MODE          synthesis | adversarial  (default: synthesis)
    NEUKRAG_DEBATE_ROUNDS max debate rounds in adversarial mode (default: 3)
    NEUKRAG_DEBATE_LEVEL  0-3 tit-for-tat intensity (default: 2)
                            0 = full consensus required on every point
                            1 = mostly disagree, minor consensus on peripheral points ok
                            2 = find the correct answer, agree only where you must (default)
                            3 = disagree on every point, no consensus whatsoever

Design follows the MAD framework (Liang et al. 2024):
  - Round 0:     each agent generates an initial KG-grounded hypothesis
  - Rounds 1..N: agents respond to full debate history from their specialist lens
  - After each round: mediator (discriminative mode) decides whether to break early
  - Final:       mediator (extractive mode) synthesizes the winning arguments
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import dspy

sys.path.insert(0, str(Path(__file__).parent))
from run_neukrag import (
    DEFAULT_QUERIES,
    LLM_MODEL,
    OLLAMA_API_KEY,
    OLLAMA_BASE_URL,
    EntityExtractor,
    format_context,
    keyword_entry_points,
    bfs_subgraph,
    load_graph,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config.toml"

# From MAD paper Table 11 — tit-for-tat levels
DEBATE_LEVEL_PROMPTS = {
    0: "All agents must reach full consensus on every point. Every claim must be agreed upon by all sides.",
    1: "Most points should show disagreement, but minor consensus on peripheral details is acceptable.",
    2: "It's not necessary to fully agree with each other's perspectives; the objective is to find the correct answer.",
    3: "All agents must disagree on every point. There should be no consensus whatsoever.",
}

AGENT_ROLES = {
    "neuroscience": (
        "Neuroscience Specialist: You prioritize biological plausibility and fidelity to known "
        "brain mechanisms — ion channel and synaptic dynamics, cortical circuit organization, "
        "and biologically grounded learning rules (e.g., STDP, BCM, homeostatic plasticity). "
        "Evaluate every proposal through one lens: does this faithfully model how the brain works?"
    ),
    "aiml": (
        "AI/ML Specialist: You prioritize measurable learning performance — accuracy on benchmarks, "
        "generalization, convergence speed, and task-adaptive training. You care about loss "
        "landscapes, gradient flow, and whether a proposed model improves over existing baselines. "
        "Evaluate every proposal through one lens: does this achieve better accuracy or learning outcomes?"
    ),
    "neuromorphic": (
        "Neuromorphic Engineering Specialist: You prioritize energy efficiency and hardware "
        "implementability in memristor-based architectures — power per spike, CMOS compatibility, "
        "on-chip weight storage, device endurance, and analog noise tolerance. "
        "Evaluate every proposal through one lens: can this be built efficiently in silicon/memristors?"
    ),
}

ROLE_LABELS = {
    "neuroscience": "Neuroscience Specialist",
    "aiml": "AI/ML Specialist",
    "neuromorphic": "Neuromorphic Engineering Specialist",
}


# ---------------------------------------------------------------------------
# DSPy signatures — initial hypothesis (one per specialty)
# ---------------------------------------------------------------------------

class NeuroscienceHypothesis(dspy.Signature):
    """You are a computational neuroscientist synthesizing a neuromorphic circuit hypothesis.
    Ground every claim in the provided KG evidence. Prioritize: biological plausibility,
    ion channel / synaptic dynamics, cortical circuit organization, and biologically realistic
    learning rules. Write 2-3 focused scientific paragraphs."""

    query: str         = dspy.InputField(desc="Research query about neuromorphic computing")
    graph_context: str = dspy.InputField(desc="(head, relation, tail) triples from neuroscience KG")
    hypothesis: str    = dspy.OutputField(desc="Bio-realism focused scientific hypothesis")


class AIMLHypothesis(dspy.Signature):
    """You are an AI/ML researcher synthesizing a neuromorphic circuit hypothesis.
    Ground every claim in the provided KG evidence. Prioritize: accuracy on benchmarks,
    learning efficiency, generalization, convergence speed, and task-adaptive training.
    Write 2-3 focused scientific paragraphs."""

    query: str         = dspy.InputField(desc="Research query about neuromorphic computing")
    graph_context: str = dspy.InputField(desc="(head, relation, tail) triples from AI/ML KG")
    hypothesis: str    = dspy.OutputField(desc="Performance-focused scientific hypothesis")


class NeuromorphicHypothesis(dspy.Signature):
    """You are a neuromorphic hardware engineer synthesizing a circuit hypothesis.
    Ground every claim in the provided KG evidence. Prioritize: energy efficiency,
    memristor device physics, CMOS implementability, on-chip learning, and power per spike.
    Write 2-3 focused scientific paragraphs."""

    query: str         = dspy.InputField(desc="Research query about neuromorphic computing")
    graph_context: str = dspy.InputField(desc="(head, relation, tail) triples from neuromorphic KG")
    hypothesis: str    = dspy.OutputField(desc="Energy-efficiency and memristor focused scientific hypothesis")


# ---------------------------------------------------------------------------
# DSPy signatures — adversarial debate
# ---------------------------------------------------------------------------

class DebateResponse(dspy.Signature):
    """You are a domain specialist in a multi-agent neuromorphic computing debate.
    Read the full debate history and respond from your specialist perspective.
    Challenge points that contradict your domain's evidence; briefly acknowledge what is valid.
    Do not simply restate your previous position — engage with the other agents' arguments.
    Follow the debate level instruction strictly."""

    query: str          = dspy.InputField(desc="Original research query")
    agent_role: str     = dspy.InputField(desc="Your specialist role and evaluation criteria")
    debate_level: str   = dspy.InputField(desc="Instruction governing how strongly you must disagree")
    debate_history: str = dspy.InputField(desc="Full debate history so far (all agents, all rounds)")
    response: str       = dspy.OutputField(desc="Your rebuttal or refinement from your specialist lens, 1-2 paragraphs")


# ---------------------------------------------------------------------------
# DSPy signatures — mediator
# ---------------------------------------------------------------------------

class MediatorSynthesis(dspy.Signature):
    """You are a senior neuromorphic computing researcher synthesizing independent hypotheses
    from three domain specialists into a single unified proposal. Integrate:
      - biological plausibility (neuroscience)
      - learning performance (AI/ML)
      - energy-efficient memristor implementability (neuromorphic hardware)
    Produce a coherent, actionable design that honors all three perspectives. Write 4-5 paragraphs."""

    query: str                   = dspy.InputField(desc="Original research query")
    neuroscience_hypothesis: str = dspy.InputField(desc="Neuroscience agent hypothesis (bio-realism)")
    aiml_hypothesis: str         = dspy.InputField(desc="AI/ML agent hypothesis (performance/accuracy)")
    neuromorphic_hypothesis: str = dspy.InputField(desc="Neuromorphic agent hypothesis (energy/memristors)")
    synthesis: str               = dspy.OutputField(desc="Unified hypothesis integrating all three perspectives")


class MediatorJudgeDiscriminative(dspy.Signature):
    """You are a moderator of a multi-agent neuromorphic computing debate. After each round,
    decide whether the debate has converged to a sufficiently complete and well-supported answer,
    or whether another round of argument would be productive. Prefer to continue if major
    technical disagreements remain unresolved."""

    query: str          = dspy.InputField(desc="Original research query")
    debate_history: str = dspy.InputField(desc="Complete debate history so far")
    concluded: str      = dspy.OutputField(desc='Answer "yes" if the debate has reached a satisfactory answer, "no" to continue')
    reasoning: str      = dspy.OutputField(desc="One sentence explaining your decision")


class MediatorJudgeExtractive(dspy.Signature):
    """You are a moderator extracting a final answer from a completed multi-agent neuromorphic
    debate. Based on the full debate history, synthesize the strongest, most evidence-grounded
    hypothesis — favoring arguments that held up under challenge from all three specialists.
    Write 4-5 scientific paragraphs."""

    query: str          = dspy.InputField(desc="Original research query")
    debate_history: str = dspy.InputField(desc="Complete debate history")
    final_answer: str   = dspy.OutputField(desc="Final synthesized hypothesis grounded in the debate evidence")


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

_HYPOTHESIS_SIGS = {
    "neuroscience": NeuroscienceHypothesis,
    "aiml": AIMLHypothesis,
    "neuromorphic": NeuromorphicHypothesis,
}


class SpecialistAgent(dspy.Module):
    def __init__(self, name: str, kg_path: Path, k_hops: int = 2, max_triples: int = 40):
        super().__init__()
        self.name = name
        self.role = AGENT_ROLES[name]
        self.k_hops = k_hops
        self.max_triples = max_triples
        self.graph = load_graph(kg_path)
        self.entity_extractor = EntityExtractor()
        self.hyp_predict = dspy.Predict(_HYPOTHESIS_SIGS[name])
        self.debate_predict = dspy.Predict(DebateResponse)

    def initial_hypothesis(self, query: str) -> tuple[str, int]:
        entities = self.entity_extractor(query=query)
        entry_nodes = keyword_entry_points(self.graph, entities)
        if not entry_nodes:
            log.warning(f"  [{self.name}] no entry nodes found — using entity names as fallback")
            entry_nodes = set(entities)
        triples = bfs_subgraph(self.graph, entry_nodes, k_hops=self.k_hops, max_triples=self.max_triples)
        context = format_context(triples)
        result = self.hyp_predict(query=query, graph_context=context)
        return result.hypothesis.strip(), len(triples)

    def debate_response(self, query: str, debate_history: str, debate_level: str) -> str:
        result = self.debate_predict(
            query=query,
            agent_role=self.role,
            debate_level=debate_level,
            debate_history=debate_history,
        )
        return result.response.strip()


class Mediator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.synthesis_predict     = dspy.Predict(MediatorSynthesis)
        self.discriminative_predict = dspy.Predict(MediatorJudgeDiscriminative)
        self.extractive_predict    = dspy.Predict(MediatorJudgeExtractive)

    def synthesize(self, query: str, hypotheses: dict[str, str]) -> str:
        result = self.synthesis_predict(
            query=query,
            neuroscience_hypothesis=hypotheses["neuroscience"],
            aiml_hypothesis=hypotheses["aiml"],
            neuromorphic_hypothesis=hypotheses["neuromorphic"],
        )
        return result.synthesis.strip()

    def can_conclude(self, query: str, debate_history: str) -> tuple[bool, str]:
        result = self.discriminative_predict(query=query, debate_history=debate_history)
        concluded = result.concluded.strip().lower().startswith("yes")
        return concluded, result.reasoning.strip()

    def extract_answer(self, query: str, debate_history: str) -> str:
        result = self.extractive_predict(query=query, debate_history=debate_history)
        return result.final_answer.strip()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def format_debate_history(history: list[dict]) -> str:
    lines = []
    for entry in history:
        label = ROLE_LABELS.get(entry["agent"], entry["agent"].upper())
        prefix = "Initial Position" if entry["round"] == 0 else f"Round {entry['round']} Response"
        lines.append(f"[{label} — {prefix}]:\n{entry['statement']}\n")
    return "\n".join(lines)


def load_config(config_path: Path) -> dict:
    try:
        import tomllib
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    except ImportError:
        pass
    try:
        import tomli
        with open(config_path, "rb") as f:
            return tomli.load(f)
    except ImportError:
        pass
    # manual fallback for simple TOML
    cfg: dict = {}
    section = None
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
                cfg[section] = {}
            elif "=" in line and section is not None:
                k, v = line.split("=", 1)
                cfg[section][k.strip()] = v.strip().strip('"').strip("'")
    return cfg


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_synthesis(
    query: str,
    agents: list[SpecialistAgent],
    mediator: Mediator,
) -> dict:
    log.info(f"Query (synthesis): {query}")
    hypotheses: dict[str, str] = {}
    triples_used: dict[str, int] = {}

    for agent in agents:
        hyp, n_triples = agent.initial_hypothesis(query)
        hypotheses[agent.name] = hyp
        triples_used[agent.name] = n_triples
        log.info(f"  [{agent.name}] {n_triples} triples used")

    log.info("  Mediator synthesizing...")
    synthesis = mediator.synthesize(query, hypotheses)

    return {
        "query": query,
        "mode": "synthesis",
        "agent_hypotheses": hypotheses,
        "triples_used": triples_used,
        "final_hypothesis": synthesis,
    }


def run_adversarial(
    query: str,
    agents: list[SpecialistAgent],
    mediator: Mediator,
    max_rounds: int,
    debate_level: int,
) -> dict:
    log.info(f"Query (adversarial, level={debate_level}, max_rounds={max_rounds}): {query}")
    level_prompt = DEBATE_LEVEL_PROMPTS[debate_level]
    history: list[dict] = []

    # Round 0 — each agent stakes its initial position from its KG
    for agent in agents:
        hyp, n_triples = agent.initial_hypothesis(query)
        history.append({"agent": agent.name, "round": 0, "statement": hyp})
        log.info(f"  [{agent.name}] round 0 complete ({n_triples} triples used)")

    rounds_completed = 0
    for round_num in range(1, max_rounds + 1):
        log.info(f"  Debate round {round_num}/{max_rounds}")
        history_str = format_debate_history(history)

        for agent in agents:
            response = agent.debate_response(query, history_str, level_prompt)
            history.append({"agent": agent.name, "round": round_num, "statement": response})

        rounds_completed = round_num
        history_str = format_debate_history(history)

        # Discriminative mode — adaptive break (MAD §2)
        concluded, reason = mediator.can_conclude(query, history_str)
        log.info(f"  Judge: concluded={concluded} — {reason}")
        if concluded:
            log.info("  Adaptive break: debate concluded early")
            break

    # Extractive mode — final answer from full history
    log.info("  Mediator extracting final answer...")
    final_answer = mediator.extract_answer(query, format_debate_history(history))

    return {
        "query": query,
        "mode": "adversarial",
        "debate_level": debate_level,
        "rounds_completed": rounds_completed,
        "debate_history": history,
        "final_hypothesis": final_answer,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Debate NeuKRAG hypothesis generation")
    parser.add_argument("query", nargs="?", default=None,
                        help="Research query (omit to run default paper queries)")
    parser.add_argument("--k-hops", type=int, default=2,
                        help="BFS depth for graph traversal per agent (default: 2)")
    parser.add_argument("--max-triples", type=int, default=40,
                        help="Max triples per agent context (default: 40)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Save results as JSON to this file")
    args = parser.parse_args()

    mode = os.environ.get("NEUKRAG_MODE", "synthesis").lower()
    if mode not in ("synthesis", "adversarial"):
        sys.exit(f"ERROR: NEUKRAG_MODE must be 'synthesis' or 'adversarial', got '{mode}'")

    max_rounds  = int(os.environ.get("NEUKRAG_DEBATE_ROUNDS", "3"))
    debate_level = int(os.environ.get("NEUKRAG_DEBATE_LEVEL", "2"))
    if debate_level not in DEBATE_LEVEL_PROMPTS:
        sys.exit(f"ERROR: NEUKRAG_DEBATE_LEVEL must be 0-3, got {debate_level}")

    cfg = load_config(CONFIG_PATH)
    kg_cfg = cfg.get("kg_paths", {})
    kg_paths = {
        "neuroscience": Path(kg_cfg.get("neuroscience_kg", "")).expanduser(),
        "aiml":         Path(kg_cfg.get("aiml_kg", "")).expanduser(),
        "neuromorphic": Path(kg_cfg.get("neuromorphic_kg", "")).expanduser(),
    }
    for name, path in kg_paths.items():
        if not path.exists():
            sys.exit(f"ERROR: KG path for '{name}' not found: {path}")

    lm = dspy.LM(LLM_MODEL, api_base=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
    dspy.configure(lm=lm)
    log.info(f"DSPy configured with {LLM_MODEL} @ {OLLAMA_BASE_URL}")
    log.info(f"Mode: {mode}" + (
        f"  (debate_level={debate_level}, max_rounds={max_rounds})" if mode == "adversarial" else ""
    ))

    agents = [
        SpecialistAgent("neuroscience", kg_paths["neuroscience"], args.k_hops, args.max_triples),
        SpecialistAgent("aiml",         kg_paths["aiml"],         args.k_hops, args.max_triples),
        SpecialistAgent("neuromorphic", kg_paths["neuromorphic"], args.k_hops, args.max_triples),
    ]
    mediator = Mediator()

    queries = [args.query] if args.query else DEFAULT_QUERIES
    results = []

    for query in queries:
        if mode == "synthesis":
            result = run_synthesis(query, agents, mediator)
        else:
            result = run_adversarial(query, agents, mediator, max_rounds, debate_level)

        results.append(result)
        print(f"\n{'='*70}")
        print(f"QUERY:  {result['query']}")
        print(f"MODE:   {result['mode'].upper()}", end="")
        if mode == "adversarial":
            print(f"  (level={result['debate_level']}, rounds={result['rounds_completed']})", end="")
        print(f"\n{'='*70}")
        print(result["final_hypothesis"])

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        log.info(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()
