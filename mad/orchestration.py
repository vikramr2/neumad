#!/usr/bin/env python3
"""
orchestration.py  —  NeuKRAG Multi-Agent Debate orchestrator

Wires three KG-grounded specialist agents into either a cooperative synthesis
or a MAD-style adversarial debate, with a baseline LLM mediator.

Runtime configuration is read from mad/environment.json (defaults) and may
be overridden by shell environment variables of the same name:
    NEUKRAG_MODE          synthesis | adversarial
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

sys.path.insert(0, str(_NEUKRAG))
sys.path.insert(0, str(_MAD))

from run_neukrag import (                          # noqa: E402
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
from agents.neuroscience import NeuroscienceHypothesis, ROLE as NEURO_ROLE, LABEL as NEURO_LABEL
from agents.aiml         import AIMLHypothesis,         ROLE as AIML_ROLE,  LABEL as AIML_LABEL
from agents.neuromorphic import NeuromorphicHypothesis, ROLE as NEURO_M_ROLE, LABEL as NEURO_M_LABEL

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

_HYPOTHESIS_SIGS = {
    "neuroscience": NeuroscienceHypothesis,
    "aiml":         AIMLHypothesis,
    "neuromorphic": NeuromorphicHypothesis,
}

_AGENT_ROLES = {
    "neuroscience": NEURO_ROLE,
    "aiml":         AIML_ROLE,
    "neuromorphic": NEURO_M_ROLE,
}

_AGENT_LABELS = {
    "neuroscience": NEURO_LABEL,
    "aiml":         AIML_LABEL,
    "neuromorphic": NEURO_M_LABEL,
}


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
    agreed: str         = dspy.OutputField(desc='Answer "yes" if you broadly agree with the other agents\' positions, "no" if you disagree')


class ExtractReferences(dspy.Signature):
    """Given a neuromorphic computing hypothesis and the knowledge graph triples it was
    grounded in, generate 3-5 APA-formatted citations for the key papers underlying the
    main claims. The KG entities were extracted from neuromorphic computing literature;
    use your knowledge of this field to identify the most relevant source papers.
    Output only the citations, one per line, no bullet points or numbering."""

    hypothesis: str    = dspy.InputField(desc="Scientific hypothesis text")
    graph_context: str = dspy.InputField(desc="KG triples (head → relation → tail) used to ground the hypothesis")
    references: str    = dspy.OutputField(desc="3-5 APA-formatted citations, one per line")


# ---------------------------------------------------------------------------
# DSPy signatures — mediator
# ---------------------------------------------------------------------------

class MediatorSynthesis(dspy.Signature):
    """You are a senior neuromorphic computing researcher synthesizing independent hypotheses
    from three domain specialists into a single unified proposal. Integrate:
      - biological plausibility (neuroscience)
      - learning performance (AI/ML)
      - energy-efficient memristor implementability (neuromorphic hardware)
    Produce a coherent, actionable design that honours all three perspectives. Write 4-5 paragraphs."""

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
    hypothesis — favouring arguments that held up under challenge from all three specialists.
    Write 4-5 scientific paragraphs."""

    query: str          = dspy.InputField(desc="Original research query")
    debate_history: str = dspy.InputField(desc="Complete debate history")
    final_answer: str   = dspy.OutputField(desc="Final synthesized hypothesis grounded in the debate evidence")


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

class SpecialistAgent(dspy.Module):
    def __init__(self, name: str, kg_path: Path, k_hops: int = 2, max_triples: int = 40):
        super().__init__()
        self.name        = name
        self.role        = _AGENT_ROLES[name]
        self.k_hops      = k_hops
        self.max_triples = max_triples
        self.graph            = load_graph(kg_path)
        self.entity_extractor = EntityExtractor()
        self.hyp_predict      = dspy.Predict(_HYPOTHESIS_SIGS[name])
        self.debate_predict   = dspy.Predict(DebateResponse)
        self.ref_predict      = dspy.Predict(ExtractReferences)

    def initial_hypothesis(self, query: str) -> tuple[str, list[dict]]:
        """Returns (hypothesis_text, triples)."""
        entities    = self.entity_extractor(query=query)
        entry_nodes = keyword_entry_points(self.graph, entities)
        if not entry_nodes:
            log.warning(f"  [{self.name}] no entry nodes found — using entity names as fallback")
            entry_nodes = set(entities)
        triples = bfs_subgraph(self.graph, entry_nodes, k_hops=self.k_hops, max_triples=self.max_triples)
        context = format_context(triples)
        result  = self.hyp_predict(query=query, graph_context=context)
        return result.hypothesis.strip(), triples

    def get_references(self, hypothesis: str, triples: list[dict]) -> str:
        """Generate APA citations from KG evidence underlying a hypothesis."""
        context = format_context(triples) if triples else "(no KG triples available)"
        result  = self.ref_predict(hypothesis=hypothesis, graph_context=context)
        return result.references.strip()

    def debate_response(self, query: str, debate_history: str, debate_level: str) -> tuple[str, bool]:
        """Returns (response_text, agreed)."""
        result = self.debate_predict(
            query=query,
            agent_role=self.role,
            debate_level=debate_level,
            debate_history=debate_history,
        )
        agreed = result.agreed.strip().lower().startswith("yes")
        return result.response.strip(), agreed


class Mediator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.synthesis_predict      = dspy.Predict(MediatorSynthesis)
        self.discriminative_predict = dspy.Predict(MediatorJudgeDiscriminative)
        self.extractive_predict     = dspy.Predict(MediatorJudgeExtractive)

    def synthesize(self, query: str, hypotheses: dict[str, str]) -> str:
        result = self.synthesis_predict(
            query=query,
            neuroscience_hypothesis=hypotheses["neuroscience"],
            aiml_hypothesis=hypotheses["aiml"],
            neuromorphic_hypothesis=hypotheses["neuromorphic"],
        )
        return result.synthesis.strip()

    def can_conclude(self, query: str, debate_history: str) -> tuple[bool, str]:
        result    = self.discriminative_predict(query=query, debate_history=debate_history)
        concluded = result.concluded.strip().lower().startswith("yes")
        return concluded, result.reasoning.strip()

    def extract_answer(self, query: str, debate_history: str) -> str:
        result = self.extractive_predict(query=query, debate_history=debate_history)
        return result.final_answer.strip()


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


def run_synthesis(
    query: str,
    agents: list[SpecialistAgent],
    mediator: Mediator,
    status_cb=None,
) -> dict:
    def _status(msg: str):
        log.info(msg)
        if status_cb:
            status_cb(msg)

    _status(f"Query (synthesis): {query}")
    agent_hypotheses: dict[str, dict] = {}

    for agent in agents:
        _status(f"  [{agent.name}] generating hypothesis…")
        hyp, triples = agent.initial_hypothesis(query)
        _status(f"  [{agent.name}] extracting references…")
        refs = agent.get_references(hyp, triples)
        agent_hypotheses[agent.name] = {
            "statement":  hyp,
            "triples":    triples,
            "references": refs,
        }
        _status(f"  [{agent.name}] done ({len(triples)} triples used)")

    _status("  Mediator synthesizing…")
    synthesis = mediator.synthesize(
        query,
        {name: d["statement"] for name, d in agent_hypotheses.items()},
    )

    return {
        "query":            query,
        "mode":             "synthesis",
        "agent_hypotheses": agent_hypotheses,
        "final_hypothesis": synthesis,
    }


def run_adversarial(
    query: str,
    agents: list[SpecialistAgent],
    mediator: Mediator,
    max_rounds: int,
    debate_level: int,
    status_cb=None,
) -> dict:
    def _status(msg: str):
        log.info(msg)
        if status_cb:
            status_cb(msg)

    _status(f"Query (adversarial, level={debate_level}, max_rounds={max_rounds}): {query}")
    level_prompt = DEBATE_LEVEL_PROMPTS[debate_level]
    history: list[dict] = []

    # Round 0 — initial KG-grounded positions + per-agent references
    agent_refs: dict[str, str] = {}
    for agent in agents:
        _status(f"  [{agent.name}] round 0 — generating hypothesis…")
        hyp, triples = agent.initial_hypothesis(query)
        _status(f"  [{agent.name}] round 0 — extracting references…")
        refs = agent.get_references(hyp, triples)
        agent_refs[agent.name] = refs
        history.append({
            "agent":      agent.name,
            "round":      0,
            "statement":  hyp,
            "triples":    triples,
            "references": refs,
            "agreed":     None,   # no prior positions to agree/disagree with
        })
        _status(f"  [{agent.name}] round 0 complete ({len(triples)} triples used)")

    rounds_completed = 0
    for round_num in range(1, max_rounds + 1):
        _status(f"  Debate round {round_num}/{max_rounds}")
        history_str = format_debate_history(history)

        for agent in agents:
            _status(f"  [{agent.name}] round {round_num} — responding…")
            response, agreed = agent.debate_response(query, history_str, level_prompt)
            history.append({
                "agent":      agent.name,
                "round":      round_num,
                "statement":  response,
                "triples":    [],
                "references": agent_refs[agent.name],  # carry round-0 KG refs forward
                "agreed":     agreed,
            })

        rounds_completed = round_num
        history_str = format_debate_history(history)

        # Discriminative mode — adaptive break (MAD §2)
        concluded, reason = mediator.can_conclude(query, history_str)
        _status(f"  Judge: concluded={concluded} — {reason}")
        if concluded:
            _status("  Adaptive break: debate concluded early")
            break

    _status("  Mediator extracting final answer…")
    final_answer = mediator.extract_answer(query, format_debate_history(history))

    return {
        "query":            query,
        "mode":             "adversarial",
        "debate_level":     debate_level,
        "rounds_completed": rounds_completed,
        "debate_history":   history,
        "final_hypothesis": final_answer,
    }


# ---------------------------------------------------------------------------
# Config / env helpers
# ---------------------------------------------------------------------------

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
    if mode not in ("synthesis", "adversarial"):
        sys.exit(f"ERROR: NEUKRAG_MODE must be 'synthesis' or 'adversarial', got '{mode}'")

    max_rounds   = int(env["NEUKRAG_DEBATE_ROUNDS"])
    debate_level = int(env["NEUKRAG_DEBATE_LEVEL"])
    if debate_level not in DEBATE_LEVEL_PROMPTS:
        sys.exit(f"ERROR: NEUKRAG_DEBATE_LEVEL must be 0-3, got {debate_level}")

    cfg    = load_toml(CONFIG_PATH)
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
