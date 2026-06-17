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
from agents.neuroscience import NeuroscienceHypothesis, ROLE as NEURO_ROLE, LABEL as NEURO_LABEL
from agents.aiml         import AIMLHypothesis,         ROLE as AIML_ROLE,  LABEL as AIML_LABEL
from agents.neuromorphic import NeuromorphicHypothesis, ROLE as NEURO_M_ROLE, LABEL as NEURO_M_LABEL

# ---------------------------------------------------------------------------
# Chamber imports — run_* functions live in their own modules.
# Placed here (after all classes/helpers are defined) so the chambers can
# import back from this module without triggering a circular-import error.
# ---------------------------------------------------------------------------

from argora.graph_builder import RoundGraph                          # noqa: E402
from argora.qsem import compute_strengths_single_pass                # noqa: E402

from chambers.synthesis     import run_synthesis      # noqa: E402
from chambers.adversarial   import run_adversarial    # noqa: E402
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
# DSPy signatures — argumentation graph parsing and graph-aware synthesis
# ---------------------------------------------------------------------------

class ExpertOutputParser(dspy.Signature):
    """Parse a neuromorphic computing specialist's hypothesis into its core argumentative structure.
    Extract only arguments the expert is making IN SUPPORT of their own position."""

    query: str        = dspy.InputField(desc="The research query being addressed")
    expert_name: str  = dspy.InputField(desc="Name of the specialist agent")
    hypothesis: str   = dspy.InputField(desc="The specialist's full hypothesis text")
    main_argument: str = dspy.OutputField(
        desc="The single core claim of this hypothesis — the expert's direct answer, in one concise sentence"
    )
    supporting_arguments: str = dspy.OutputField(
        desc="Semicolon-separated supporting sub-claims or evidence that STRENGTHEN this expert's main claim; "
             "empty string if none. Do NOT include attacks on other experts here."
    )


class AgentArgumentMiner(dspy.Signature):
    """You are a domain specialist generating a single argument about your main hypothesis claim.
    Ground your argument specifically in the knowledge base context provided.
    Generate one concise argument (1-2 sentences) that either supports or attacks the main claim
    from your specialist domain perspective. Only generate an argument if you have a valid,
    domain-grounded one — otherwise return exactly 'N/A'."""

    query: str         = dspy.InputField(desc="The research query being addressed")
    agent_role: str    = dspy.InputField(desc="Your specialist domain role and expertise")
    graph_context: str = dspy.InputField(desc="Knowledge graph triples from your domain — ground your argument in these")
    main_claim: str    = dspy.InputField(desc="Your main hypothesis claim to argue about")
    polarity: str      = dspy.InputField(
        desc="'supporting' — generate an argument that strengthens the claim, "
             "or 'attacking' — generate one that challenges it"
    )
    argument: str = dspy.OutputField(
        desc="A single concise domain-grounded argument (1-2 sentences), or exactly 'N/A' if none can be made"
    )


class ArgumentStrengthAttributor(dspy.Signature):
    """You are a domain expert assessing the quality of a single argument.
    Score how compelling and well-grounded this argument is as evidence for or against its parent claim.
    Base your score on factuality, domain accuracy, logical coherence, and relevance to the parent claim."""

    agent_role: str   = dspy.InputField(desc="Your specialist domain role")
    argument: str     = dspy.InputField(desc="The argument to evaluate")
    parent_claim: str = dspy.InputField(desc="The claim this argument supports or attacks")
    polarity: str     = dspy.InputField(desc="'supporting' or 'attacking'")
    confidence: str   = dspy.OutputField(
        desc="An integer from 0 to 100: your confidence that this argument is valid and compelling. "
             "0 = definitely invalid or irrelevant, 100 = definitely valid and strongly compelling. "
             "Reply with the number only."
    )


class PeerArgumentElicitor(dspy.Signature):
    """You are a senior mediator in a neuromorphic computing research debate.
    An expert from one domain has made a main claim. A peer expert from a different
    domain must now evaluate it from their domain perspective.

    Produce a single, concise, domain-grounded reaction (≤ 60 words).
    The stance must be 'agree' if the peer's expertise supports the claim,
    or 'disagree' if their expertise reveals a flaw, limitation, or contradiction."""

    query: str         = dspy.InputField(desc="The research query being debated")
    author_name: str   = dspy.InputField(desc="Name of the expert who made the main claim")
    main_argument: str = dspy.InputField(desc="The main claim being evaluated")
    peer_name: str     = dspy.InputField(desc="Name of the peer expert evaluating this claim")
    peer_hypothesis: str = dspy.InputField(
        desc="The peer expert's own position on the query — use this to ground their reaction"
    )
    stance: str = dspy.OutputField(
        desc="Exactly 'agree' or 'disagree' — whether the peer's domain perspective supports or challenges the main argument"
    )
    reasoning: str = dspy.OutputField(
        desc="One concise sentence (≤ 60 words) giving the peer's key reason from their domain expertise"
    )


class MediatorGraphSynthesis(dspy.Signature):
    """You are a senior neuromorphic computing researcher. You have a structured
    argumentation graph of three domain specialists' positions computed using
    DFQuAD quantitative bipolar argumentation semantics (Baroni et al.).

    Each node has an integer id, an agent name, a statement, and a 'strength'
    value in [0, 1] representing its dialectical acceptability — how well-supported
    it is after accounting for all supporting and attacking arguments in the graph.
    Strength > 0.6 is well-supported; < 0.4 is significantly undermined.

    Reason over the graph:
    1. Prioritise claims with high dialectical strength (> 0.6).
    2. Treat contested claims (low strength) with scepticism; note the tension.
    3. Identify unique insights each agent contributes.
    4. Build a synthesis integrating the best-supported, most coherent positions.

    IMPORTANT — inline attribution: whenever you write a claim that draws directly
    from a node in the graph, wrap that text with an attribution tag:
        <label agent="AGENT_NAME" node_id="NODE_ID">your text here</label>
    Use the exact agent name and integer node_id from the graph JSON.
    Label every specific claim that has a clear source; leave general prose unlabelled."""

    query: str          = dspy.InputField(desc="Original research query")
    argument_graph: str = dspy.InputField(
        desc="JSON list of agent positions. Each entry has 'agent', 'main_claim' (with 'id', "
             "'statement', 'strength'), 'supporting' and 'attacking' sub-arguments (each with "
             "'id', 'statement', 'strength'). Strength ∈ [0,1] is DFQuAD dialectical acceptability."
    )
    synthesis: str = dspy.OutputField(
        desc="Unified hypothesis in markdown with inline <label> attribution tags. Open with a brief "
             "summary, then ## headers for each integrated perspective and a ## Synthesis Conclusion; "
             "bullet points for key claims, inline LaTeX ($...$) for equations, ``` for code/circuits. "
             "Wrap sourced claims: <label agent=\"NAME\" node_id=\"ID\">claim text</label>"
    )


class MediatorGraphExtractAnswer(dspy.Signature):
    """You are a moderator extracting a final answer from a completed multi-agent
    neuromorphic debate. You have the full debate history and a structured
    argumentation graph of the agents' most recent positions computed using
    DFQuAD quantitative bipolar argumentation semantics.

    Each node has a 'strength' ∈ [0,1] — its dialectical acceptability after
    accounting for all supports and attacks. Prioritise high-strength claims;
    note tensions around low-strength ones.

    IMPORTANT — inline attribution: wrap every claim drawn from a graph node with:
        <label agent="AGENT_NAME" node_id="NODE_ID">your text here</label>
    Use the exact agent name and integer node_id from the graph JSON."""

    query: str          = dspy.InputField(desc="Original research query")
    debate_history: str = dspy.InputField(desc="Complete debate history")
    argument_graph: str = dspy.InputField(
        desc="JSON list of agent positions. Each entry has 'agent', 'main_claim' (with 'id', "
             "'statement', 'strength'), 'supporting' and 'attacking' sub-arguments (each with "
             "'id', 'statement', 'strength'). Strength ∈ [0,1] is DFQuAD dialectical acceptability."
    )
    final_answer: str = dspy.OutputField(
        desc="Final synthesized hypothesis in markdown with inline <label> attribution tags. "
             "## headers per section, bullet points for key claims, inline LaTeX ($...$) for equations. "
             "Wrap sourced claims: <label agent=\"NAME\" node_id=\"ID\">claim text</label>"
    )


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


class ChoreographedAgreementResponse(dspy.Signature):
    """You are a domain specialist reviewing a mediator's synthesis of a multi-agent neuromorphic
    computing debate. Evaluate the synthesis from your specialist perspective.
    Acknowledge what the synthesis gets right from your domain's viewpoint and note any remaining
    gaps or corrections. Lean toward endorsing where the science is sound."""

    query: str          = dspy.InputField(desc="Original research query")
    agent_role: str     = dspy.InputField(desc="Your specialist role and evaluation criteria")
    synthesis: str      = dspy.InputField(desc="The mediator's synthesized hypothesis")
    debate_history: str = dspy.InputField(desc="Full debate history prior to this synthesis")
    response: str       = dspy.OutputField(desc="Your evaluation of the synthesis from your specialist lens, 1-2 paragraphs")
    agreed: str         = dspy.OutputField(desc='Answer "yes" if you broadly agree with the synthesis, "no" if you have significant remaining concerns')




# ---------------------------------------------------------------------------
# DSPy signatures — mediator
# ---------------------------------------------------------------------------


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
    final_answer: str   = dspy.OutputField(
        desc="Final synthesized hypothesis in markdown: ## headers per section, bullet points "
             "for key claims, inline LaTeX ($...$) for equations, ``` for circuit or algorithm descriptions"
    )

# TODO: Make the follow-up answers multi-agent too
class FollowUpAnswer(dspy.Signature):
    """You are a neuromorphic computing expert answering a follow-up question about a
    previously generated hypothesis. Use the synthesis as your primary knowledge base.
    Be concise and directly address the question."""

    previous_synthesis: str = dspy.InputField(desc="Previously generated neuromorphic hypothesis")
    followup_question: str  = dspy.InputField(desc="The follow-up question")
    answer: str             = dspy.OutputField(
        desc="Markdown-formatted answer using ## headers, bullet points, and LaTeX ($...$) where relevant"
    )


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

class SpecialistAgent(dspy.Module):
    def __init__(
        self,
        name: str,
        kg_path: Path,
        k_hops: int = 2,
        max_triples: int = 40,
        metadata: dict[int, dict] | None = None,
    ):
        super().__init__()
        self.name        = name
        self.role        = _AGENT_ROLES[name]
        self.k_hops      = k_hops
        self.max_triples = max_triples
        self.metadata         = metadata or {}
        self.graph            = load_graph(kg_path)
        self.entity_extractor = EntityExtractor()
        self.hyp_predict        = dspy.Predict(_HYPOTHESIS_SIGS[name])
        self.debate_predict     = dspy.Predict(DebateResponse)
        self.agreement_predict  = dspy.Predict(ChoreographedAgreementResponse)
        self.parse_predict      = dspy.Predict(ExpertOutputParser)
        self.arg_miner_predict  = dspy.Predict(AgentArgumentMiner)
        self.strength_attr_predict = dspy.Predict(ArgumentStrengthAttributor)

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

    def get_references(self, triples: list[dict]) -> str:
        """Return APA citations for the top-5 most-cited source papers in the triple set."""
        from collections import Counter
        counts = Counter(
            t["document_id"] for t in triples if t.get("document_id") is not None
        )
        refs = []
        for doc_id, _ in counts.most_common(5):
            paper = self.metadata.get(doc_id)
            if paper:
                refs.append(format_apa(paper))
        return "\n".join(refs)

    def build_local_arguments(self, query: str, hypothesis: str, graph_context: str) -> dict:
        """ArgLLMs Γ+ε per-agent QBAF construction.

        Extracts the agent's main claim, then generates one KG-grounded supporting
        argument and one attacking argument (Γ), scoring each with a domain-calibrated
        confidence (ε → base score τ).  Returns a dict that the Mediator merges into
        the shared RoundGraph.
        """
        parsed     = self.parse_predict(query=query, expert_name=self.name, hypothesis=hypothesis)
        main_claim = parsed.main_argument.strip()

        arguments: list[dict] = []
        for polarity in ("supporting", "attacking"):
            try:
                gen = self.arg_miner_predict(
                    query=query,
                    agent_role=self.role,
                    graph_context=graph_context,
                    main_claim=main_claim,
                    polarity=polarity,
                )
                arg_text = gen.argument.strip()
                if not arg_text or arg_text.lower() == "n/a":
                    continue
                score = self.strength_attr_predict(
                    agent_role=self.role,
                    argument=arg_text,
                    parent_claim=main_claim,
                    polarity=polarity,
                )
                raw = score.confidence.strip().rstrip("%").strip()
                try:
                    strength = min(1.0, max(0.0, float(raw) / 100.0))
                except ValueError:
                    strength = 0.5
                arguments.append({"text": arg_text, "polarity": polarity, "strength": strength})
            except Exception:
                continue

        return {"main_claim": main_claim, "arguments": arguments}

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

    def review_synthesis(self, query: str, synthesis: str, debate_history: str) -> tuple[str, bool]:
        """Returns (response_text, agreed) — used in choreographed round 5."""
        result = self.agreement_predict(
            query=query,
            agent_role=self.role,
            synthesis=synthesis,
            debate_history=debate_history,
        )
        agreed = result.agreed.strip().lower().startswith("yes")
        return result.response.strip(), agreed


class Mediator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.discriminative_predict  = dspy.Predict(MediatorJudgeDiscriminative)
        self.extractive_predict      = dspy.Predict(MediatorJudgeExtractive)
        self.followup_predict        = dspy.Predict(FollowUpAnswer)
        self.peer_elicit_predict     = dspy.Predict(PeerArgumentElicitor)
        self.graph_synthesis_predict = dspy.Predict(MediatorGraphSynthesis)
        self.graph_extract_predict   = dspy.Predict(MediatorGraphExtractAnswer)

    def build_argument_graph(self, query: str, agent_data: dict[str, dict]) -> RoundGraph:
        """Build a RoundGraph using the ArgLLMs Design B + ARGORA methodology:

        - Phase 1: add each agent's main claim from its per-agent local QBAF
        - Phase 2: per main claim, add Γ+ε local arguments (with ε base scores)
          and cross-agent peer reactions (MArgE-style meshing)

        agent_data[agent_name] = {"statement": str, "local_qbaf": dict}
        local_qbaf = {"main_claim": str, "arguments": [{"text", "polarity", "strength"}]}
        """
        graph = RoundGraph(topic=query, round=0)

        # ── Phase 1: register all main claims ───────────────────────────────
        main_ids: dict[str, int] = {}
        local_qbafs: dict[str, dict] = {}
        for agent_name, data in agent_data.items():
            lq      = data["local_qbaf"]
            main_id = graph.add_main(agent_name, lq["main_claim"])
            main_ids[agent_name]    = main_id
            local_qbafs[agent_name] = lq

        # ── Phase 2: per main claim, add local args + peer reactions ────────
        for target_agent, lq in local_qbafs.items():
            main_id   = main_ids[target_agent]
            main_stmt = lq["main_claim"]

            # Γ+ε local arguments — KG-grounded with domain-calibrated base scores
            for arg in lq["arguments"]:
                b       = arg["strength"]
                polarity = "support" if arg["polarity"] == "supporting" else "attack"
                if polarity == "support":
                    graph.add_support_to(target_agent, arg["text"], main_id, base=b)
                else:
                    graph.add_attack_to(target_agent, arg["text"], main_id, base=b)

            # Cross-agent peer reactions (MArgE-style meshing) — neutral base score
            for peer_agent, peer_data in agent_data.items():
                if peer_agent == target_agent:
                    continue
                try:
                    result = self.peer_elicit_predict(
                        query=query,
                        author_name=target_agent,
                        main_argument=main_stmt,
                        peer_name=peer_agent,
                        peer_hypothesis=peer_data["statement"],
                    )
                    stance    = result.stance.strip().lower()
                    reasoning = result.reasoning.strip()
                except Exception:
                    continue
                if not reasoning:
                    continue
                if stance.startswith("agree"):
                    graph.add_support_to(peer_agent, reasoning, main_id, base=0.5)
                else:
                    graph.add_attack_to(peer_agent, reasoning, main_id, base=0.5)

        return graph

    @staticmethod
    def _build_labeled_graph_json(graph: RoundGraph, strengths: dict[int, float]) -> str:
        """Build JSON with node IDs and DFQuAD strengths for LLM inline attribution."""
        gd = graph.to_dict()
        node_map = {n["id"]: n for n in gd.get("nodes", [])}
        supports_by_target: dict[int, list] = {}
        attacks_by_target: dict[int, list]  = {}
        for e in gd.get("edges", []):
            if e["edge_type"] == "support_edge":
                supports_by_target.setdefault(e["target"], []).append(e["source"])
            else:
                attacks_by_target.setdefault(e["target"], []).append(e["source"])

        def _strength(nid: int) -> float:
            return round(strengths.get(nid, node_map.get(nid, {}).get("strength", 0.5)), 3)

        items = []
        for node in gd.get("nodes", []):
            if node["type"] != "main_argument":
                continue
            nid = node["id"]
            items.append({
                "agent":      node["expert"],
                "main_claim": {"id": nid, "statement": node["statement"], "strength": _strength(nid)},
                "supporting": [
                    {"id": s, "statement": node_map[s]["statement"], "strength": _strength(s)}
                    for s in supports_by_target.get(nid, [])
                    if s in node_map
                ],
                "attacking": [
                    {"id": a, "statement": node_map[a]["statement"], "strength": _strength(a)}
                    for a in attacks_by_target.get(nid, [])
                    if a in node_map
                ],
            })
        return json.dumps(items, indent=2, ensure_ascii=False)

    @staticmethod
    def _enrich_graph_dict(graph: RoundGraph, strengths: dict[int, float]) -> dict:
        """Return to_dict() payload enriched with qsem_strength on every node."""
        gd = graph.to_dict()
        for node in gd.get("nodes", []):
            node["qsem_strength"] = round(
                strengths.get(node["id"], node.get("strength", 0.5)), 3
            )
        return gd

    def synthesize(self, query: str, agent_data: dict[str, dict]) -> dict:
        """Build an argumentation graph from per-agent ArgLLMs QBAFs, compute
        DFQuAD strengths, then synthesize.

        agent_data[name] = {"statement": str, "local_qbaf": dict}
        Returns {"text": str, "graph": dict}.
        """
        graph     = self.build_argument_graph(query, agent_data)
        strengths = compute_strengths_single_pass(graph, semantics="DFQuADModel")
        result    = self.graph_synthesis_predict(
            query=query,
            argument_graph=self._build_labeled_graph_json(graph, strengths),
        )
        return {
            "text":  result.synthesis.strip(),
            "graph": self._enrich_graph_dict(graph, strengths),
        }

    def can_conclude(self, query: str, debate_history: str) -> tuple[bool, str]:
        result    = self.discriminative_predict(query=query, debate_history=debate_history)
        concluded = result.concluded.strip().lower().startswith("yes")
        return concluded, result.reasoning.strip()

    def extract_answer(
        self,
        query: str,
        debate_history: str,
        *,
        agent_data: dict[str, dict] | None = None,
    ) -> dict:
        """Extract a final answer, optionally using per-agent ArgLLMs QBAFs.

        agent_data[name] = {"statement": str, "local_qbaf": dict}
        Returns {"text": str, "graph": dict | None}.
        """
        if agent_data:
            graph     = self.build_argument_graph(query, agent_data)
            strengths = compute_strengths_single_pass(graph, semantics="DFQuADModel")
            result    = self.graph_extract_predict(
                query=query,
                debate_history=debate_history,
                argument_graph=self._build_labeled_graph_json(graph, strengths),
            )
            return {
                "text":  result.final_answer.strip(),
                "graph": self._enrich_graph_dict(graph, strengths),
            }
        result = self.extractive_predict(query=query, debate_history=debate_history)
        return {
            "text":  result.final_answer.strip(),
            "graph": None,
        }

    def answer_followup(self, synthesis: str, question: str) -> str:
        result = self.followup_predict(previous_synthesis=synthesis, followup_question=question)
        return result.answer.strip()


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

# ---------------------------------------------------------------------------
# Metadata helpers — real APA citations from CSV paper records
# ---------------------------------------------------------------------------

def _format_author(name: str) -> str:
    """Convert 'First [Middle] Last' → 'Last, F. [M.]'"""
    parts = name.strip().rstrip(";").split()
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    last     = parts[-1]
    initials = " ".join(p[0] + "." for p in parts[:-1] if p)
    return f"{last}, {initials}"


def format_apa(paper: dict) -> str:
    """Format a paper metadata dict as an APA 7th-edition citation."""
    raw = str(paper.get("authors") or "")
    authors = [a.strip() for a in raw.split(";") if a.strip()]
    if len(authors) > 6:
        author_str = ", ".join(_format_author(a) for a in authors[:6]) + ", et al."
    elif len(authors) > 1:
        author_str = (
            ", ".join(_format_author(a) for a in authors[:-1])
            + f", & {_format_author(authors[-1])}"
        )
    elif authors:
        author_str = _format_author(authors[0])
    else:
        author_str = "Unknown"

    # year — "date" column is YYYY-MM-DD; "year" column is float
    year = ""
    if paper.get("date"):
        year = str(paper["date"])[:4]
    elif paper.get("year"):
        try:
            year = str(int(float(paper["year"])))
        except (ValueError, TypeError):
            year = str(paper["year"])

    title = paper.get("title", "Untitled")
    doi   = paper.get("doi", "")
    doi_str = f" https://doi.org/{doi}" if doi else ""
    return f"{author_str} ({year}). {title}.{doi_str}"


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
    if mode not in ("synthesis", "adversarial", "choreographed", "neukrag", "neukrag-inter"):
        sys.exit(f"ERROR: NEUKRAG_MODE must be 'synthesis', 'adversarial', 'choreographed', 'neukrag', or 'neukrag-inter', got '{mode}'")

    max_rounds   = int(env["NEUKRAG_DEBATE_ROUNDS"])
    debate_level = int(env["NEUKRAG_DEBATE_LEVEL"])
    if debate_level not in DEBATE_LEVEL_PROMPTS:
        sys.exit(f"ERROR: NEUKRAG_DEBATE_LEVEL must be 0-3, got {debate_level}")

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
        f"  (debate_level={debate_level}, max_rounds={max_rounds})" if mode == "adversarial" else ""
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
                result = run_adversarial(query, agents, mediator, max_rounds, debate_level)
            else:
                result = run_choreographed(query, agents, mediator)

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
