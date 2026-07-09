"""
Mediator — argumentation-graph builder and synthesis/extraction module.

Builds a cross-agent QBAF using ArgLLMs Design B + ARGORA, computes DFQuAD
dialectical strengths, and synthesizes or extracts the final answer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import dspy

_ROOT   = Path(__file__).parent.parent.parent   # /home/vr9/neumad/
_ARGORA = _ROOT / "argora-public"
sys.path.insert(0, str(_ARGORA))

from argora.graph_builder import RoundGraph                    # noqa: E402
from argora.qsem import compute_strengths_single_pass          # noqa: E402


# ---------------------------------------------------------------------------
# DSPy signatures
# ---------------------------------------------------------------------------

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


class AgentMediatedSynthesis(dspy.Signature):
    """You are the {agent_role} — write in this voice, not as a neutral third party. You
    already staked out your own position on the query. Since then, two other domain
    specialists have debated it adversarially. Read their exchange and the structured
    argumentation graph (DFQuAD dialectical strengths), then produce YOUR final,
    conclusive position on the query — revised or reinforced by whatever held up under
    the debate, from your own domain's perspective. Prioritise claims with high
    dialectical strength (> 0.6); treat contested claims (< 0.4) with scepticism.

    IMPORTANT — inline attribution: wrap every claim drawn from a graph node with:
        <label agent="AGENT_NAME" node_id="NODE_ID">your text here</label>
    Use the exact agent name and integer node_id from the graph JSON."""

    query:          str = dspy.InputField(desc="Original research query")
    agent_role:     str = dspy.InputField(desc="Your own specialist role and expertise — answer in this voice")
    own_position:   str = dspy.InputField(desc="Your own initial position on the query, staked out before the debate")
    debate_history: str = dspy.InputField(desc="The adversarial debate between your two peers")
    argument_graph: str = dspy.InputField(
        desc="JSON list of agent positions. Each entry has 'agent', 'main_claim' (with 'id', "
             "'statement', 'strength'), 'supporting' and 'attacking' sub-arguments (each with "
             "'id', 'statement', 'strength'). Strength ∈ [0,1] is DFQuAD dialectical acceptability."
    )
    final_answer: str = dspy.OutputField(
        desc="Your final, conclusive position on the query, in your own specialist voice, in "
             "markdown with inline <label> attribution tags. ## headers per section, bullet "
             "points for key claims, inline LaTeX ($...$) for equations. Wrap sourced claims: "
             "<label agent=\"NAME\" node_id=\"ID\">claim text</label>"
    )


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


class PositionMatch(dspy.Signature):
    """You are a neutral moderator tracking how a specialist's position evolves across
    a multi-agent debate. Compare the specialist's updated claim against a small set of
    candidate prior positions: their own previous claim, and each peer's previous claim.
    Judge substantive agreement, not surface wording — a claim can be reworded and still
    be 'own', or share vocabulary with a peer while still disagreeing with it."""

    query:           str = dspy.InputField(desc="Original research query")
    agent_name:      str = dspy.InputField(desc="Name of the specialist whose position is being tracked")
    new_claim:       str = dspy.InputField(desc="The specialist's updated main claim this round")
    own_prior_claim: str = dspy.InputField(desc="The specialist's own main claim from the previous round")
    peer_claims:     str = dspy.InputField(
        desc="JSON object mapping peer agent name to that peer's main claim from the previous round"
    )
    match: str = dspy.OutputField(
        desc="Exactly one of: 'own' (still substantively the same as their own prior claim), "
             "the exact peer name as given in peer_claims (the claim now substantively matches "
             "that peer's prior claim), or 'novel' (a genuinely new position matching none of the above)"
    )


class SynthesisProvenance(dspy.Signature):
    """You are a neutral moderator assessing whose position a synthesized answer draws from.
    Read the synthesis and compare its core claim against each specialist's original claim.
    Determine whether the synthesis's substance primarily reflects one specialist's position,
    or is a genuine blend that doesn't reduce to any single one."""

    query:        str = dspy.InputField(desc="Original research query")
    synthesis:    str = dspy.InputField(desc="The mediator's full synthesized answer")
    agent_claims: str = dspy.InputField(desc="JSON object mapping agent name to their original main claim")
    match: str = dspy.OutputField(
        desc="Exactly one of: the exact agent name from agent_claims whose position the synthesis "
             "primarily reflects, or 'blended' if it's a genuine mix that doesn't reduce to any single agent"
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
# Mediator
# ---------------------------------------------------------------------------

class Mediator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.discriminative_predict  = dspy.Predict(MediatorJudgeDiscriminative)
        self.extractive_predict      = dspy.Predict(MediatorJudgeExtractive)
        self.followup_predict        = dspy.Predict(FollowUpAnswer)
        self.peer_elicit_predict     = dspy.Predict(PeerArgumentElicitor)
        self.graph_synthesis_predict = dspy.Predict(MediatorGraphSynthesis)
        self.graph_extract_predict   = dspy.Predict(MediatorGraphExtractAnswer)
        self.position_match_predict  = dspy.Predict(PositionMatch)
        self.synthesis_provenance_predict = dspy.Predict(SynthesisProvenance)
        self.agent_mediated_predict  = dspy.Predict(AgentMediatedSynthesis)

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
                b        = arg["strength"]
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

    def classify_transition(
        self,
        query: str,
        agent_name: str,
        new_claim: str,
        own_prior_claim: str,
        peer_claims: dict[str, str],
    ) -> str:
        """Classify an agent's round-over-round position change.

        Returns 'own' (unchanged), a peer name from peer_claims (position moved to
        match that peer), or 'novel' (changed but matches no one in particular).
        """
        result = self.position_match_predict(
            query=query,
            agent_name=agent_name,
            new_claim=new_claim,
            own_prior_claim=own_prior_claim,
            peer_claims=json.dumps(peer_claims, ensure_ascii=False),
        )
        match = result.match.strip()
        if match == "own" or match == "novel" or match in peer_claims:
            return match
        for name in peer_claims:
            if match.lower() == name.lower():
                return name
        return "novel"

    def classify_synthesis_provenance(
        self,
        query: str,
        synthesis: str,
        agent_claims: dict[str, str],
    ) -> str:
        """Classify whose position a synthesis primarily reflects.

        Returns an agent name from agent_claims, or 'blended' if it's a genuine mix.
        """
        result = self.synthesis_provenance_predict(
            query=query,
            synthesis=synthesis,
            agent_claims=json.dumps(agent_claims, ensure_ascii=False),
        )
        match = result.match.strip()
        if match == "blended" or match in agent_claims:
            return match
        for name in agent_claims:
            if match.lower() == name.lower():
                return name
        return "blended"

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

    def mediate_as_agent(
        self,
        query: str,
        agent_role: str,
        own_position: str,
        debate_history: str,
        *,
        agent_data: dict[str, dict],
    ) -> dict:
        """Synthesize a debate into a final answer written in a specific agent's own
        voice (neuromorphic-mediator mode), rather than a neutral third-party mediator.

        agent_data[name] = {"statement": str, "local_qbaf": dict} for every contributing
        agent, including the one whose voice this is — so its own claim gets a graph
        node and can be labelled/hover-attributed too, same as every other mode.
        Returns {"text": str, "graph": dict}.
        """
        graph     = self.build_argument_graph(query, agent_data)
        strengths = compute_strengths_single_pass(graph, semantics="DFQuADModel")
        result    = self.agent_mediated_predict(
            query=query,
            agent_role=agent_role,
            own_position=own_position,
            debate_history=debate_history,
            argument_graph=self._build_labeled_graph_json(graph, strengths),
        )
        return {
            "text":  result.final_answer.strip(),
            "graph": self._enrich_graph_dict(graph, strengths),
        }

    def answer_followup(self, synthesis: str, question: str) -> str:
        result = self.followup_predict(previous_synthesis=synthesis, followup_question=question)
        return result.answer.strip()
