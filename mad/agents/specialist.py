"""
SpecialistAgent — KG-grounded debate participant.

Wraps a domain-specific knowledge graph and a set of DSPy predictors to produce
initial hypotheses, per-agent QBAFs, debate responses, and synthesis reviews.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import dspy

_ROOT    = Path(__file__).parent.parent.parent   # /home/vr9/neumad/
_NEUKRAG = _ROOT / "neukrag"
sys.path.insert(0, str(_NEUKRAG))

from run_neukrag import (                         # noqa: E402
    EntityExtractor,
    format_context,
    keyword_entry_points,
    bfs_subgraph,
    load_graph,
)
from agents.neuroscience import NeuroscienceHypothesis, ROLE as NEURO_ROLE
from agents.aiml         import AIMLHypothesis,         ROLE as AIML_ROLE
from agents.neuromorphic import NeuromorphicHypothesis, ROLE as NEURO_M_ROLE

log = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# DSPy signatures
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


class PositionEdit(dspy.Signature):
    """You are a domain specialist participating in a round-robin refinement of a shared
    research position. This is a SURGICAL EDIT task, not a rewrite: keep the existing
    wording verbatim wherever you agree with it, and only change, add, or remove the
    specific sentences that need it from your domain's perspective. Do not restate the
    position in your own words, and do not reorganize or rephrase existing content —
    edit the text you were given in place. Return the full position after your edits,
    with unchanged parts reproduced verbatim, not a summary of your changes.

    If this is the final rotation (is_final_round = 'yes'), make sure the result reads
    as a complete, conclusive answer to the original query, still by editing the
    existing text rather than replacing it with a fresh write-up. In that case only,
    you may also insert a small number of markdown '##' section headers between
    existing, unchanged paragraphs (e.g. a short-answer summary, supporting evidence,
    remaining caveats) purely to organize the final answer for readability — this adds
    headers around content, it does not rewrite, move, or rephrase any existing
    sentence. If is_final_round is 'no', do not add headers."""

    query:            str = dspy.InputField(desc="Original research query")
    agent_role:       str = dspy.InputField(desc="Your specialist role and expertise")
    graph_context:    str = dspy.InputField(desc="Knowledge graph triples from your domain — ground your edits in these")
    current_position: str = dspy.InputField(desc="The current shared position to edit — preserve wording you agree with verbatim")
    is_final_round:   str = dspy.InputField(
        desc="'yes' if this is the final rotation and your output is the final answer, "
             "'no' if it will be passed to another specialist"
    )
    revised_position: str = dspy.OutputField(
        desc="The full position after your edits: parts you didn't change reproduced "
             "verbatim, plus your additions/corrections. Not a rewrite or summary in your own words."
    )


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
# APA citation helpers
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


# ---------------------------------------------------------------------------
# SpecialistAgent
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
        self.edit_predict       = dspy.Predict(PositionEdit)

    def retrieve_context(self, query: str) -> tuple[str, list[dict]]:
        """KG retrieval only (no generation) — the lookup initial_hypothesis and
        edit_position both need, exposed for callers that want triples/context without
        also generating a fresh hypothesis (e.g. grounding a debate response after the
        fact, to build a QBAF from a position the agent already stated)."""
        entities    = self.entity_extractor(query=query)
        entry_nodes = keyword_entry_points(self.graph, entities)
        if not entry_nodes:
            log.warning(f"  [{self.name}] no entry nodes found — using entity names as fallback")
            entry_nodes = set(entities)
        triples = bfs_subgraph(self.graph, entry_nodes, k_hops=self.k_hops, max_triples=self.max_triples)
        return format_context(triples), triples

    def initial_hypothesis(self, query: str) -> tuple[str, list[dict]]:
        """Returns (hypothesis_text, triples)."""
        context, triples = self.retrieve_context(query)
        result = self.hyp_predict(query=query, graph_context=context)
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

    def extract_main_claim(self, query: str, statement: str) -> str:
        """Extract just the single core claim from a statement (reuses ExpertOutputParser).

        Cheaper than build_local_arguments — no argument mining, just the claim
        extraction — so it's safe to call every round for position tracking.
        """
        parsed = self.parse_predict(query=query, expert_name=self.name, hypothesis=statement)
        return parsed.main_argument.strip()

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

    def edit_position(self, query: str, current_position: str, is_final: bool = False) -> tuple[str, list[dict]]:
        """Revise a shared position from this agent's domain perspective (rotation mode).

        Returns (revised_text, triples). Re-grounds in this agent's own KG each turn,
        same retrieval as initial_hypothesis, since the point of passing the position
        around is for each domain to inject its own evidence.
        """
        context, triples = self.retrieve_context(query)
        result  = self.edit_predict(
            query=query,
            agent_role=self.role,
            graph_context=context,
            current_position=current_position,
            is_final_round="yes" if is_final else "no",
        )
        return result.revised_position.strip(), triples

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
