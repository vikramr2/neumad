from __future__ import annotations

import logging

log = logging.getLogger(__name__)

CHOREOGRAPHED_ROUND_LABELS = {
    1: "Establishing Positions",
    2: "Adversarial Challenge",
    3: "Finding Convergence",
    4: "Mediator Synthesis",
    5: "Reviewing Synthesis",
}

CHOREOGRAPHED_COVARIANCE = {
    1: "moderate",
    2: "low",
    3: "high",
    4: "none",
    5: "moderate-high",
}

# Inlined to avoid module-level circular import with orchestration
_CHOREOGRAPHED_DEBATE_PROMPTS = {
    2: "All agents must disagree on every point. There should be no consensus whatsoever.",
    3: (
        "CONVERGENCE ROUND: Actively seek common ground. "
        "For each point where you previously disagreed, examine whether the other agents' "
        "evidence from their domain supports or modifies your position. "
        "Explicitly identify every claim you can now endorse. "
        "Minimize remaining disagreement — your goal is to find what all agents agree on."
    ),
}


def format_choreographed_history(history: list[dict]) -> str:
    from orchestration import _AGENT_LABELS
    lines = []
    for entry in history:
        if entry["agent"] == "mediator":
            label = "MEDIATOR"
        else:
            label = _AGENT_LABELS.get(entry["agent"], entry["agent"].upper())
        round_label = CHOREOGRAPHED_ROUND_LABELS.get(entry["round"], f"Round {entry['round']}")
        lines.append(f"[{label} — {round_label}]:\n{entry['statement']}\n")
    return "\n".join(lines)


def run_choreographed(query: str, agents, mediator, neuromorphic_mediator: bool = False, status_cb=None) -> dict:
    """If neuromorphic_mediator is True: neuromorphic states its position alone in round
    1 — aiml and neuroscience have no independent stance there. Their round-2 turn is a
    direct reaction to the query plus neuromorphic's stated position, not an independent
    one; they continue through round 3; neuromorphic itself synthesizes round 4 and
    doesn't join the round-5 review of its own synthesis.
    """
    from orchestration import format_context, annotate_transitions

    def _status(msg: str):
        log.info(msg)
        if status_cb:
            status_cb(msg)

    _status(f"Query (choreographed, neuromorphic_mediator={neuromorphic_mediator}): {query}")
    history: list[dict] = []
    agent_refs: dict[str, str] = {}
    agent_local_qbafs: dict[str, dict] = {}
    debating_agents = [a for a in agents if a.name != "neuromorphic"] if neuromorphic_mediator else agents
    opening_agents  = [a for a in agents if a.name == "neuromorphic"] if neuromorphic_mediator else agents

    # ── Round 1: Establish opinion + build Γ+ε QBAFs (covariance: moderate) ─
    # When neuromorphic_mediator is on, only neuromorphic opens here — aiml/neuroscience's
    # first turn is their round-2 reaction to it, not an independent stance of their own.
    _status("  Round 1 — establishing positions…")
    for agent in opening_agents:
        _status(f"  [{agent.name}] round 1 — generating hypothesis…")
        hyp, triples = agent.initial_hypothesis(query)
        _status(f"  [{agent.name}] round 1 — building argument QBAF (Γ+ε)…")
        agent_local_qbafs[agent.name] = agent.build_local_arguments(query, hyp, format_context(triples))
        refs = agent.get_references(triples)
        agent_refs[agent.name] = refs
        history.append({
            "agent":      agent.name,
            "round":      1,
            "statement":  hyp,
            "triples":    triples,
            "references": refs,
            "agreed":     None,
        })
        _status(f"  [{agent.name}] round 1 complete ({len(triples)} triples used)")

    position_history: dict[str, str] = {
        name: qbaf["main_claim"] for name, qbaf in agent_local_qbafs.items()
    }

    # ── Round 2: Adversarial attack (covariance: low) ─────────────────────
    _status("  Round 2 — adversarial challenge…")
    round2_entries: list[tuple] = []
    for agent in debating_agents:
        _status(f"  [{agent.name}] round 2 — attacking…")
        response, agreed = agent.debate_response(
            query,
            format_choreographed_history(history),
            _CHOREOGRAPHED_DEBATE_PROMPTS[2],
        )
        entry = {
            "agent":      agent.name,
            "round":      2,
            "statement":  response,
            "triples":    [],
            "references": agent_refs.get(agent.name, ""),
            "agreed":     agreed,
        }
        history.append(entry)
        round2_entries.append((agent, entry))

    if neuromorphic_mediator:
        # Debaters' first turn is their baseline position (a reaction to neuromorphic,
        # not a prior stance of their own) — build their QBAF from it here, the same
        # role round 1 plays for everyone in the non-toggle path, and skip position
        # tracking this round since there's nothing yet to compare against.
        for agent, entry in round2_entries:
            _status(f"  [{agent.name}] round 2 — building argument QBAF (Γ+ε)…")
            context, triples = agent.retrieve_context(query)
            entry["triples"] = triples
            agent_local_qbafs[agent.name] = agent.build_local_arguments(query, entry["statement"], context)
            refs = agent.get_references(triples)
            agent_refs[agent.name] = refs
            entry["references"] = refs
            position_history[agent.name] = agent_local_qbafs[agent.name]["main_claim"]
    else:
        _status("  Round 2 — tracking position transitions…")
        annotate_transitions(history, 2, query, agents, mediator, position_history)

    # ── Round 3: Convergence (covariance: high) ────────────────────────────
    _status("  Round 3 — finding convergence…")
    for agent in debating_agents:
        _status(f"  [{agent.name}] round 3 — converging…")
        response, agreed = agent.debate_response(
            query,
            format_choreographed_history(history),
            _CHOREOGRAPHED_DEBATE_PROMPTS[3],
        )
        history.append({
            "agent":      agent.name,
            "round":      3,
            "statement":  response,
            "triples":    [],
            "references": agent_refs[agent.name],
            "agreed":     agreed,
        })

    _status("  Round 3 — tracking position transitions…")
    annotate_transitions(history, 3, query, agents, mediator, position_history)

    # ── Round 4: Synthesis using round-1 KG-grounded QBAFs ────────────────
    # Per-agent latest-statement lookup (not "round 3 only") since neuromorphic_mediator
    # leaves neuromorphic frozen at round 1 while the debaters progress to round 3.
    def _latest_statement(name: str) -> str:
        entries = [e for e in history if e["agent"] == name]
        return max(entries, key=lambda e: e["round"])["statement"]

    agent_data = {
        agent.name: {"statement": _latest_statement(agent.name), "local_qbaf": agent_local_qbafs[agent.name]}
        for agent in agents
    }

    if neuromorphic_mediator:
        neuromorphic = next(a for a in agents if a.name == "neuromorphic")
        _status(f"  Round 4 — [{neuromorphic.name}] mediating synthesis…")
        synthesis_result = mediator.mediate_as_agent(
            query,
            neuromorphic.role,
            agent_data["neuromorphic"]["statement"],
            format_choreographed_history(history),
            agent_data=agent_data,
        )
        synthesis_speaker = "neuromorphic"
    else:
        _status("  Round 4 — mediator synthesizing…")
        synthesis_result = mediator.extract_answer(
            query,
            format_choreographed_history(history),
            agent_data=agent_data,
        )
        synthesis_speaker = "mediator"
    synthesis = synthesis_result["text"]
    history.append({
        "agent":      synthesis_speaker,
        "round":      4,
        "statement":  synthesis,
        "triples":    [],
        "references": "",
        "agreed":     None,
    })

    # ── Round 5: Do you agree? (covariance: moderate-high) ────────────────
    # When neuromorphic mediated round 4, it doesn't also review its own synthesis.
    _status("  Round 5 — agents reviewing synthesis…")
    history_str = format_choreographed_history(history)
    for agent in debating_agents:
        _status(f"  [{agent.name}] round 5 — reviewing synthesis…")
        response, agreed = agent.review_synthesis(query, synthesis, history_str)
        history.append({
            "agent":      agent.name,
            "round":      5,
            "statement":  response,
            "triples":    [],
            "references": agent_refs[agent.name],
            "agreed":     agreed,
        })

    _status("  Round 5 — tracking position transitions…")
    annotate_transitions(history, 5, query, agents, mediator, position_history)

    return {
        "query":                 query,
        "mode":                  "choreographed",
        "neuromorphic_mediator": neuromorphic_mediator,
        "debate_history":       history,
        "final_hypothesis":    synthesis,
        "argumentation_graph": synthesis_result["graph"],
    }
