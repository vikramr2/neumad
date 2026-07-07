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


def run_choreographed(query: str, agents, mediator, status_cb=None) -> dict:
    from orchestration import format_context, annotate_transitions

    def _status(msg: str):
        log.info(msg)
        if status_cb:
            status_cb(msg)

    _status(f"Query (choreographed): {query}")
    history: list[dict] = []
    agent_refs: dict[str, str] = {}
    agent_local_qbafs: dict[str, dict] = {}

    # ── Round 1: Establish opinion + build Γ+ε QBAFs (covariance: moderate) ─
    _status("  Round 1 — establishing positions…")
    for agent in agents:
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
    for agent in agents:
        _status(f"  [{agent.name}] round 2 — attacking…")
        response, agreed = agent.debate_response(
            query,
            format_choreographed_history(history),
            _CHOREOGRAPHED_DEBATE_PROMPTS[2],
        )
        history.append({
            "agent":      agent.name,
            "round":      2,
            "statement":  response,
            "triples":    [],
            "references": agent_refs[agent.name],
            "agreed":     agreed,
        })

    _status("  Round 2 — tracking position transitions…")
    annotate_transitions(history, 2, query, agents, mediator, position_history)

    # ── Round 3: Convergence (covariance: high) ────────────────────────────
    _status("  Round 3 — finding convergence…")
    for agent in agents:
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

    # ── Round 4: Mediator synthesis using round-1 KG-grounded QBAFs ──────
    _status("  Round 4 — mediator synthesizing…")
    round3_stmts = {
        e["agent"]: e["statement"]
        for e in history
        if e["round"] == 3 and e["agent"] != "mediator"
    }
    agent_data = {
        name: {"statement": stmt, "local_qbaf": agent_local_qbafs[name]}
        for name, stmt in round3_stmts.items()
    }
    synthesis_result = mediator.extract_answer(
        query,
        format_choreographed_history(history),
        agent_data=agent_data,
    )
    synthesis = synthesis_result["text"]
    history.append({
        "agent":      "mediator",
        "round":      4,
        "statement":  synthesis,
        "triples":    [],
        "references": "",
        "agreed":     None,
    })

    # ── Round 5: Do you agree? (covariance: moderate-high) ────────────────
    _status("  Round 5 — agents reviewing synthesis…")
    history_str = format_choreographed_history(history)
    for agent in agents:
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
        "query":                query,
        "mode":                 "choreographed",
        "debate_history":       history,
        "final_hypothesis":    synthesis,
        "argumentation_graph": synthesis_result["graph"],
    }
