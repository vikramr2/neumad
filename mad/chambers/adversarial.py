from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def run_adversarial(
    query: str,
    agents,
    mediator,
    max_rounds: int,
    debate_level: int,
    neuromorphic_mediator: bool = False,
    status_cb=None,
) -> dict:
    """If neuromorphic_mediator is True: neuromorphic states its position alone before
    the debate starts — aiml and neuroscience have no independent round-0 stance. Their
    round-1 turn is a direct reaction to the query plus neuromorphic's stated position,
    not an independent one; neuromorphic itself — not the neutral Mediator — then
    synthesizes the final answer.
    """
    from orchestration import (
        format_debate_history, DEBATE_LEVEL_PROMPTS, format_context, annotate_transitions,
    )

    def _status(msg: str):
        log.info(msg)
        if status_cb:
            status_cb(msg)

    _status(f"Query (adversarial, level={debate_level}, max_rounds={max_rounds}, "
             f"neuromorphic_mediator={neuromorphic_mediator}): {query}")
    level_prompt = DEBATE_LEVEL_PROMPTS[debate_level]
    history: list[dict] = []
    debating_agents = [a for a in agents if a.name != "neuromorphic"] if neuromorphic_mediator else agents
    opening_agents  = [a for a in agents if a.name == "neuromorphic"] if neuromorphic_mediator else agents

    # Round 0 — initial KG-grounded position(s) + Γ+ε QBAF(s). When neuromorphic_mediator
    # is on, only neuromorphic opens — aiml/neuroscience's first turn is their round-1
    # reaction to it (below), not an independent stance of their own.
    agent_refs: dict[str, str] = {}
    agent_local_qbafs: dict[str, dict] = {}
    for agent in opening_agents:
        _status(f"  [{agent.name}] round 0 — generating hypothesis…")
        hyp, triples = agent.initial_hypothesis(query)
        _status(f"  [{agent.name}] round 0 — building argument QBAF (Γ+ε)…")
        agent_local_qbafs[agent.name] = agent.build_local_arguments(query, hyp, format_context(triples))
        _status(f"  [{agent.name}] round 0 — extracting references…")
        refs = agent.get_references(triples)
        agent_refs[agent.name] = refs
        history.append({
            "agent":      agent.name,
            "round":      0,
            "statement":  hyp,
            "triples":    triples,
            "references": refs,
            "agreed":     None,
        })
        _status(f"  [{agent.name}] round 0 complete ({len(triples)} triples used)")

    position_history: dict[str, str] = {
        name: qbaf["main_claim"] for name, qbaf in agent_local_qbafs.items()
    }

    rounds_completed = 0
    for round_num in range(1, max_rounds + 1):
        _status(f"  Debate round {round_num}/{max_rounds}")
        history_str = format_debate_history(history)

        round_entries: list[tuple] = []
        for agent in debating_agents:
            _status(f"  [{agent.name}] round {round_num} — responding…")
            response, agreed = agent.debate_response(query, history_str, level_prompt)
            entry = {
                "agent":      agent.name,
                "round":      round_num,
                "statement":  response,
                "triples":    [],
                "references": agent_refs.get(agent.name, ""),
                "agreed":     agreed,
            }
            history.append(entry)
            round_entries.append((agent, entry))

        # A debater's very first turn (round 1, only when neuromorphic_mediator is on)
        # is their baseline position — a reaction to neuromorphic, not to a prior stance
        # of their own. Build their QBAF from it here, the same role round 0 plays for
        # everyone in the non-toggle path, and skip position tracking this round since
        # there's nothing yet to compare against.
        if neuromorphic_mediator and round_num == 1:
            for agent, entry in round_entries:
                _status(f"  [{agent.name}] round {round_num} — building argument QBAF (Γ+ε)…")
                context, triples = agent.retrieve_context(query)
                entry["triples"] = triples
                agent_local_qbafs[agent.name] = agent.build_local_arguments(query, entry["statement"], context)
                refs = agent.get_references(triples)
                agent_refs[agent.name] = refs
                entry["references"] = refs
                position_history[agent.name] = agent_local_qbafs[agent.name]["main_claim"]
        else:
            _status(f"  Round {round_num} — tracking position transitions…")
            annotate_transitions(history, round_num, query, agents, mediator, position_history)

        rounds_completed = round_num
        history_str = format_debate_history(history)

        # Discriminative mode — adaptive break (MAD §2)
        concluded, reason = mediator.can_conclude(query, history_str)
        _status(f"  Judge: concluded={concluded} — {reason}")
        if concluded:
            _status("  Adaptive break: debate concluded early")
            break

    # Build graph from each agent's latest statement using its own KG-grounded QBAF.
    # A per-agent lookup (rather than "the final round") since neuromorphic_mediator
    # leaves neuromorphic frozen at round 0 while the debaters progress further.
    def _latest_statement(name: str) -> str:
        entries = [e for e in history if e["agent"] == name]
        return max(entries, key=lambda e: e["round"])["statement"]

    agent_data = {
        agent.name: {"statement": _latest_statement(agent.name), "local_qbaf": agent_local_qbafs[agent.name]}
        for agent in agents
    }

    if neuromorphic_mediator:
        neuromorphic = next(a for a in agents if a.name == "neuromorphic")
        _status(f"  [{neuromorphic.name}] mediating final answer…")
        answer_result = mediator.mediate_as_agent(
            query,
            neuromorphic.role,
            agent_data["neuromorphic"]["statement"],
            format_debate_history(history),
            agent_data=agent_data,
        )
    else:
        _status("  Mediator extracting final answer…")
        answer_result = mediator.extract_answer(
            query,
            format_debate_history(history),
            agent_data=agent_data,
        )

    return {
        "query":                 query,
        "mode":                  "adversarial",
        "debate_level":          debate_level,
        "rounds_completed":      rounds_completed,
        "neuromorphic_mediator": neuromorphic_mediator,
        "debate_history":        history,
        "final_hypothesis":     answer_result["text"],
        "argumentation_graph":  answer_result["graph"],
    }
