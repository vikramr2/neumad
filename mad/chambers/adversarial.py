from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def run_adversarial(
    query: str,
    agents,
    mediator,
    max_rounds: int,
    debate_level: int,
    status_cb=None,
) -> dict:
    from orchestration import format_debate_history, DEBATE_LEVEL_PROMPTS

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
                "references": agent_refs[agent.name],
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
