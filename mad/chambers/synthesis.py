from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def run_synthesis(query: str, agents, mediator, status_cb=None) -> dict:
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
        refs = agent.get_references(triples)
        agent_hypotheses[agent.name] = {
            "statement":  hyp,
            "triples":    triples,
            "references": refs,
        }
        _status(f"  [{agent.name}] done ({len(triples)} triples used)")

    _status("  Mediator building argumentation graph and synthesizing…")
    synthesis_result = mediator.synthesize(
        query,
        {name: d["statement"] for name, d in agent_hypotheses.items()},
    )

    return {
        "query":                query,
        "mode":                 "synthesis",
        "agent_hypotheses":     agent_hypotheses,
        "final_hypothesis":    synthesis_result["text"],
        "argumentation_graph": synthesis_result["graph"],
    }