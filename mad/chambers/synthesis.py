from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def run_synthesis(query: str, agents, mediator, status_cb=None) -> dict:
    from orchestration import format_context

    def _status(msg: str):
        log.info(msg)
        if status_cb:
            status_cb(msg)

    _status(f"Query (synthesis): {query}")
    agent_hypotheses: dict[str, dict] = {}

    for agent in agents:
        _status(f"  [{agent.name}] generating hypothesis…")
        hyp, triples = agent.initial_hypothesis(query)
        _status(f"  [{agent.name}] building argument QBAF (Γ+ε)…")
        local_qbaf = agent.build_local_arguments(query, hyp, format_context(triples))
        _status(f"  [{agent.name}] extracting references…")
        refs = agent.get_references(triples)
        agent_hypotheses[agent.name] = {
            "statement":  hyp,
            "triples":    triples,
            "references": refs,
            "local_qbaf": local_qbaf,
        }
        _status(f"  [{agent.name}] done ({len(triples)} triples used)")

    _status("  Mediator merging QBAFs and synthesizing…")
    synthesis_result = mediator.synthesize(
        query,
        {name: {"statement": d["statement"], "local_qbaf": d["local_qbaf"]}
         for name, d in agent_hypotheses.items()},
    )

    _status("  Mediator attributing synthesis provenance…")
    agent_claims = {name: d["local_qbaf"]["main_claim"] for name, d in agent_hypotheses.items()}
    provenance = mediator.classify_synthesis_provenance(query, synthesis_result["text"], agent_claims)

    return {
        "query":                query,
        "mode":                 "synthesis",
        "agent_hypotheses":     agent_hypotheses,
        "provenance":           provenance,
        "final_hypothesis":    synthesis_result["text"],
        "argumentation_graph": synthesis_result["graph"],
    }
