from __future__ import annotations

import logging
from difflib import SequenceMatcher

log = logging.getLogger(__name__)

# Agent order after the anchor within each rotation.
_CYCLE_ORDER = ["aiml", "neuroscience"]
_ANCHOR      = "neuromorphic"

# ---------------------------------------------------------------------------
# Span-level provenance via diffing.
#
# Rotation has no QBAF, but it does have something a QBAF can't give you for free:
# every round is a full rewrite of the *same* document, so diffing round N against
# round N-1 tells you exactly which lines that agent's edit introduced, with no LLM
# call and no risk of a model mis-attributing its own text. We reuse the existing
# <label agent="..." node_id="...">...</label> hover-card format (node_id = round
# number instead of a graph-node id) so the UI needs zero changes to pick it up.
# ---------------------------------------------------------------------------

_Span = tuple[str, str, int]  # (text, agent_name, round_num)


def _tokenize_lines(text: str) -> list[str]:
    """Split into lines (keeping line endings). Line-level, not word-level, so a
    diff boundary can't land in the middle of markdown syntax like **bold** or a
    ## header and break rendering."""
    lines = text.splitlines(keepends=True)
    return lines if lines else [text]


def _attribute_edit(prev_spans: list[_Span], new_text: str, agent_name: str, round_num: int) -> list[_Span]:
    """Diff new_text against the previously-attributed text. Unchanged lines keep
    their prior attribution; changed/added lines are attributed to this edit."""
    prev_lines = [line for line, _, _ in prev_spans]
    new_lines  = _tokenize_lines(new_text)
    sm = SequenceMatcher(None, prev_lines, new_lines, autojunk=False)
    spans: list[_Span] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            spans.extend(prev_spans[i1:i2])
        elif tag in ("replace", "insert"):
            spans.extend((line, agent_name, round_num) for line in new_lines[j1:j2])
        # "delete": lines removed by this edit — nothing carried forward
    return spans


def _coalesce_spans(spans: list[_Span]) -> list[_Span]:
    """Merge adjacent lines that share the same (agent, round) attribution."""
    coalesced: list[_Span] = []
    for text, agent, rnd in spans:
        if coalesced and coalesced[-1][1] == agent and coalesced[-1][2] == rnd:
            coalesced[-1] = (coalesced[-1][0] + text, agent, rnd)
        else:
            coalesced.append((text, agent, rnd))
    return coalesced


def _label_spans(spans: list[_Span]) -> str:
    """Wrap each attributed span in the same <label> format the synthesis/debate
    hover feature already parses, so render_result_in_chat picks it up unchanged."""
    parts = []
    for text, agent, rnd in spans:
        if not text.strip():
            parts.append(text)
        else:
            parts.append(f'<label agent="{agent}" node_id="{rnd}">{text}</label>')
    return "".join(parts)


def _spans_to_graph(spans: list[_Span]) -> dict:
    """One node per contributing round — same shape _render_argumentation_graph and
    _node_popup_card expect, just without DFQuAD strengths (they're None-safe)."""
    nodes: dict[int, dict] = {}
    for text, agent, rnd in spans:
        if not text.strip():
            continue
        if rnd not in nodes:
            nodes[rnd] = {
                "id":        rnd,
                "expert":    agent,
                "type":      "main_argument" if rnd == 0 else "edit",
                "statement": text.strip(),
            }
        else:
            nodes[rnd]["statement"] += " […] " + text.strip()
    return {"nodes": list(nodes.values()), "edges": []}


def run_rotation(
    query: str,
    agents,
    mediator,
    n_rotations: int = 1,
    status_cb=None,
) -> dict:
    """Round-robin position editing.

    The neuromorphic agent drafts the initial position. It is then passed in turn to
    the aiml and neuroscience agents, each of which revises it from their domain
    perspective, before returning to the neuromorphic agent — that's one rotation.
    After n_rotations, the neuromorphic agent's final revision is the answer.

    Position provenance (transition_type/adopted_peer) is tracked the same way as the
    debate chambers: each agent's edit is classified against its own last contribution
    and the other two agents' most recent contributions.
    """
    from orchestration import annotate_transitions

    def _status(msg: str):
        log.info(msg)
        if status_cb:
            status_cb(msg)

    agent_by_name = {a.name: a for a in agents}
    anchor = agent_by_name[_ANCHOR]
    cycle  = [agent_by_name[name] for name in _CYCLE_ORDER]

    _status(f"Query (rotation, n={n_rotations}): {query}")
    history: list[dict] = []
    agent_refs: dict[str, str] = {}

    _status(f"  [{anchor.name}] drafting initial position…")
    position, triples = anchor.initial_hypothesis(query)
    agent_refs[anchor.name] = anchor.get_references(triples)
    history.append({
        "agent":      anchor.name,
        "round":      0,
        "statement":  position,
        "triples":    triples,
        "references": agent_refs[anchor.name],
        "agreed":     None,
    })

    position_history: dict[str, str] = {anchor.name: anchor.extract_main_claim(query, position)}
    spans: list[_Span] = [(line, anchor.name, 0) for line in _tokenize_lines(position)]

    round_num = 0
    for rotation in range(1, n_rotations + 1):
        for agent in cycle:
            round_num += 1
            _status(f"  [{agent.name}] rotation {rotation}/{n_rotations} — editing position…")
            position, triples = agent.edit_position(query, position)
            agent_refs[agent.name] = agent.get_references(triples)
            history.append({
                "agent":      agent.name,
                "round":      round_num,
                "statement":  position,
                "triples":    triples,
                "references": agent_refs[agent.name],
                "agreed":     None,
            })
            annotate_transitions(history, round_num, query, agents, mediator, position_history)
            spans = _attribute_edit(spans, position, agent.name, round_num)

        round_num += 1
        is_final = rotation == n_rotations
        _status(f"  [{anchor.name}] rotation {rotation}/{n_rotations} — "
                f"{'finalizing' if is_final else 'editing'} position…")
        position, triples = anchor.edit_position(query, position, is_final=is_final)
        agent_refs[anchor.name] = anchor.get_references(triples)
        history.append({
            "agent":      anchor.name,
            "round":      round_num,
            "statement":  position,
            "triples":    triples,
            "references": agent_refs[anchor.name],
            "agreed":     None,
        })
        annotate_transitions(history, round_num, query, agents, mediator, position_history)
        spans = _attribute_edit(spans, position, anchor.name, round_num)

    coalesced = _coalesce_spans(spans)

    return {
        "query":                query,
        "mode":                 "rotation",
        "n_rotations":          n_rotations,
        "debate_history":       history,
        "final_hypothesis":    _label_spans(coalesced),
        "argumentation_graph": _spans_to_graph(coalesced),
    }
