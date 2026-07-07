"""Per-round agent-column rendering: agreement badges and position-transition tracking."""

from __future__ import annotations

import streamlit as st

from orchestration import _AGENT_LABELS

_AGENT_COLORS = {
    "neuroscience": "#1f6aa5",
    "aiml":         "#2d8a4e",
    "neuromorphic": "#8b4513",
}


def _agreement_badge(agreed: bool | None) -> str:
    if agreed is None:
        return ""
    return " 🟢" if agreed else " 🔴"


def _transition_icon(transition_type: str | None, adopted_peer: str | None) -> str:
    if transition_type == "unchanged":
        return "↔"
    if transition_type == "independent_revision":
        return "✦"
    if transition_type == "peer_aligned":
        short = _AGENT_LABELS.get(adopted_peer, adopted_peer or "?").replace(" Specialist", "")
        return f"→{short}"
    return "·"


def _transition_badge(entry: dict) -> str:
    if entry.get("transition_type") is None:
        return ""
    return " " + _transition_icon(entry.get("transition_type"), entry.get("adopted_peer"))


def _render_position_trajectory(history: list[dict], agent_names: list[str]):
    """One row per agent showing its transition_type/adopted_peer across rounds."""
    by_agent: dict[str, dict[int, dict]] = {name: {} for name in agent_names}
    for entry in history:
        if entry["agent"] in by_agent and entry.get("transition_type") is not None:
            by_agent[entry["agent"]][entry["round"]] = entry

    if not any(by_agent.values()):
        return

    st.markdown("**Position trajectory**")
    for name in agent_names:
        rounds = by_agent[name]
        if not rounds:
            continue
        cells = " · ".join(
            _transition_icon(e.get("transition_type"), e.get("adopted_peer"))
            for _, e in sorted(rounds.items())
        )
        st.markdown(
            f"<span style='color:{_AGENT_COLORS[name]};font-weight:700'>{_AGENT_LABELS[name]}</span>: {cells}",
            unsafe_allow_html=True,
        )
    st.caption("↔ unchanged · →Name restated that peer's claim · ✦ independent revision")


def _transition_summary(entries: list[dict]) -> str:
    counted = [e.get("transition_type") for e in entries if e.get("transition_type")]
    if not counted:
        return ""
    peer_aligned = counted.count("peer_aligned")
    independent  = counted.count("independent_revision")
    parts = []
    if peer_aligned:
        parts.append(f"{peer_aligned} peer-aligned")
    if independent:
        parts.append(f"{independent} independent")
    return f" · {', '.join(parts)}" if parts else ""


def _render_rotation_sequence(history: list[dict], n_rotations: int):
    """Render rotation-mode history as a sequential pipeline: one agent edit per round,
    in order, rather than per-round columns (only one agent acts per round here)."""
    entries = sorted(history, key=lambda e: e["round"])
    agent_names = list(dict.fromkeys(e["agent"] for e in entries))  # first-seen order
    _render_position_trajectory(history, agent_names)
    st.divider()

    st.markdown(f"**{n_rotations} rotation(s)** — position passed in sequence, edited by each agent in turn")
    for i, entry in enumerate(entries):
        name  = entry["agent"]
        label = _AGENT_LABELS.get(name, name)
        color = _AGENT_COLORS.get(name, "#666666")
        badge = _transition_badge(entry)
        step_label = "Initial position" if entry["round"] == 0 else f"Round {entry['round']}"
        st.markdown(
            f"**{step_label}** — <span style='color:{color};font-weight:700'>{label}{badge}</span>",
            unsafe_allow_html=True,
        )
        st.write(entry["statement"])
        refs = entry.get("references", "").strip()
        if refs:
            with st.expander("References", expanded=False):
                for line in refs.splitlines():
                    if line.strip():
                        st.markdown(f"- {line.strip()}")
        if i < len(entries) - 1:
            st.divider()


def _render_agent_columns(entries: list[dict], *, show_agreement: bool):
    cols = st.columns(3)
    for col, entry in zip(cols, entries):
        name  = entry["agent"]
        label = _AGENT_LABELS[name]
        badge = _agreement_badge(entry.get("agreed")) if show_agreement else ""
        badge += _transition_badge(entry) if show_agreement else ""
        with col:
            st.markdown(
                f"<span style='color:{_AGENT_COLORS[name]};font-weight:700'>"
                f"{label}{badge}</span>",
                unsafe_allow_html=True,
            )
            st.write(entry["statement"])
            refs = entry.get("references", "").strip()
            if refs:
                with st.expander("References", expanded=False):
                    for line in refs.splitlines():
                        if line.strip():
                            st.markdown(f"- {line.strip()}")
