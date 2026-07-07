"""Per-message rendering: dispatches to render_debate/render_graph, handles the
artifact-save button, and renders the full chat message list."""

from __future__ import annotations

import streamlit as st

from history_store import _save_artifacts
from orchestration import CHOREOGRAPHED_COVARIANCE, CHOREOGRAPHED_ROUND_LABELS, _AGENT_LABELS
from paths import ROOT
from render_debate import (
    _AGENT_COLORS,
    _render_agent_columns,
    _render_position_trajectory,
    _render_rotation_sequence,
    _transition_summary,
)
from render_graph import _LABEL_RE, _render_argumentation_graph, _render_synthesis_with_labels


def render_result_in_chat(result: dict):
    """Render a full pipeline result inside an st.chat_message block."""
    graph = result.get("argumentation_graph")
    text  = result["final_hypothesis"]

    if graph and _LABEL_RE.search(text):
        _render_synthesis_with_labels(text, graph)
    elif graph:
        st.markdown(text)
    else:
        st.markdown(text)

    # Expandable detail section
    mode = result["mode"]
    if mode == "synthesis":
        provenance = result.get("provenance")
        if provenance == "blended":
            st.caption("🔗 Provenance: a genuine blend — no single agent's position dominates")
        elif provenance:
            st.caption(f"🔗 Provenance: primarily reflects **{_AGENT_LABELS.get(provenance, provenance)}**")

        with st.expander("Agent hypotheses", expanded=False):
            cols = st.columns(3)
            for col, name in zip(cols, ["neuroscience", "aiml", "neuromorphic"]):
                data = result["agent_hypotheses"][name]
                with col:
                    st.markdown(
                        f"<span style='color:{_AGENT_COLORS[name]};font-weight:700'>"
                        f"{_AGENT_LABELS[name]}</span>",
                        unsafe_allow_html=True,
                    )
                    st.write(data["statement"])
                    refs = data.get("references", "").strip()
                    if refs:
                        with st.expander("References", expanded=False):
                            for line in refs.splitlines():
                                if line.strip():
                                    st.markdown(f"- {line.strip()}")

    elif mode == "adversarial":
        rounds: dict[int, list[dict]] = {}
        for entry in result["debate_history"]:
            rounds.setdefault(entry["round"], []).append(entry)

        label = (f"Debate detail — {result['rounds_completed']} round(s), "
                 f"level {result['debate_level']}")
        with st.expander(label, expanded=False):
            _render_position_trajectory(result["debate_history"], list(_AGENT_LABELS.keys()))
            st.divider()
            for round_num, entries in sorted(rounds.items()):
                if round_num == 0:
                    st.markdown("**Round 0 — Initial Positions**")
                else:
                    agreed_count = sum(1 for e in entries if e.get("agreed") is True)
                    trans_str = _transition_summary(entries)
                    st.markdown(f"**Round {round_num}** ({agreed_count}/{len(entries)} agree{trans_str})")
                _render_agent_columns(entries, show_agreement=(round_num > 0))
                if round_num < max(rounds):
                    st.divider()

    elif mode == "choreographed":
        c_rounds: dict[int, list[dict]] = {}
        for entry in result["debate_history"]:
            c_rounds.setdefault(entry["round"], []).append(entry)

        with st.expander("Choreographed debate — 5 rounds", expanded=False):
            _render_position_trajectory(result["debate_history"], list(_AGENT_LABELS.keys()))
            st.divider()
            for round_num, entries in sorted(c_rounds.items()):
                round_label = CHOREOGRAPHED_ROUND_LABELS.get(round_num, f"Round {round_num}")
                covariance  = CHOREOGRAPHED_COVARIANCE.get(round_num, "")
                cov_badge   = f" *(covariance: {covariance})*" if covariance != "none" else ""

                if round_num == 4:
                    st.markdown(f"**Round 4 — {round_label}**")
                    st.markdown(entries[0]["statement"])
                else:
                    show_agree = round_num > 1
                    agreed_count = sum(1 for e in entries if e.get("agreed") is True)
                    trans_str = _transition_summary(entries) if show_agree else ""
                    agree_str = f" · {agreed_count}/{len(entries)} agree{trans_str}" if show_agree else ""
                    st.markdown(f"**Round {round_num} — {round_label}**{cov_badge}{agree_str}")
                    _render_agent_columns(entries, show_agreement=show_agree)

                if round_num < max(c_rounds):
                    st.divider()

    elif mode == "rotation":
        label = f"Rotation detail — {result['n_rotations']} rotation(s)"
        with st.expander(label, expanded=False):
            _render_rotation_sequence(result["debate_history"], result["n_rotations"])

    elif mode in ("neukrag", "neukrag-inter"):
        triples = result.get("triples", [])
        label   = "NeuKRAG-inter" if mode == "neukrag-inter" else "NeuKRAG"
        with st.expander(f"{label} — KG context ({len(triples)} triples)", expanded=False):
            refs = result.get("references", "").strip()
            if refs:
                st.markdown("**References**")
                for line in refs.splitlines():
                    if line.strip():
                        st.markdown(f"- {line.strip()}")
                st.divider()
            if triples:
                st.markdown("**Triples retrieved**")
                for t in triples:
                    st.markdown(
                        f"`{t['h']}` &nbsp;—[{t['r']}]→&nbsp; `{t['t']}`",
                        unsafe_allow_html=True,
                    )

    if graph:
        with st.expander("Argumentation graph", expanded=False):
            _render_argumentation_graph(graph)


def _render_save_button(result: dict, msg_idx: int):
    """Save-artifacts button + inline artifact-name prompt for one assistant message."""
    if st.session_state["pending_save"] == msg_idx:
        name_key = f"artifact_name_{msg_idx}"
        artifact_name = st.text_input(
            "Artifact name", key=name_key,
            placeholder="e.g. spiking_net_run1",
        )
        col_save, col_cancel = st.columns([1, 1])
        with col_save:
            if st.button("Save", key=f"save_confirm_{msg_idx}", type="primary"):
                name = (artifact_name or "").strip()
                if name:
                    saved_path = _save_artifacts(result, name)
                    st.session_state["pending_save"] = None
                    st.toast(f"Saved to {saved_path.relative_to(ROOT)}/", icon="💾")
                    st.rerun()
        with col_cancel:
            if st.button("Cancel", key=f"save_cancel_{msg_idx}"):
                st.session_state["pending_save"] = None
                st.rerun()
    else:
        if st.button("💾 Save artifacts", key=f"save_btn_{msg_idx}"):
            st.session_state["pending_save"] = msg_idx
            st.rerun()


def render_messages(messages: list[dict], *, read_only: bool = False):
    """Render a list of chat messages."""
    for i, msg in enumerate(messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.write(msg["content"])
            elif msg.get("result"):
                render_result_in_chat(msg["result"])
                if not read_only:
                    _render_save_button(msg["result"], i)
            else:
                st.markdown(msg["content"])
