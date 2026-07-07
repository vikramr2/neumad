"""Sidebar: new-debate button, mode/parameter settings, conversation history list."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from history_store import _save_history
from orchestration import DEBATE_LEVEL_PROMPTS

_MODE_LABELS = {
    "synthesis":     "🤝 Synthesis",
    "adversarial":   "⚔️ Adversarial",
    "choreographed": "🎼 Choreographed",
    "neukrag":       "🔬 NeuKRAG",
    "neukrag-inter": "🌐 NeuKRAG-inter",
}


def render_sidebar() -> dict:
    """Render the sidebar and return the active run settings (mode, rounds, level,
    k_hops, max_triples)."""
    with st.sidebar:
        st.title("🧠 NeuMAD")

        # --- New Debate button ---
        if st.button("＋ New Debate", use_container_width=True, type="primary"):
            # Archive current conversation if non-empty
            if st.session_state["messages"]:
                first_user = next(
                    (m["content"] for m in st.session_state["messages"] if m["role"] == "user"),
                    "Untitled",
                )
                st.session_state["conv_history"].insert(0, {
                    "title":     first_user[:60] + ("…" if len(first_user) > 60 else ""),
                    "messages":  list(st.session_state["messages"]),
                    "timestamp": datetime.now().strftime("%b %d %H:%M"),
                })
            st.session_state["messages"]         = []
            st.session_state["active_synthesis"] = None
            st.session_state["viewing_history"]  = False
            _save_history()
            st.rerun()

        st.divider()

        # --- Settings ---
        st.subheader("Settings")

        mode = st.radio(
            "Mode",
            options=["synthesis", "adversarial", "choreographed", "neukrag", "neukrag-inter"],
            format_func=_MODE_LABELS.get,
            help=(
                "Synthesis: agents generate independently, mediator integrates.\n"
                "Adversarial: MAD-style debate with rebuttals and adaptive break.\n"
                "Choreographed: fixed 5-round arc — establish → attack → converge → synthesize → review.\n"
                "NeuKRAG: single-agent hypothesis from the neuromorphic KG only.\n"
                "NeuKRAG-inter: single-agent hypothesis from the unified cross-domain KG."
            ),
        )

        if mode == "adversarial":
            debate_rounds = st.slider("Max Rounds", 1, 5, 3)
            debate_level  = st.slider(
                "Tit-for-tat Level", 0, 3, 2,
                help="\n".join(f"{k}: {v}" for k, v in DEBATE_LEVEL_PROMPTS.items()),
            )
        else:
            debate_rounds, debate_level = 3, 2

        k_hops      = st.slider("K-hops", 1, 3, 2)
        max_triples = st.slider("Max Triples / Agent", 10, 60, 40, step=5)

        # --- Conversation history ---
        if st.session_state["conv_history"]:
            st.divider()
            st.subheader("History")
            # CSS: hide the × delete button until the row is hovered
            st.markdown("""
<style>
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]
    > div:last-child button {
    opacity: 0;
    border-radius: 50% !important;
    width: 26px !important;
    min-width: 26px !important;
    height: 26px !important;
    min-height: 26px !important;
    padding: 0 !important;
    font-size: 16px !important;
    line-height: 1 !important;
    transition: opacity 0.15s ease;
}
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:hover
    > div:last-child button {
    opacity: 1;
}
</style>
""", unsafe_allow_html=True)
            for i, conv in enumerate(st.session_state["conv_history"]):
                col_btn, col_del = st.columns([7, 1])
                with col_btn:
                    if st.button(conv["title"], key=f"hist_{i}", use_container_width=True):
                        st.session_state["viewing_history"]  = True
                        st.session_state["history_snapshot"] = conv["messages"]
                        st.rerun()
                with col_del:
                    if st.button("×", key=f"del_{i}", help="Delete from history"):
                        st.session_state["conv_history"].pop(i)
                        _save_history()
                        st.rerun()

    return {
        "mode":          mode,
        "debate_rounds": debate_rounds,
        "debate_level":  debate_level,
        "k_hops":        k_hops,
        "max_triples":   max_triples,
    }
