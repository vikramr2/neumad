"""Sidebar: new-debate button, mode/parameter settings, conversation history list."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from history_store import _save_history
from orchestration import DEBATE_LEVEL_PROMPTS

_TOP_MODE_LABELS = {
    "debate":    "⚔️ Debate",
    "synthesis": "🤝 Synthesis",
    "neukrag":   "🔬 NeuKRAG",
}

_TOP_MODE_HELP = (
    "Debate: agents challenge and rebut each other across multiple rounds.\n"
    "Synthesis: agents generate independently, mediator integrates.\n"
    "NeuKRAG: single-agent hypothesis grounded directly in a knowledge graph."
)

# Sub-modes offered under each top-level mode, in display order.
_SUB_MODES = {
    "debate":    ["adversarial", "choreographed"],
    "synthesis": ["synthesis", "rotation"],
    "neukrag":   ["neukrag", "neukrag-inter"],
}

_SUB_MODE_LABELS = {
    "adversarial":   "⚔️ Adversarial",
    "choreographed": "🎼 Choreographed",
    "synthesis":     "🤝 Synthesis",
    "rotation":      "🔄 Rotation",
    "neukrag":       "🔬 NeuKRAG",
    "neukrag-inter": "🌐 NeuKRAG-inter",
}

_SUB_MODE_SELECT_LABEL = {
    "debate":    "Debate style",
    "synthesis": "Synthesis style",
    "neukrag":   "KG scope",
}

_SUB_MODE_HELP = {
    "debate": (
        "Adversarial: MAD-style debate with rebuttals and an adaptive early stop.\n"
        "Choreographed: fixed 5-round arc — establish → attack → converge → synthesize → review."
    ),
    "synthesis": (
        "Synthesis: agents generate independently, mediator integrates.\n"
        "Rotation: neuromorphic drafts a position, then aiml and neuroscience revise it in "
        "turn before it returns to neuromorphic — one rotation. After n rotations, "
        "neuromorphic's revision is the final answer."
    ),
    "neukrag": (
        "NeuKRAG: single-agent hypothesis from the neuromorphic KG only.\n"
        "NeuKRAG-inter: single-agent hypothesis from the unified cross-domain KG."
    ),
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

        top_mode = st.radio(
            "Mode",
            options=["debate", "synthesis", "neukrag"],
            format_func=_TOP_MODE_LABELS.get,
            help=_TOP_MODE_HELP,
        )

        sub_options = _SUB_MODES[top_mode]
        if len(sub_options) > 1:
            mode = st.selectbox(
                _SUB_MODE_SELECT_LABEL.get(top_mode, "Style"),
                options=sub_options,
                format_func=_SUB_MODE_LABELS.get,
                help=_SUB_MODE_HELP.get(top_mode),
            )
        else:
            mode = sub_options[0]

        if mode == "adversarial":
            debate_rounds = st.slider("Max Rounds", 1, 5, 3)
            debate_level  = st.slider(
                "Tit-for-tat Level", 0, 3, 2,
                help="\n".join(f"{k}: {v}" for k, v in DEBATE_LEVEL_PROMPTS.items()),
            )
        else:
            debate_rounds, debate_level = 3, 2

        if mode == "rotation":
            n_rotations = st.slider(
                "Rotations", 1, 5, 1,
                help="How many full aiml → neuroscience → neuromorphic cycles before "
                     "neuromorphic's revision becomes the final answer.",
            )
        else:
            n_rotations = 1

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
        "n_rotations":   n_rotations,
        "k_hops":        k_hops,
        "max_triples":   max_triples,
    }
