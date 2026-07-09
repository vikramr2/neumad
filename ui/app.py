"""
NeuMAD — Streamlit Chat UI

Chat-style interface for the NeuKRAG Multi-Agent Debate system.
- st.chat_input anchors the message bar to the bottom; Enter sends
- Sidebar shows settings and a clickable conversation history
- Follow-up questions are answered by the mediator in context of the active synthesis
- "New Debate" starts a fresh conversation and saves the current one to history

The rendering/state logic lives in sibling modules (history_store, system_bootstrap,
sidebar, render_debate, render_graph, render_chat) — this file is just the top-level
Streamlit script: bootstrap, page config, and the chat-input driver loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Path bootstrap — must run before importing orchestration/run_neukrag, or any
# sibling ui module that transitively imports them. Streamlit's scriptrunner
# exec()s this file rather than invoking it as `python ui/app.py`, so this
# file's own directory isn't automatically on sys.path — add it explicitly so
# flat imports of the sibling ui modules (history_store, sidebar, etc.) resolve.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
_UI   = Path(__file__).parent
sys.path.insert(0, str(_UI))
sys.path.insert(0, str(_ROOT / "mad"))
sys.path.insert(0, str(_ROOT / "neukrag"))

from orchestration import (   # noqa: E402
    Mediator,
    run_adversarial,
    run_choreographed,
    run_followup,
    run_neukrag_single,
    run_rotation,
    run_synthesis,
)

from history_store    import _init_state, _save_history            # noqa: E402
from render_chat       import render_messages, render_result_in_chat # noqa: E402
from sidebar           import render_sidebar                        # noqa: E402
from system_bootstrap  import build_neukrag_system, build_system    # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="NeuMAD",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

_init_state()

settings              = render_sidebar()
mode                  = settings["mode"]
debate_rounds         = settings["debate_rounds"]
debate_level          = settings["debate_level"]
neuromorphic_mediator = settings["neuromorphic_mediator"]
n_rotations           = settings["n_rotations"]
k_hops                = settings["k_hops"]
max_triples           = settings["max_triples"]

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("🧠 NeuMAD — (Neu)romorphic Multi-Agent Debate")
st.caption(
    "Three KG-grounded agents (Neuroscience · AI/ML · Neuromorphic) synthesize or debate "
    "neuromorphic hypotheses. Follow up with questions after each result."
)

# --- History viewer mode ---
if st.session_state["viewing_history"]:
    st.info("📖 Viewing past conversation — read only")
    if st.button("← Back to active chat"):
        st.session_state["viewing_history"] = False
        st.rerun()
    render_messages(st.session_state["history_snapshot"], read_only=True)
    st.stop()  # don't show chat input in read-only mode

# --- Active conversation ---
render_messages(st.session_state["messages"])

# --- Status placeholder (updated during inference) ---
status_box = st.empty()

# --- Chat input (anchored to bottom, Enter to send) ---
is_followup = st.session_state["active_synthesis"] is not None
placeholder = (
    "Ask a follow-up question about the synthesis…"
    if is_followup else
    "Enter a neuromorphic computing research question…"
)

if prompt := st.chat_input(placeholder):
    _is_neukrag = mode in ("neukrag", "neukrag-inter")
    if _is_neukrag:
        kg_name = "all" if mode == "neukrag-inter" else "neuromorphic"
        neukrag_graph, neukrag_meta, neukrag_extractor, neukrag_hyp = build_neukrag_system(
            kg_name, k_hops, max_triples
        )
        agents, mediator = None, None
    else:
        agents, mediator = build_system(k_hops, max_triples)
        neukrag_graph = neukrag_meta = neukrag_extractor = neukrag_hyp = None

    # Append user message immediately
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    def on_status(msg: str):
        status_box.info(f"⏳ {msg}")

    with st.chat_message("assistant"):
        result_placeholder = st.empty()

        if is_followup:
            with st.spinner("Answering follow-up…"):
                result = run_followup(
                    prompt,
                    st.session_state["active_synthesis"],
                    mediator if mediator else Mediator(),
                    status_cb=on_status,
                )
            status_box.empty()
            result_placeholder.markdown(result["final_hypothesis"])
            st.session_state["messages"].append({
                "role":    "assistant",
                "content": result["final_hypothesis"],
                "result":  None,
            })
            _save_history()
        else:
            spinner_msg = {
                "synthesis":     "Running synthesis…",
                "adversarial":   "Running adversarial debate…",
                "choreographed": "Running choreographed debate…",
                "rotation":      "Running rotation synthesis…",
                "neukrag":       "Running NeuKRAG…",
                "neukrag-inter": "Running NeuKRAG-inter…",
            }.get(mode, "Running…")
            if neuromorphic_mediator and mode in ("adversarial", "choreographed"):
                spinner_msg = spinner_msg.rstrip("…") + " (neuromorphic-mediated)…"
            with st.spinner(spinner_msg):
                if mode == "synthesis":
                    result = run_synthesis(prompt, agents, mediator, status_cb=on_status)
                elif mode == "adversarial":
                    result = run_adversarial(
                        prompt, agents, mediator,
                        max_rounds=debate_rounds,
                        debate_level=debate_level,
                        neuromorphic_mediator=neuromorphic_mediator,
                        status_cb=on_status,
                    )
                elif mode == "choreographed":
                    result = run_choreographed(
                        prompt, agents, mediator,
                        neuromorphic_mediator=neuromorphic_mediator,
                        status_cb=on_status,
                    )
                elif mode == "rotation":
                    result = run_rotation(
                        prompt, agents, mediator,
                        n_rotations=n_rotations,
                        status_cb=on_status,
                    )
                else:
                    result = run_neukrag_single(
                        prompt, neukrag_graph, neukrag_meta,
                        neukrag_extractor, neukrag_hyp,
                        k_hops=k_hops, max_triples=max_triples,
                        mode=mode, status_cb=on_status,
                    )
            status_box.empty()
            result_placeholder.empty()
            render_result_in_chat(result)
            st.session_state["active_synthesis"] = result["final_hypothesis"]
            st.session_state["messages"].append({
                "role":    "assistant",
                "content": result["final_hypothesis"],
                "result":  result,
            })
            _save_history()
