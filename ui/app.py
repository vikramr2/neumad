"""
NeuMAD — Streamlit Chat UI

Chat-style interface for the NeuKRAG Multi-Agent Debate system.
- st.chat_input anchors the message bar to the bottom; Enter sends
- Sidebar shows settings and a clickable conversation history
- Follow-up questions are answered by the mediator in context of the active synthesis
- "New Debate" starts a fresh conversation and saves the current one to history
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
_HISTORY_FILE = _ROOT / "chat_history.json"
sys.path.insert(0, str(_ROOT / "mad"))
sys.path.insert(0, str(_ROOT / "neukrag"))

from orchestration import (   # noqa: E402
    CONFIG_PATH,
    CHOREOGRAPHED_COVARIANCE,
    CHOREOGRAPHED_ROUND_LABELS,
    DEBATE_LEVEL_PROMPTS,
    LLM_MODEL,
    OLLAMA_API_KEY,
    OLLAMA_BASE_URL,
    Mediator,
    SpecialistAgent,
    _AGENT_LABELS,
    load_metadata,
    load_toml,
    run_adversarial,
    run_choreographed,
    run_followup,
    run_synthesis,
)

import dspy  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="NeuMAD",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# History persistence
# ---------------------------------------------------------------------------

def _load_history() -> dict:
    if _HISTORY_FILE.exists():
        try:
            with open(_HISTORY_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_history():
    data = {
        "messages":         st.session_state["messages"],
        "conv_history":     st.session_state["conv_history"],
        "active_synthesis": st.session_state["active_synthesis"],
    }
    try:
        with open(_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    except Exception:
        pass  # never crash the UI over a failed write


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

def _init_state():
    if "history_loaded" not in st.session_state:
        saved = _load_history()
        st.session_state["messages"]         = saved.get("messages", [])
        st.session_state["conv_history"]     = saved.get("conv_history", [])
        st.session_state["active_synthesis"] = saved.get("active_synthesis", None)
        st.session_state["viewing_history"]  = False
        st.session_state["history_snapshot"] = []
        st.session_state["history_loaded"]   = True
    else:
        st.session_state.setdefault("viewing_history",  False)
        st.session_state.setdefault("history_snapshot", [])

_init_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

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

    _MODE_LABELS = {
        "synthesis":     "🤝 Synthesis",
        "adversarial":   "⚔️ Adversarial",
        "choreographed": "🎼 Choreographed",
    }
    mode = st.radio(
        "Mode",
        options=["synthesis", "adversarial", "choreographed"],
        format_func=_MODE_LABELS.get,
        help=(
            "Synthesis: agents generate independently, mediator integrates.\n"
            "Adversarial: MAD-style debate with rebuttals and adaptive break.\n"
            "Choreographed: fixed 5-round arc — establish → attack → converge → synthesize → review."
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

# ---------------------------------------------------------------------------
# Cached system bootstrap
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading knowledge graphs…")
def build_system(k_hops: int, max_triples: int):
    lm = dspy.LM(LLM_MODEL, api_base=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
    dspy.configure(lm=lm)

    cfg      = load_toml(CONFIG_PATH)
    kg_cfg   = cfg.get("kg_paths", {})
    meta_cfg = cfg.get("metadata_paths", {})

    metadata = {
        name: load_metadata(Path(meta_cfg[f"{name}_metadata"]).expanduser())
        for name in ("neuroscience", "aiml", "neuromorphic")
        if meta_cfg.get(f"{name}_metadata")
    }

    agents = [
        SpecialistAgent("neuroscience", Path(kg_cfg["neuroscience_kg"]).expanduser(),
                        k_hops, max_triples, metadata=metadata.get("neuroscience")),
        SpecialistAgent("aiml",         Path(kg_cfg["aiml_kg"]).expanduser(),
                        k_hops, max_triples, metadata=metadata.get("aiml")),
        SpecialistAgent("neuromorphic", Path(kg_cfg["neuromorphic_kg"]).expanduser(),
                        k_hops, max_triples, metadata=metadata.get("neuromorphic")),
    ]
    return agents, Mediator()

# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

_AGENT_COLORS = {
    "neuroscience": "#1f6aa5",
    "aiml":         "#2d8a4e",
    "neuromorphic": "#8b4513",
}


def _agreement_badge(agreed: bool | None) -> str:
    if agreed is None:
        return ""
    return " 🟢" if agreed else " 🔴"


def _render_agent_columns(entries: list[dict], *, show_agreement: bool):
    cols = st.columns(3)
    for col, entry in zip(cols, entries):
        name  = entry["agent"]
        label = _AGENT_LABELS[name]
        badge = _agreement_badge(entry.get("agreed")) if show_agreement else ""
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


def render_result_in_chat(result: dict):
    """Render a full pipeline result inside an st.chat_message block."""
    # Final synthesis always shown at top
    st.markdown(result["final_hypothesis"])

    # Expandable detail section
    mode = result["mode"]
    if mode == "synthesis":
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
            for round_num, entries in sorted(rounds.items()):
                if round_num == 0:
                    st.markdown("**Round 0 — Initial Positions**")
                else:
                    agreed_count = sum(1 for e in entries if e.get("agreed") is True)
                    st.markdown(f"**Round {round_num}** ({agreed_count}/{len(entries)} agree)")
                _render_agent_columns(entries, show_agreement=(round_num > 0))
                if round_num < max(rounds):
                    st.divider()

    elif mode == "choreographed":
        c_rounds: dict[int, list[dict]] = {}
        for entry in result["debate_history"]:
            c_rounds.setdefault(entry["round"], []).append(entry)

        with st.expander("Choreographed debate — 5 rounds", expanded=False):
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
                    agree_str = f" · {agreed_count}/{len(entries)} agree" if show_agree else ""
                    st.markdown(f"**Round {round_num} — {round_label}**{cov_badge}{agree_str}")
                    _render_agent_columns(entries, show_agreement=show_agree)

                if round_num < max(c_rounds):
                    st.divider()


def render_messages(messages: list[dict], *, read_only: bool = False):
    """Render a list of chat messages."""
    for msg in messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.write(msg["content"])
            elif msg.get("result"):
                render_result_in_chat(msg["result"])
            else:
                st.markdown(msg["content"])

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
    agents, mediator = build_system(k_hops, max_triples)

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
                    mediator,
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
            }.get(mode, "Running…")
            with st.spinner(spinner_msg):
                if mode == "synthesis":
                    result = run_synthesis(prompt, agents, mediator, status_cb=on_status)
                elif mode == "adversarial":
                    result = run_adversarial(
                        prompt, agents, mediator,
                        max_rounds=debate_rounds,
                        debate_level=debate_level,
                        status_cb=on_status,
                    )
                else:
                    result = run_choreographed(prompt, agents, mediator, status_cb=on_status)
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
