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
_HISTORY_FILE  = _ROOT / "chat_history.json"
_ARTIFACTS_DIR = _ROOT / "artifacts"
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
    HypothesisGenerator,
    Mediator,
    SpecialistAgent,
    _AGENT_LABELS,
    load_graph,
    load_metadata,
    load_toml,
    run_adversarial,
    run_choreographed,
    run_followup,
    run_neukrag_single,
    run_synthesis,
)
from run_neukrag import EntityExtractor  # noqa: E402

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


def _save_artifacts(result: dict, artifact_name: str) -> Path:
    import csv as _csv
    base          = _ARTIFACTS_DIR / artifact_name
    responses_dir = base / "responses"
    kg_dir        = base / "kg_triples"
    responses_dir.mkdir(parents=True, exist_ok=True)
    kg_dir.mkdir(parents=True, exist_ok=True)

    mode = result["mode"]

    # Normalise all modes to a flat list of history entries
    if mode in ("neukrag", "neukrag-inter"):
        agent_label = "neukrag_inter" if mode == "neukrag-inter" else "neukrag"
        entries = [{
            "agent":      agent_label,
            "round":      1,
            "statement":  result["final_hypothesis"],
            "references": result.get("references", ""),
            "triples":    result.get("triples", []),
            "agreed":     None,
        }]
    elif mode == "synthesis":
        entries = [
            {
                "agent":      name,
                "round":      1,
                "statement":  data["statement"],
                "references": data["references"],
                "triples":    data.get("triples", []),
                "agreed":     None,
            }
            for name, data in result["agent_hypotheses"].items()
        ]
    else:
        entries = result.get("debate_history", [])

    for entry in entries:
        agent     = entry["agent"]
        round_num = entry["round"]
        stem      = f"round_{round_num:02d}_{agent}"

        # Response JSON (everything except raw triples)
        resp = {k: v for k, v in entry.items() if k != "triples"}
        with open(responses_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(resp, f, indent=2, ensure_ascii=False, default=str)

        # KG triples CSV (only when the agent queried the KG this round)
        triples = entry.get("triples") or []
        if triples:
            with open(kg_dir / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
                writer = _csv.DictWriter(f, fieldnames=["h", "r", "t", "document_id"])
                writer.writeheader()
                for t in triples:
                    writer.writerow({
                        "h":           t.get("h", ""),
                        "r":           t.get("r", ""),
                        "t":           t.get("t", ""),
                        "document_id": t.get("document_id", ""),
                    })

    # Final synthesis
    with open(responses_dir / "final_synthesis.json", "w", encoding="utf-8") as f:
        json.dump(
            {"mode": mode, "query": result.get("query", ""), "final_hypothesis": result["final_hypothesis"]},
            f, indent=2, ensure_ascii=False,
        )

    return base


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
        st.session_state["pending_save"]     = None  # msg index waiting for artifact name
    else:
        st.session_state.setdefault("viewing_history",  False)
        st.session_state.setdefault("history_snapshot", [])
        st.session_state.setdefault("pending_save",     None)

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
        "neukrag":       "🔬 NeuKRAG",
        "neukrag-inter": "🌐 NeuKRAG-inter",
    }
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


@st.cache_resource(show_spinner="Loading knowledge graph…")
def build_neukrag_system(kg_name: str, k_hops: int, max_triples: int):
    lm = dspy.LM(LLM_MODEL, api_base=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
    dspy.configure(lm=lm)

    cfg      = load_toml(CONFIG_PATH)
    kg_cfg   = cfg.get("kg_paths", {})
    meta_cfg = cfg.get("metadata_paths", {})

    kg_key  = "all_kg" if kg_name == "all" else f"{kg_name}_kg"
    graph   = load_graph(Path(kg_cfg[kg_key]).expanduser())

    if kg_name == "all":
        unified_meta: dict = {}
        for name in ("neuroscience", "aiml", "neuromorphic"):
            if meta_cfg.get(f"{name}_metadata"):
                unified_meta.update(
                    load_metadata(Path(meta_cfg[f"{name}_metadata"]).expanduser())
                )
        metadata = unified_meta
    else:
        metadata = (
            load_metadata(Path(meta_cfg[f"{kg_name}_metadata"]).expanduser())
            if meta_cfg.get(f"{kg_name}_metadata") else {}
        )

    return graph, metadata, EntityExtractor(), HypothesisGenerator()

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
                    st.toast(f"Saved to {saved_path.relative_to(_ROOT)}/", icon="💾")
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
                "neukrag":       "Running NeuKRAG…",
                "neukrag-inter": "Running NeuKRAG-inter…",
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
                elif mode == "choreographed":
                    result = run_choreographed(prompt, agents, mediator, status_cb=on_status)
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
            _render_save_button(result, len(st.session_state["messages"]) - 1)
