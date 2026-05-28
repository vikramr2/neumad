"""
NeuKRAG Multi-Agent Debate — Streamlit UI

Each debate round is shown as a collapsible section with three side-by-side agent columns.
Each column displays the agent's response, APA references, and a 🟢/🔴 agreement indicator.
The mediator's final synthesis is shown below all rounds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Path bootstrap — find orchestration.py
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "mad"))
sys.path.insert(0, str(_ROOT / "neukrag"))

from orchestration import (   # noqa: E402
    CONFIG_PATH,
    ENV_PATH,
    DEBATE_LEVEL_PROMPTS,
    LLM_MODEL,
    OLLAMA_API_KEY,
    OLLAMA_BASE_URL,
    Mediator,
    SpecialistAgent,
    _AGENT_LABELS,
    load_env,
    load_metadata,
    load_toml,
    run_adversarial,
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

st.title("🧠 NeuMAD — (Neu)romorphic Multi-Agent Debate")
st.caption(
    "Three KG-grounded specialist agents (Neuroscience · AI/ML · Neuromorphic) "
    "synthesize or debate neuromorphic computing hypotheses."
)

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    mode = st.radio(
        "Mode",
        options=["synthesis", "adversarial"],
        format_func=lambda x: "🤝 Synthesis" if x == "synthesis" else "⚔️ Adversarial",
        help="Synthesis: agents generate independently, mediator integrates. "
             "Adversarial: MAD-style debate with rebuttals and adaptive break.",
    )

    if mode == "adversarial":
        st.subheader("Debate Settings")
        debate_rounds = st.slider("Max Rounds", min_value=1, max_value=5, value=3)
        debate_level  = st.slider(
            "Tit-for-tat Level",
            min_value=0, max_value=3, value=2,
            help="\n".join(f"{k}: {v}" for k, v in DEBATE_LEVEL_PROMPTS.items()),
        )
    else:
        debate_rounds = 3
        debate_level  = 2

    st.subheader("Graph Settings")
    k_hops      = st.slider("K-hops", min_value=1, max_value=3, value=2)
    max_triples = st.slider("Max Triples / Agent", min_value=10, max_value=60, value=40, step=5)

# ---------------------------------------------------------------------------
# Cached system bootstrap — loads KGs and configures DSPy once
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
        SpecialistAgent(
            "neuroscience",
            Path(kg_cfg["neuroscience_kg"]).expanduser(),
            k_hops, max_triples,
            metadata=metadata.get("neuroscience"),
        ),
        SpecialistAgent(
            "aiml",
            Path(kg_cfg["aiml_kg"]).expanduser(),
            k_hops, max_triples,
            metadata=metadata.get("aiml"),
        ),
        SpecialistAgent(
            "neuromorphic",
            Path(kg_cfg["neuromorphic_kg"]).expanduser(),
            k_hops, max_triples,
            metadata=metadata.get("neuromorphic"),
        ),
    ]
    mediator = Mediator()
    return agents, mediator

# ---------------------------------------------------------------------------
# Query input
# ---------------------------------------------------------------------------

query = st.text_area(
    "Research Query",
    placeholder="Design a biologically plausible, scalable spiking neuron model for neuromorphic hardware",
    height=80,
)

run_btn = st.button("▶ Run", type="primary", disabled=not query.strip())

# ---------------------------------------------------------------------------
# Display helpers
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


def _render_agent_column(col, entry: dict, *, show_agreement: bool):
    name  = entry["agent"]
    label = _AGENT_LABELS[name]
    badge = _agreement_badge(entry.get("agreed")) if show_agreement else ""

    with col:
        st.markdown(
            f"<span style='color:{_AGENT_COLORS[name]};font-weight:700;font-size:1rem'>"
            f"{label}{badge}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(entry["statement"])

        refs = entry.get("references", "").strip()
        if refs:
            with st.expander("References", expanded=False):
                for line in refs.splitlines():
                    line = line.strip()
                    if line:
                        st.markdown(f"- {line}")


def _render_synthesis_column(col, name: str, data: dict):
    label = _AGENT_LABELS[name]
    with col:
        st.markdown(
            f"<span style='color:{_AGENT_COLORS[name]};font-weight:700;font-size:1rem'>"
            f"{label}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(data["statement"])

        refs = data.get("references", "").strip()
        if refs:
            with st.expander("References", expanded=False):
                for line in refs.splitlines():
                    line = line.strip()
                    if line:
                        st.markdown(f"- {line}")


def render_synthesis_result(result: dict):
    with st.expander("**Agent Hypotheses**", expanded=True):
        cols = st.columns(3)
        for col, name in zip(cols, ["neuroscience", "aiml", "neuromorphic"]):
            _render_synthesis_column(col, name, result["agent_hypotheses"][name])

    st.divider()
    st.subheader("🔬 Mediator Synthesis")
    st.markdown(result["final_hypothesis"])


def render_adversarial_result(result: dict):
    # Group history entries by round
    rounds: dict[int, list[dict]] = {}
    for entry in result["debate_history"]:
        rounds.setdefault(entry["round"], []).append(entry)

    for round_num, entries in sorted(rounds.items()):
        if round_num == 0:
            label = "**Round 0 — Initial Positions**"
        else:
            # Tally agreement indicators for the expander header
            agreed_count = sum(1 for e in entries if e.get("agreed") is True)
            total        = len(entries)
            tally        = f"({agreed_count}/{total} agree)"
            label        = f"**Round {round_num} — Debate** {tally}"

        with st.expander(label, expanded=(round_num == 0)):
            cols = st.columns(3)
            for col, entry in zip(cols, entries):
                _render_agent_column(col, entry, show_agreement=(round_num > 0))

    st.divider()
    rounds_info = (
        f"Concluded after {result['rounds_completed']} round(s) · "
        f"Debate level {result['debate_level']}"
    )
    st.subheader(f"🔬 Mediator Final Synthesis")
    st.caption(rounds_info)
    st.markdown(result["final_hypothesis"])

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if run_btn and query.strip():
    agents, mediator = build_system(k_hops, max_triples)

    status_box = st.empty()

    def on_status(msg: str):
        status_box.info(f"⏳ {msg}")

    try:
        if mode == "synthesis":
            with st.spinner("Running synthesis…"):
                result = run_synthesis(query.strip(), agents, mediator, status_cb=on_status)
        else:
            with st.spinner("Running adversarial debate…"):
                result = run_adversarial(
                    query.strip(), agents, mediator,
                    max_rounds=debate_rounds,
                    debate_level=debate_level,
                    status_cb=on_status,
                )
    finally:
        status_box.empty()

    st.session_state["last_result"] = result

# Persist result across sidebar interactions
if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    st.markdown(f"**Query:** {result['query']}")
    st.divider()
    if result["mode"] == "synthesis":
        render_synthesis_result(result)
    else:
        render_adversarial_result(result)
