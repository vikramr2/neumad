"""
NeuMAD — Streamlit Chat UI

Chat-style interface for the NeuKRAG Multi-Agent Debate system.
- st.chat_input anchors the message bar to the bottom; Enter sends
- Sidebar shows settings and a clickable conversation history
- Follow-up questions are answered by the mediator in context of the active synthesis
- "New Debate" starts a fresh conversation and saves the current one to history
"""

from __future__ import annotations

import html as _html
import json
import re as _re
import sys
from datetime import datetime
from pathlib import Path

import textwrap as _textwrap

import markdown as _md_lib
import plotly.graph_objects as _go
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
    lm = dspy.LM(LLM_MODEL, api_base=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY, cache=False)
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
    lm = dspy.LM(LLM_MODEL, api_base=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY, cache=False)
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


# ---------------------------------------------------------------------------
# Argumentation graph helpers
# ---------------------------------------------------------------------------

_NODE_COLORS = {
    "main_argument":       "#2563eb",
    "supporting_argument": "#10b981",
    "attacking_argument":  "#ef4444",
}
_NODE_TYPE_LABELS = {
    "main_argument":       "main claim",
    "supporting_argument": "supports",
    "attacking_argument":  "challenges",
}
_EXPERT_HEX = {
    "neuroscience": "#1f6aa5",
    "aiml":         "#2d8a4e",
    "neuromorphic": "#8b4513",
}


def _hover_wrap(s: str, width: int = 58) -> str:
    return "<br>".join(_textwrap.wrap(s or "", width=width))


def _hex_rgba(hex_color: str, alpha: float) -> str:
    """Convert a #rrggbb hex string to an rgba() string with the given alpha."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _render_argumentation_graph(graph_dict: dict) -> None:
    """Render a QBAF as an interactive Plotly graph with per-node hover cards."""
    nodes = graph_dict.get("nodes", [])
    edges = graph_dict.get("edges", [])
    if not nodes:
        st.info("No argumentation graph available.")
        return

    node_map = {n["id"]: n for n in nodes}

    # ── Column layout ────────────────────────────────────────────────────────
    expert_order = ["neuroscience", "aiml", "neuromorphic"]
    seen: set[str] = set()
    actual_experts: list[str] = []
    for exp in expert_order:
        if any(n["expert"] == exp for n in nodes):
            actual_experts.append(exp)
            seen.add(exp)
    for n in nodes:
        if n["expert"] not in seen:
            actual_experts.append(n["expert"])
            seen.add(n["expert"])

    n_exp = len(actual_experts)
    col_x = {exp: (i + 0.5) / n_exp for i, exp in enumerate(actual_experts)}
    col_half = 0.42 / n_exp  # half-width of each column

    # Group nodes by expert
    by_expert: dict[str, list] = {exp: [] for exp in actual_experts}
    for n in nodes:
        by_expert[n["expert"]].append(n)

    # ── Assign positions ────────────────────────────────────────────────────
    node_pos: dict[int, tuple[float, float]] = {}
    for exp in actual_experts:
        xc = col_x[exp]
        exp_nodes = by_expert[exp]
        mains  = [n for n in exp_nodes if n["type"] == "main_argument"]
        others = [n for n in exp_nodes if n["type"] != "main_argument"]

        for mn in mains:
            node_pos[mn["id"]] = (xc, 0.82)

        n_o = len(others)
        # Spread others in a grid below the main node
        cols_per_row = 3
        row_gap, col_gap = 0.22, col_half * 0.6
        for i, nd in enumerate(others):
            row, col = divmod(i, cols_per_row)
            x_off = (col - (min(n_o, cols_per_row) - 1) / 2) * col_gap
            node_pos[nd["id"]] = (xc + x_off, 0.52 - row * row_gap)

    # ── Edge arrow annotations ───────────────────────────────────────────────
    arrow_annotations: list[dict] = []
    edge_line_traces: list[_go.Scatter] = []
    for e in edges:
        src, tgt = e["source"], e["target"]
        if src not in node_pos or tgt not in node_pos:
            continue
        x0, y0 = node_pos[src]
        x1, y1 = node_pos[tgt]
        is_sup = e["edge_type"] == "support_edge"
        color  = "#10b981" if is_sup else "#ef4444"
        arrow_annotations.append(dict(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=3, arrowsize=1.2, arrowwidth=1.6,
            arrowcolor=color,
        ))
        # Thin line body so the arrow shaft is visible beyond the arrowhead
        edge_line_traces.append(_go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(color=color, width=1.4),
            hoverinfo="none",
            showlegend=False,
        ))

    # ── Node scatter trace ───────────────────────────────────────────────────
    nx_l, ny_l, nc_l, ns_l, nh_l = [], [], [], [], []
    for nid, (x, y) in node_pos.items():
        node = node_map.get(nid)
        if not node:
            continue
        nx_l.append(x)
        ny_l.append(y)

        exp   = node.get("expert", "")
        ntype = node.get("type", "")
        hex_c = _EXPERT_HEX.get(exp, "#6b7280")
        nc_l.append(hex_c)
        ns_l.append(18 if ntype == "main_argument" else 11)

        # Hover card
        stmt      = node.get("statement", "")
        base_tau  = node.get("base")
        sigma     = node.get("qsem_strength")
        type_lbl  = _NODE_TYPE_LABELS.get(ntype, ntype.replace("_", " "))
        agent_lbl = _AGENT_LABELS.get(exp, exp)

        score_lines = ""
        if base_tau is not None:
            fe = round(base_tau * 10)
            score_lines += f"<br><b>ε</b>: {'▓'*fe}{'░'*(10-fe)} {base_tau:.2f}"
        if sigma is not None:
            fs = round(sigma * 10)
            score_lines += f"<br><b>σ</b>: {'▓'*fs}{'░'*(10-fs)} {sigma:.2f}"

        nh_l.append(
            f"<b>{agent_lbl}</b>  ·  <i>{type_lbl}</i>"
            f"{score_lines}"
            f"<br><br>{_hover_wrap(stmt)}"
        )

    node_trace = _go.Scatter(
        x=nx_l, y=ny_l,
        mode="markers",
        marker=dict(
            size=ns_l,
            color=nc_l,
            line=dict(width=2, color="white"),
        ),
        hovertext=nh_l,
        hoverinfo="text",
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#cbd5e1",
            font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                      size=12, color="#1e293b"),
            align="left",
            namelength=0,
        ),
        showlegend=False,
    )

    # ── Background column shapes ─────────────────────────────────────────────
    shapes: list[dict] = []
    for exp in actual_experts:
        xc   = col_x[exp]
        hex_c = _EXPERT_HEX.get(exp, "#6b7280")
        shapes.append(dict(
            type="rect",
            x0=xc - col_half, x1=xc + col_half,
            y0=-0.08, y1=0.97,
            fillcolor=_hex_rgba(hex_c, 0.07),
            line=dict(color=_hex_rgba(hex_c, 0.35), width=1, dash="dot"),
            layer="below",
        ))

    # ── Annotations: expert labels + edge legend ─────────────────────────────
    annotations = list(arrow_annotations)
    for exp in actual_experts:
        xc    = col_x[exp]
        hex_c = _EXPERT_HEX.get(exp, "#6b7280")
        annotations.append(dict(
            x=xc, y=0.965,
            xref="x", yref="y",
            text=f"<b>{_AGENT_LABELS.get(exp, exp)}</b>",
            showarrow=False,
            font=dict(size=11, color=hex_c),
            xanchor="center", yanchor="bottom",
        ))
    # Edge-type legend (paper coords, bottom-left)
    for i, (color, label) in enumerate([("#10b981", "supports"), ("#ef4444", "challenges")]):
        annotations.append(dict(
            x=0.01 + i * 0.14, y=0.01,
            xref="paper", yref="paper",
            text=f'<span style="color:{color}">——▶</span> {label}',
            showarrow=False,
            font=dict(size=11, color="#64748b"),
            xanchor="left", yanchor="bottom",
        ))

    fig = _go.Figure(
        data=edge_line_traces + [node_trace],
        layout=_go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(l=5, r=5, t=8, b=28),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       range=[-0.01, 1.01]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       range=[-0.12, 1.0]),
            shapes=shapes,
            annotations=annotations,
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=460,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


_LABEL_RE = _re.compile(
    r'<label\s+agent=["\']?([^"\'>\s]+)["\']?\s+node_id=["\']?(\d+)["\']?\s*>(.*?)</label>',
    _re.DOTALL,
)

# Matches LaTeX math blocks that the Python markdown library would mangle.
# Order matters: longer/greedier patterns first.
_MATH_RE = _re.compile(
    r'\$\$[\s\S]*?\$\$'                              # $$...$$ display
    r'|\\\[[\s\S]*?\\\]'                             # \[...\] display
    r'|\\begin\{[^}]+\}[\s\S]*?\\end\{[^}]+\}'      # \begin{env}...\end{env}
    r'|\$[^$\n]+?\$',                                # $...$ inline
    _re.DOTALL,
)

_MATHJAX_SCRIPT = """<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
    processEscapes: true,
    processEnvironments: true,
    packages: {'[+]': ['ams']}
  },
  options: { skipHtmlTags: ['script','noscript','style','textarea'] }
};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>"""

_SYNTHESIS_CSS = """<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #1e293b;
    background: transparent;
    margin: 0;
    padding: 6px 2px;
    font-size: 15px;
    line-height: 1.65;
  }
  h1 { font-size: 1.2em; font-weight: 700; color: #0f172a; margin: 0 0 0.8em; }
  h2 { font-size: 1.05em; font-weight: 700; color: #1d4ed8; margin: 1.3em 0 0.35em; }
  h3 { font-size: 0.95em; font-weight: 600; color: #1e40af; margin: 1em 0 0.25em; }
  p  { margin: 0.35em 0; }
  ul, ol { padding-left: 1.4em; }
  li { margin: 0.2em 0; }
  code { background: #f1f5f9; padding: 1px 5px; border-radius: 4px; font-size: 0.85em; color: #0f172a; }
  pre  { background: #f1f5f9; padding: 10px 14px; border-radius: 8px; overflow-x: auto; }
  strong { color: #0f172a; }
  em { color: #475569; }

  /* ── Labelled span ── */
  .lbl {
    position: relative;
    display: inline;
    border-bottom: 1.5px dotted #2563eb;
    cursor: help;
    border-radius: 2px;
    padding-bottom: 1px;
  }
  .lbl:hover { background: #eff6ff; }

  /* ── Popup ── */
  .popup {
    display: none;
    position: absolute;
    left: 0;
    top: 1.6em;
    width: 680px;
    max-height: 460px;
    background: #ffffff;
    border: 1px solid #bfdbfe;
    border-radius: 14px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.13);
    z-index: 9999;
    padding: 22px 26px 18px;
    overflow-y: auto;
    box-sizing: border-box;
    text-align: left;
    font-size: 14px;
  }
  .popup-label {
    font-size: 0.68em;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #2563eb;
    margin-bottom: 8px;
  }
  .node-card {
    background: #f8fafc;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
    border-left: 3px solid;
  }
  .node-card-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
  }
  .expert-badge {
    font-size: 0.68em;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 2px 7px;
    border-radius: 4px;
    color: white;
  }
  .node-type-badge {
    font-size: 0.68em;
    color: #64748b;
    font-style: italic;
  }
  .node-statement {
    font-size: 0.87em;
    color: #1e293b;
    line-height: 1.5;
    margin: 0;
  }
  .no-prov { color: #94a3b8; font-size: 0.87em; font-style: italic; }

  /* ── Strength bar ── */
  .strength-row {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 6px;
  }
  .strength-label {
    font-size: 0.68em;
    color: #64748b;
    white-space: nowrap;
    min-width: 90px;
  }
  .strength-bar-bg {
    flex: 1;
    height: 5px;
    background: #e2e8f0;
    border-radius: 3px;
    overflow: hidden;
  }
  .strength-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s;
  }
  .strength-val {
    font-size: 0.72em;
    font-weight: 600;
    min-width: 30px;
    text-align: right;
  }
</style>"""

_POPUP_HOVER_JS = """<script>
(function() {
  var timers = {};
  document.querySelectorAll('.lbl').forEach(function(el) {
    var popup = el.querySelector('.popup');
    if (!popup) return;
    var id = el.dataset.nid;
    el.addEventListener('mouseenter', function() {
      timers[id] = setTimeout(function() { popup.style.display = 'block'; }, 200);
    });
    el.addEventListener('mouseleave', function() {
      clearTimeout(timers[id]);
      popup.style.display = 'none';
    });
    popup.addEventListener('mouseenter', function() { clearTimeout(timers[id]); });
    popup.addEventListener('mouseleave', function() { popup.style.display = 'none'; });
  });
})();
</script>"""


def _strength_bar_html(label: str, strength: float | None) -> str:
    if strength is None:
        return ""
    pct   = int(strength * 100)
    color = "#10b981" if strength >= 0.6 else ("#f59e0b" if strength >= 0.4 else "#ef4444")
    return (
        f'<div class="strength-row">'
        f'<span class="strength-label">{label}</span>'
        f'<div class="strength-bar-bg">'
        f'<div class="strength-bar-fill" style="width:{pct}%;background:{color}"></div>'
        f'</div>'
        f'<span class="strength-val" style="color:{color}">{strength:.2f}</span>'
        f'</div>'
    )


def _node_popup_card(node: dict) -> str:
    expert   = node.get("expert", "")
    ntype    = node.get("type", "")
    stmt     = _html.escape(node.get("statement", ""))
    color    = _EXPERT_HEX.get(expert, "#6b7280")
    label    = _NODE_TYPE_LABELS.get(ntype, ntype.replace("_", " "))
    base     = node.get("base")
    sigma    = node.get("qsem_strength")
    return (
        f'<div class="node-card" style="border-left-color:{color}">'
        f'<div class="node-card-header">'
        f'<span class="expert-badge" style="background:{color}">{_html.escape(expert)}</span>'
        f'<span class="node-type-badge">{_html.escape(label)}</span>'
        f'</div>'
        f'{_strength_bar_html("ε intrinsic", base)}'
        f'{_strength_bar_html("σ dialectical", sigma)}'
        f'<p class="node-statement">{stmt}</p>'
        f'</div>'
    )


def _render_synthesis_with_labels(synthesis_text: str, graph_dict: dict) -> None:
    """Render synthesis HTML with inline <label> tags turned into hoverable spans."""
    node_map = {n["id"]: n for n in (graph_dict or {}).get("nodes", [])}

    def _replace_label(m: _re.Match) -> str:
        agent   = m.group(1)
        node_id = int(m.group(2))
        text    = m.group(3).strip()
        node    = node_map.get(node_id)
        if not node:
            return _html.escape(text)
        card = _node_popup_card(node)
        popup = (
            f'<span class="popup">'
            f'<span class="popup-label">{_html.escape(agent)} · node {node_id}</span>'
            f'{card}'
            f'</span>'
        )
        return (
            f'<span class="lbl" data-nid="{node_id}" data-agent="{_html.escape(agent)}">'
            f'{_html.escape(text)}{popup}'
            f'</span>'
        )

    # Stash both <label> tags and LaTeX math blocks before markdown processes the
    # text, so the markdown library can't mangle subscripts, backslashes, etc.
    placeholders: dict[str, str] = {}

    def _stash_label(m: _re.Match) -> str:
        key = f"\x00PH{len(placeholders)}\x00"
        placeholders[key] = _replace_label(m)
        return key

    def _stash_math(m: _re.Match) -> str:
        key = f"\x00PH{len(placeholders)}\x00"
        placeholders[key] = m.group(0)   # restore verbatim for MathJax
        return key

    stashed = _LABEL_RE.sub(_stash_label, synthesis_text)
    stashed = _MATH_RE.sub(_stash_math, stashed)

    md_html = _md_lib.markdown(stashed, extensions=["tables", "fenced_code"])

    for key, val in placeholders.items():
        md_html = md_html.replace(_html.escape(key), val).replace(key, val)

    full_html = (
        f"<!DOCTYPE html><html><head>{_MATHJAX_SCRIPT}{_SYNTHESIS_CSS}</head>"
        f"<body>{md_html}{_POPUP_HOVER_JS}</body></html>"
    )
    est_lines  = synthesis_text.count("\n") + 1
    est_height = max(600, min(4000, est_lines * 30 + 220))
    st.components.v1.html(full_html, height=est_height, scrolling=True)


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
