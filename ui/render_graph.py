"""Argumentation-graph (QBAF) rendering and inline-label synthesis rendering."""

from __future__ import annotations

import html as _html
import re as _re
import textwrap as _textwrap

import markdown as _md_lib
import plotly.graph_objects as _go
import streamlit as st

from orchestration import _AGENT_LABELS

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
