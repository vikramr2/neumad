"""Argumentation-graph (QBAF) rendering and inline-label synthesis rendering."""

from __future__ import annotations

import html as _html
import re as _re
import textwrap as _textwrap

import markdown as _md_lib
import plotly.graph_objects as _go
import streamlit as st

from load_html import load, load_template
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

_MATHJAX_SCRIPT = load("mathjax.html").rstrip()
_SYNTHESIS_CSS  = f"<style>\n{load('synthesis.css').rstrip()}\n</style>"
_POPUP_HOVER_JS = f"<script>\n{load('popup_hover.js').rstrip()}\n</script>"

_STRENGTH_BAR_TEMPLATE = load_template("strength_bar.html")
_NODE_CARD_TEMPLATE    = load_template("node_card.html")


def _strength_bar_html(label: str, strength: float | None) -> str:
    if strength is None:
        return ""
    pct   = int(strength * 100)
    color = "#10b981" if strength >= 0.6 else ("#f59e0b" if strength >= 0.4 else "#ef4444")
    return _STRENGTH_BAR_TEMPLATE.substitute(
        label=label, pct=pct, color=color, strength=f"{strength:.2f}",
    )


def _node_popup_card(node: dict) -> str:
    expert   = node.get("expert", "")
    ntype    = node.get("type", "")
    stmt     = _markdown_inline_or_block(node.get("statement", ""))
    color    = _EXPERT_HEX.get(expert, "#6b7280")
    label    = _NODE_TYPE_LABELS.get(ntype, ntype.replace("_", " "))
    base     = node.get("base")
    sigma    = node.get("qsem_strength")
    return _NODE_CARD_TEMPLATE.substitute(
        color=color,
        expert=_html.escape(expert),
        type_label=_html.escape(label),
        # div, not p — a rotation node's statement can itself contain block-level
        # markdown (e.g. a header line), which <p> can't validly contain.
        base_bar=_strength_bar_html("ε intrinsic", base),
        sigma_bar=_strength_bar_html("σ dialectical", sigma),
        statement=stmt,
    )


def _markdown_inline_or_block(text: str) -> str:
    """Render label content as markdown. Synthesis-mode labels usually wrap a short
    inline clause — markdown wraps that in a single <p>, which would break the
    surrounding sentence's flow, so strip it when it's the sole wrapper. Rotation-mode
    labels can wrap whole lines (headers, bold-prefixed lines) — those don't start
    with <p>, so they pass through as real block elements instead of literal text."""
    html = _md_lib.markdown(text, extensions=["tables", "fenced_code"]).strip()
    if html.startswith("<p>") and html.endswith("</p>") and html.count("<p>") == 1:
        html = html[3:-4]
    return html


def _render_synthesis_with_labels(synthesis_text: str, graph_dict: dict) -> None:
    """Render synthesis HTML with inline <label> tags turned into hoverable spans."""
    node_map = {n["id"]: n for n in (graph_dict or {}).get("nodes", [])}

    def _replace_label(m: _re.Match) -> str:
        agent   = m.group(1)
        node_id = int(m.group(2))
        text    = m.group(3).strip()
        node    = node_map.get(node_id)
        if not node:
            return _markdown_inline_or_block(text)
        card = _node_popup_card(node)
        popup = (
            f'<span class="popup">'
            f'<span class="popup-label">{_html.escape(agent)} · node {node_id}</span>'
            f'{card}'
            f'</span>'
        )
        return (
            f'<span class="lbl" data-nid="{node_id}" data-agent="{_html.escape(agent)}">'
            f'{_markdown_inline_or_block(text)}{popup}'
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
    # st.components.v1.html only supports a fixed height (no auto-sizing at all,
    # which is why it's deprecated) — st.iframe's height="content" natively measures
    # the actual rendered srcdoc height on the backend, so the iframe fits its content
    # exactly and the chat page scrolls as a single unit with no dead space or clipping.
    st.iframe(full_html, height="content")
