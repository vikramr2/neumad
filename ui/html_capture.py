"""Headless-browser capture of a chat message's live-rendered DOM, for artifact export.

Streamlit renders markdown, expanders (native <details>), the popup-hover synthesis
labels (a self-contained srcdoc iframe), and the argumentation graph (inline Plotly
SVG) straight into the page. The most faithful way to snapshot "exactly what was on
screen" is to load the same session in a headless browser and serialize the real DOM,
rather than trying to reconstruct it server-side.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import re
from urllib.parse import urljoin

from playwright.sync_api import sync_playwright

from render_graph import _build_argumentation_graph_figure

APP_URL = os.environ.get("NEUMAD_APP_URL", "http://localhost:8501")

_CSS_URL_RE = re.compile(r'url\((["\']?)([^"\')]+)\1\)')


def _inline_css_urls(css_text: str, base_href: str, fetch) -> str:
    """Rewrite url(...) references in `css_text` to base64 data: URIs.

    Cross-origin @font-face loads are blocked by the browser's font-specific CORS
    policy (unlike images) — the Streamlit static server sends no
    Access-Control-Allow-Origin header, so a reference that just points an
    absolute URL back at the live server silently fails to load from any other
    origin (file://, or a different http:// port). That's what was actually
    causing the expander caret (the "keyboard_arrow_right" icon-font ligature) to
    render as raw overlapping text instead of an arrow glyph, which in turn
    covered and blocked clicks on the nested References summary. Inlining the
    font bytes as data: URIs sidesteps CORS entirely and also makes the artifact
    genuinely self-contained — it keeps rendering correctly even after the dev
    server is shut down.
    """
    def repl(m: re.Match) -> str:
        quote, url = m.group(1), m.group(2)
        if url.startswith("data:"):
            return m.group(0)
        abs_url = url if url.startswith(("http://", "https://", "//")) else urljoin(base_href, url)
        data = fetch(abs_url)
        if data is None:
            return m.group(0)
        mime = mimetypes.guess_type(abs_url)[0] or "application/octet-stream"
        b64 = base64.b64encode(data).decode("ascii")
        return f"url({quote}data:{mime};base64,{b64}{quote})"

    return _CSS_URL_RE.sub(repl, css_text)


def capture_response_html(
    msg_idx: int,
    *,
    graph_dict: dict | None = None,
    app_url: str = APP_URL,
    timeout_ms: int = 30_000,
) -> str:
    """Open `app_url` headlessly, find the chat message tagged `msg_idx`, and return a
    standalone HTML document with its rendered DOM and all CSS inlined.

    Chat history is persisted to disk and reloaded on every fresh session (see
    history_store._init_state), so a brand-new headless session shows the same
    transcript — and therefore the same message at the same index — as the live one.

    `graph_dict` (result["argumentation_graph"]) is optional: when given, the dead
    static Plotly SVG that got captured (a frozen snapshot with no Plotly.js runtime
    behind it, so hover does nothing) is replaced with a freshly rebuilt figure
    rendered via fig.to_html(), which embeds the actual Plotly.js library and a
    Plotly.newPlot() call — so the exported file has the same live hover/pan/zoom
    on the argumentation graph as the Streamlit app, with no server dependency.
    """
    marker_sel = f'[data-neumad-msg-idx="{msg_idx}"]'

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page()
            page.goto(app_url, timeout=timeout_ms)
            # state="attached": the marker is intentionally display:none, so the
            # default visible-wait would time out even though it's present.
            page.wait_for_selector(marker_sel, state="attached", timeout=timeout_ms)
            page.wait_for_load_state("networkidle", timeout=timeout_ms)
            page.wait_for_timeout(1500)  # let MathJax typeset and Plotly finish drawing

            if graph_dict:
                fig = _build_argumentation_graph_figure(graph_dict)
                if fig is not None:
                    fig_html = fig.to_html(
                        full_html=False, include_plotlyjs=True, config={"responsive": True}
                    )
                    # innerHTML doesn't execute <script> tags, but we don't need it to
                    # here — we only need the *markup* to be correct once it's
                    # serialized below; the script runs normally when the exported
                    # file is later opened as a real page in the user's browser.
                    page.evaluate(
                        """([markerSel, figHtml]) => {
                            const marker = document.querySelector(markerSel);
                            const msg = marker.closest('[data-testid="stChatMessage"]');
                            const wrapper = msg.querySelector('[data-testid="stPlotlyChart"]');
                            if (wrapper) wrapper.innerHTML = figHtml;
                        }""",
                        [marker_sel, fig_html],
                    )

            container = page.locator(marker_sel).locator(
                'xpath=ancestor::*[@data-testid="stChatMessage"]'
            )
            body_html = container.evaluate("el => el.outerHTML")

            # Emotion (Streamlit's CSS-in-JS) inserts rules straight into the CSSOM via
            # sheet.insertRule() rather than writing to the <style> tag's textContent,
            # so reading textContent silently returns near-empty strings — read
            # sheet.cssRules instead to get the rules that are actually in effect.
            css_chunks = page.evaluate("""
                () => Array.from(document.querySelectorAll('style')).map(el => {
                    try {
                        return Array.from(el.sheet.cssRules).map(r => r.cssText).join('\\n');
                    } catch (e) {
                        return el.textContent;
                    }
                })
            """)
            def _fetch_bytes(url: str) -> bytes | None:
                try:
                    resp = page.request.get(url)
                    return resp.body() if resp.ok else None
                except Exception:
                    return None

            stylesheet_hrefs = page.eval_on_selector_all(
                'link[rel="stylesheet"]', "els => els.map(e => e.href)"
            )
            for href in stylesheet_hrefs:
                try:
                    resp = page.request.get(href)
                    if resp.ok:
                        css_chunks.append(_inline_css_urls(resp.text(), href, _fetch_bytes))
                except Exception:
                    pass
        finally:
            browser.close()

    css = "\n".join(css_chunks)
    # Streamlit marks a collapsed expander's content wrapper `inert` (fully
    # non-interactive, for accessibility) and clears it via its own React toggle
    # handler when opened. That handler doesn't exist in a static export, so a
    # snapshot taken while collapsed freezes `inert` in place — clicking the
    # native <summary> still flips the real `open` attribute (plain browser
    # behavior), the content becomes visible, but everything inside it — e.g. a
    # nested "References" dropdown — stays unclickable. This listener restores
    # just that one piece of behavior generically for any <details>.
    _INERT_FIX_SCRIPT = (
        "<script>"
        "document.querySelectorAll('details').forEach(function(d){"
        "d.addEventListener('toggle', function(){"
        "if (d.open) d.querySelectorAll('[inert]').forEach(function(el){"
        "el.removeAttribute('inert'); }); }); });"
        "</script>"
    )
    # <base> lets any asset URL that wasn't inlined (e.g. an icon font) still resolve
    # by falling back to the live server, if it's still running.
    return (
        "<!DOCTYPE html>\n"
        '<html><head><meta charset="utf-8">'
        f'<base href="{app_url}/">'
        f"<style>{css}</style>"
        "</head><body>"
        f"{body_html}"
        f"{_INERT_FIX_SCRIPT}"
        "</body></html>"
    )
