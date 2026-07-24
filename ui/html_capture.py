"""Headless-browser capture of a chat message's live-rendered DOM, for artifact export.

Streamlit renders markdown, expanders (native <details>), the popup-hover synthesis
labels (a self-contained srcdoc iframe), and the argumentation graph (inline Plotly
SVG) straight into the page. The most faithful way to snapshot "exactly what was on
screen" is to load the same session in a headless browser and serialize the real DOM,
rather than trying to reconstruct it server-side.
"""

from __future__ import annotations

import os

from playwright.sync_api import sync_playwright

APP_URL = os.environ.get("NEUMAD_APP_URL", "http://localhost:8501")


def capture_response_html(msg_idx: int, *, app_url: str = APP_URL, timeout_ms: int = 30_000) -> str:
    """Open `app_url` headlessly, find the chat message tagged `msg_idx`, and return a
    standalone HTML document with its rendered DOM and all CSS inlined.

    Chat history is persisted to disk and reloaded on every fresh session (see
    history_store._init_state), so a brand-new headless session shows the same
    transcript — and therefore the same message at the same index — as the live one.
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
            stylesheet_hrefs = page.eval_on_selector_all(
                'link[rel="stylesheet"]', "els => els.map(e => e.href)"
            )
            for href in stylesheet_hrefs:
                try:
                    resp = page.request.get(href)
                    if resp.ok:
                        css_chunks.append(resp.text())
                except Exception:
                    pass
        finally:
            browser.close()

    css = "\n".join(css_chunks)
    # <base> lets any asset URL that wasn't inlined (e.g. an icon font) still resolve
    # by falling back to the live server, if it's still running.
    return (
        "<!DOCTYPE html>\n"
        '<html><head><meta charset="utf-8">'
        f'<base href="{app_url}/">'
        f"<style>{css}</style>"
        "</head><body>"
        f"{body_html}"
        "</body></html>"
    )
