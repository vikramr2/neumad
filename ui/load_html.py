"""Loads static HTML/CSS/JS snippets from ui/html/.

Long HTML/CSS/JS blocks embedded as triple-quoted Python strings render as one flat
color in most editors, which makes them much harder to skim than actual .html/.css/.js
files with real syntax highlighting. Anything fully static goes in ui/html/ and is read
in verbatim; anything needing per-call interpolation is loaded as a string.Template
using $name substitution rather than str.format()'s {name} — CSS and JS are full of
literal curly braces, which would constantly collide with {}-style placeholders.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from string import Template

_HTML_DIR = Path(__file__).parent / "html"


@lru_cache(maxsize=None)
def load(filename: str) -> str:
    """Read a static file from ui/html/ verbatim."""
    return (_HTML_DIR / filename).read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def load_template(filename: str) -> Template:
    """Read a file from ui/html/ as a $-substitution template."""
    return Template(load(filename))
