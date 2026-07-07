"""Shared filesystem paths for the NeuMAD UI."""

from __future__ import annotations

from pathlib import Path

ROOT          = Path(__file__).parent.parent
HISTORY_FILE  = ROOT / "chat_history.json"
ARTIFACTS_DIR = ROOT / "artifacts"
