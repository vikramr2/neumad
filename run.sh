#!/usr/bin/env bash
# NeuKRAG Multi-Agent Debate — entry point
#
# Usage:
#   ./run.sh "<query>"
#   ./run.sh                          # runs default paper queries
#   ./run.sh "<query>" --out out.json
#   NEUKRAG_MODE=adversarial ./run.sh "<query>"
#
# Runtime defaults live in mad/environment.json.
# Any variable exported in the shell overrides the JSON defaults.
#
# NEUKRAG_MODE          synthesis | adversarial
# NEUKRAG_DEBATE_ROUNDS max debate rounds (adversarial)
# NEUKRAG_DEBATE_LEVEL  0-3 tit-for-tat intensity (adversarial)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

source ~/vikram_venv/bin/activate
python "$SCRIPT_DIR/mad/orchestration.py" "$@"
