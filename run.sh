#!/usr/bin/env bash
# NeuKRAG Multi-Agent Debate — entry point
#
# Usage:
#   ./run.sh                          # Streamlit UI (default)
#   ./run.sh --ui                     # Streamlit UI (explicit)
#   ./run.sh "<query>"                # CLI, default paper queries
#   ./run.sh "<query>" --out out.json
#   NEUKRAG_MODE=adversarial ./run.sh "<query>"
#
# Runtime defaults live in mad/environment.json.
# Any shell-exported variable overrides the JSON defaults.
#
# NEUKRAG_MODE          synthesis | adversarial
# NEUKRAG_DEBATE_ROUNDS max debate rounds (adversarial)
# NEUKRAG_DEBATE_LEVEL  0-3 tit-for-tat intensity (adversarial)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

source ~/vikram_venv/bin/activate

if [[ $# -eq 0 || "${1:-}" == "--ui" ]]; then
    streamlit run "$SCRIPT_DIR/ui/app.py"
else
    python "$SCRIPT_DIR/mad/orchestration.py" "$@"
fi
