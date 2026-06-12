#!/usr/bin/env python3
"""
Usage: python render_artifact.py <artifact_folder>

Renders the responses/ folder of an artifact into a markdown report
saved as <artifact_folder>/report.md.
"""

import json
import re
import sys
from pathlib import Path

AGENT_LABELS = {
    "aiml": "AI/ML",
    "neuromorphic": "Neuromorphic Engineering",
    "neuroscience": "Neuroscience",
    "mediator": "Mediator",
    "neukrag": "NeuKRAG",
}

MODE_LABELS = {
    "choreographed": "Choreographed",
    "adversarial": "Adversarial",
    "neukrag": "NeuKRAG",
    "neukrag_inter": "NeuKRAG (Inter-Agent)",
    "synthesis": "Synthesis",
}


def agent_label(agent: str) -> str:
    return AGENT_LABELS.get(agent.lower(), agent.title())


def parse_filename(name: str) -> tuple[int, str] | None:
    m = re.match(r"round_(\d+)_(.+)\.json", name)
    if m:
        return int(m.group(1)), m.group(2)
    return None


def render(artifact_dir: Path) -> str:
    responses_dir = artifact_dir / "responses"
    if not responses_dir.is_dir():
        sys.exit(f"No responses/ folder found in: {artifact_dir}")

    synthesis_path = responses_dir / "final_synthesis.json"
    synthesis = None
    if synthesis_path.exists():
        with synthesis_path.open() as f:
            synthesis = json.load(f)

    mode = synthesis.get("mode", "unknown") if synthesis else "unknown"
    query = synthesis.get("query", "") if synthesis else ""

    lines: list[str] = []

    lines.append(f"# {artifact_dir.name}")
    lines.append("")
    if query:
        lines.append(f"**Query:** {query}")
        lines.append("")
    lines.append(f"**Mode:** {MODE_LABELS.get(mode, mode.title())}")
    lines.append("")
    lines.append("---")
    lines.append("")

    round_files: list[tuple[int, str, Path]] = []
    for path in sorted(responses_dir.glob("round_*.json")):
        parsed = parse_filename(path.name)
        if parsed:
            round_num, agent = parsed
            round_files.append((round_num, agent, path))

    rounds: dict[int, list[tuple[str, dict]]] = {}
    for round_num, agent, path in round_files:
        with path.open() as f:
            data = json.load(f)
        rounds.setdefault(round_num, []).append((agent, data))

    for round_num in sorted(rounds.keys()):
        agents_in_round = rounds[round_num]
        is_mediator_round = any(a == "mediator" for a, _ in agents_in_round)
        round_title = f"Mediator Synthesis (Round {round_num})" if is_mediator_round else f"Round {round_num}"
        lines.append(f"## {round_title}")
        lines.append("")

        for agent, data in agents_in_round:
            lines.append(f"### {agent_label(agent)}")
            lines.append("")

            statement = data.get("statement", "").strip()
            if statement:
                lines.append(statement)
                lines.append("")

            references = data.get("references", "").strip()
            if references:
                lines.append("**References:**")
                lines.append("")
                for ref in references.splitlines():
                    ref = ref.strip()
                    if ref:
                        lines.append(f"- {ref}")
                lines.append("")

            agreed = data.get("agreed")
            if agreed is not None:
                lines.append(f"*Agreement status: {agreed}*")
                lines.append("")

        lines.append("---")
        lines.append("")

    if synthesis and "final_hypothesis" in synthesis:
        lines.append("## Final Hypothesis")
        lines.append("")
        lines.append(synthesis["final_hypothesis"].strip())
        lines.append("")

    return "\n".join(lines)


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: render_artifact.py <artifact_folder>")

    artifact_dir = Path(sys.argv[1]).resolve()
    if not artifact_dir.is_dir():
        sys.exit(f"Not a directory: {artifact_dir}")

    doc = render(artifact_dir)
    output_path = artifact_dir / "report.md"
    output_path.write_text(doc)
    print(f"Written to {output_path}")


if __name__ == "__main__":
    main()
