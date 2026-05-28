"""
Neuromorphic engineering specialist agent.

Focuses on energy efficiency and hardware implementability in memristor-based
architectures: power per spike, CMOS compatibility, on-chip learning, device
endurance, and analog noise tolerance.
"""
from __future__ import annotations

import dspy

ROLE = (
    "Neuromorphic Engineering Specialist: You prioritize energy efficiency and hardware "
    "implementability in memristor-based architectures — power per spike, CMOS compatibility, "
    "on-chip weight storage, device endurance, and analog noise tolerance. "
    "Evaluate every proposal through one lens: can this be built efficiently in silicon/memristors?"
)

LABEL = "Neuromorphic Engineering Specialist"


class NeuromorphicHypothesis(dspy.Signature):
    """You are a neuromorphic hardware engineer synthesizing a circuit hypothesis.
    Ground every claim in the provided KG evidence. Prioritize: energy efficiency,
    memristor device physics, CMOS implementability, on-chip learning, and power per spike.
    Write 2-3 focused scientific paragraphs."""

    query: str         = dspy.InputField(desc="Research query about neuromorphic computing")
    graph_context: str = dspy.InputField(desc="(head, relation, tail) triples from neuromorphic KG")
    hypothesis: str    = dspy.OutputField(desc="Energy-efficiency and memristor focused scientific hypothesis")
