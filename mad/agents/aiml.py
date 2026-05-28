"""
AI/ML specialist agent.

Focuses on measurable learning performance: accuracy on benchmarks, generalization,
convergence speed, and task-adaptive training objectives.
"""
from __future__ import annotations

import dspy

ROLE = (
    "AI/ML Specialist: You prioritize measurable learning performance — accuracy on benchmarks, "
    "generalization, convergence speed, and task-adaptive training. You care about loss "
    "landscapes, gradient flow, and whether a proposed model improves over existing baselines. "
    "Evaluate every proposal through one lens: does this achieve better accuracy or learning outcomes?"
)

LABEL = "AI/ML Specialist"


class AIMLHypothesis(dspy.Signature):
    """You are an AI/ML researcher synthesizing a neuromorphic circuit hypothesis.
    Ground every claim in the provided KG evidence. Prioritize: accuracy on benchmarks,
    learning efficiency, generalization, convergence speed, and task-adaptive training.
    Write 2-3 focused scientific paragraphs."""

    query: str         = dspy.InputField(desc="Research query about neuromorphic computing")
    graph_context: str = dspy.InputField(desc="(head, relation, tail) triples from AI/ML KG")
    hypothesis: str    = dspy.OutputField(desc="Performance-focused scientific hypothesis")
