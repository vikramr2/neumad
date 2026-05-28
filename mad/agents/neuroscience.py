"""
Neuroscience specialist agent.

Focuses on biological plausibility and fidelity to known brain mechanisms:
ion channel / synaptic dynamics, cortical circuit organization, and
biologically grounded learning rules (STDP, BCM, homeostatic plasticity).
"""
from __future__ import annotations

import dspy

ROLE = (
    "Neuroscience Specialist: You prioritize biological plausibility and fidelity to known "
    "brain mechanisms — ion channel and synaptic dynamics, cortical circuit organization, "
    "and biologically realistic learning rules (e.g., STDP, BCM, homeostatic plasticity). "
    "Evaluate every proposal through one lens: does this faithfully model how the brain works?"
)

LABEL = "Neuroscience Specialist"


class NeuroscienceHypothesis(dspy.Signature):
    """You are a computational neuroscientist synthesizing a neuromorphic circuit hypothesis.
    Ground every claim in the provided KG evidence. Prioritize: biological plausibility,
    ion channel / synaptic dynamics, cortical circuit organization, and biologically realistic
    learning rules. Write 2-3 focused scientific paragraphs."""

    query: str         = dspy.InputField(desc="Research query about neuromorphic computing")
    graph_context: str = dspy.InputField(desc="(head, relation, tail) triples from neuroscience KG")
    hypothesis: str    = dspy.OutputField(desc="Bio-realism focused scientific hypothesis")
