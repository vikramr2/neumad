"""dspy.Module wrappers around each NeuMAD chamber.

Each program holds the exact same SpecialistAgent/Mediator objects and calls the
exact same chamber orchestration function the live app uses — just packaged as a
single dspy.Module so GEPA can discover and jointly optimize every DSPy signature
involved (all three specialists' predictors plus the mediator's), not just one.
Verified separately that dspy.Module.predictors() recurses into named sub-module
attributes (self.neuroscience, self.aiml, ...), giving each nested predictor a
distinct dotted name (e.g. "neuroscience.hyp_predict", "mediator.graph_synthesis_predict")
even though the three specialists share the same SpecialistAgent class.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT    = Path(__file__).parent.parent
_MAD     = _ROOT / "mad"
_NEUKRAG = _ROOT / "neukrag"
_ARGORA  = _ROOT / "argora-public"
sys.path.insert(0, str(_MAD))
sys.path.insert(0, str(_NEUKRAG))
sys.path.insert(0, str(_ARGORA))

import dspy

from orchestration import CONFIG_PATH, load_toml, load_metadata  # noqa: E402
from agents.specialist import SpecialistAgent                    # noqa: E402
from agents.mediator import Mediator                              # noqa: E402
from chambers.synthesis import run_synthesis                      # noqa: E402
from chambers.adversarial import run_adversarial                  # noqa: E402
from chambers.choreographed import run_choreographed              # noqa: E402
from chambers.rotation import run_rotation                        # noqa: E402


def load_kg_paths_and_metadata(k_hops: int = 2, max_triples: int = 40):
    cfg      = load_toml(CONFIG_PATH)
    kg_cfg   = cfg.get("kg_paths", {})
    meta_cfg = cfg.get("metadata_paths", {})
    kg_paths = {
        name: Path(kg_cfg[f"{name}_kg"]).expanduser()
        for name in ("neuroscience", "aiml", "neuromorphic")
    }
    metadata = {
        name: load_metadata(Path(meta_cfg[f"{name}_metadata"]).expanduser())
        for name in ("neuroscience", "aiml", "neuromorphic")
        if meta_cfg.get(f"{name}_metadata")
    }
    return kg_paths, metadata


class _NeuMADProgram(dspy.Module):
    """Base class: builds the 3 specialists + mediator as named sub-modules."""

    def __init__(self, k_hops: int = 2, max_triples: int = 40):
        super().__init__()
        kg_paths, metadata = load_kg_paths_and_metadata(k_hops, max_triples)
        self.neuroscience = SpecialistAgent(
            "neuroscience", kg_paths["neuroscience"], k_hops, max_triples,
            metadata=metadata.get("neuroscience"),
        )
        self.aiml = SpecialistAgent(
            "aiml", kg_paths["aiml"], k_hops, max_triples,
            metadata=metadata.get("aiml"),
        )
        self.neuromorphic = SpecialistAgent(
            "neuromorphic", kg_paths["neuromorphic"], k_hops, max_triples,
            metadata=metadata.get("neuromorphic"),
        )
        self.mediator = Mediator()

    @property
    def _agents(self) -> list[SpecialistAgent]:
        return [self.neuroscience, self.aiml, self.neuromorphic]


class SynthesisProgram(_NeuMADProgram):
    MODE = "synthesis"

    def forward(self, query: str) -> dspy.Prediction:
        result = run_synthesis(query, self._agents, self.mediator)
        return dspy.Prediction(**result)


class AdversarialProgram(_NeuMADProgram):
    MODE = "adversarial"

    def __init__(self, *args, max_rounds: int = 3, debate_level: int = 2,
                 neuromorphic_mediator: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_rounds            = max_rounds
        self.debate_level          = debate_level
        self.neuromorphic_mediator = neuromorphic_mediator

    def forward(self, query: str) -> dspy.Prediction:
        result = run_adversarial(
            query, self._agents, self.mediator,
            max_rounds=self.max_rounds,
            debate_level=self.debate_level,
            neuromorphic_mediator=self.neuromorphic_mediator,
        )
        return dspy.Prediction(**result)


class ChoreographedProgram(_NeuMADProgram):
    MODE = "choreographed"

    def __init__(self, *args, neuromorphic_mediator: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.neuromorphic_mediator = neuromorphic_mediator

    def forward(self, query: str) -> dspy.Prediction:
        result = run_choreographed(
            query, self._agents, self.mediator,
            neuromorphic_mediator=self.neuromorphic_mediator,
        )
        return dspy.Prediction(**result)


class RotationProgram(_NeuMADProgram):
    MODE = "rotation"

    def __init__(self, *args, n_rotations: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_rotations = n_rotations

    def forward(self, query: str) -> dspy.Prediction:
        result = run_rotation(query, self._agents, self.mediator, n_rotations=self.n_rotations)
        return dspy.Prediction(**result)


# Registry the CLI uses to instantiate the right program for --mode. The "-nm"
# variants are the neuromorphic-mediated toggle, not a separate chamber.
PROGRAM_FACTORIES = {
    "synthesis":        lambda: SynthesisProgram(),
    "adversarial":      lambda: AdversarialProgram(neuromorphic_mediator=False),
    "adversarial-nm":   lambda: AdversarialProgram(neuromorphic_mediator=True),
    "choreographed":    lambda: ChoreographedProgram(neuromorphic_mediator=False),
    "choreographed-nm": lambda: ChoreographedProgram(neuromorphic_mediator=True),
    "rotation":         lambda: RotationProgram(),
}

# Which cc_metric extractor each --mode needs (only synthesis differs in shape).
METRIC_MODE = {
    "synthesis":        "synthesis",
    "adversarial":      "adversarial",
    "adversarial-nm":   "adversarial",
    "choreographed":    "choreographed",
    "choreographed-nm": "choreographed",
    "rotation":         "rotation",
}
