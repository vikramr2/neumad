"""In-memory Combinatorial Creativity (CC) scorer for GEPA optimization.

Reuses the exact same pure functions eval/cc/compute_cc.py uses to score saved
artifacts (assemble_creative_landscape, compute_novelty, compute_utility, DOI
matching against the citation KGs) — just applied directly to a chamber's
in-memory result dict instead of round-tripping through responses/*.json and
kg_triples/*.csv on disk. GEPA calls the metric on every rollout, so avoiding
file I/O per call matters here.

Score = novelty * utility, matching compute_cc.py's "final synthesis" row: the
accumulated triples/references across every agent and every round, exactly
what compute_cc.py computes as the whole debate's creativity score.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import dspy
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

_ROOT    = Path(__file__).parent.parent
_EVAL_CC = _ROOT / "eval" / "cc"
sys.path.insert(0, str(_EVAL_CC))

from build_landscape import load_kg_all, assemble_creative_landscape  # noqa: E402
from compute_cc import (                                              # noqa: E402
    build_doi_index, extract_dois, compute_novelty, compute_utility, paper_subgraph_nodes,
)

CONFIG_PATH = _ROOT / "config.toml"
CACHE_DIR   = _EVAL_CC / "cache"


def _load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


class CreativityScorer:
    """Loads the kg_all landscape graph + DOI index once (both are cached on disk
    already, see eval/cc/cache/), then scores any accumulated (triples, references)
    pair from a debate/synthesis run in-memory."""

    def __init__(
        self,
        config_path: Path = CONFIG_PATH,
        cache_dir: Path = CACHE_DIR,
        alpha_h: float = 0.5,
        alpha_r: float = 0.5,
        alpha_R: float = 1.0,
    ):
        cfg = _load_toml(config_path)
        self.graph         = load_kg_all(config_path, cache_dir)
        self.doi_index     = build_doi_index(cfg)
        self.doc_node_cache: dict = {}
        self.alpha_h, self.alpha_r, self.alpha_R = alpha_h, alpha_r, alpha_R

    def score(self, triples: list[dict], references: str) -> dict:
        """triples: accumulated {"h","r","t","document_id"} dicts across every agent
        and round. references: accumulated APA-citation text (DOIs are pulled out of
        it via regex, same as compute_cc.py).

        Returns {"novelty", "utility", "creativity", "ref_grounded_frac", "n_refs"}.
        novelty/utility/creativity are intentionally UNBOUNDED — per the CC paper both
        scale with how much material accumulated (more triples -> longer assembled walk;
        more cited references -> larger U_R multiplier). That's meaningful for comparing
        artifacts of similar size post-hoc, but not usable directly as a GEPA score (GEPA
        treats score >= perfect_score=1.0 as "solved" and stops trying to improve it —
        see make_cc_metric below for the bounded transform). ref_grounded_frac is a
        genuinely bounded [0,1] diagnostic (fraction of cited references whose KG nodes
        were actually touched), used only for the feedback text.
        """
        if not triples:
            return {"novelty": 0.0, "utility": 0.0, "creativity": 0.0, "ref_grounded_frac": 1.0, "n_refs": 0}

        P_nodes = {str(t.get("h", "")) for t in triples} | {str(t.get("t", "")) for t in triples}
        refs    = list(set(extract_dois(references)))

        P_dict, pi = assemble_creative_landscape(triples, self.graph)
        novelty = compute_novelty(pi, P_dict, self.alpha_h, self.alpha_r)
        utility = compute_utility(
            P_nodes, set(), set(), refs,
            self.doi_index, self.doc_node_cache,
            alpha_R=self.alpha_R,
        )

        if refs:
            used = sum(
                1 for doi in refs
                if paper_subgraph_nodes(doi, self.doi_index, self.doc_node_cache) & P_nodes
            )
            ref_grounded_frac = used / len(refs)
        else:
            ref_grounded_frac = 1.0

        return {
            "novelty": novelty,
            "utility": utility,
            "creativity": novelty * utility,
            "ref_grounded_frac": ref_grounded_frac,
            "n_refs": len(refs),
        }


# ---------------------------------------------------------------------------
# Accumulating triples/references out of a chamber's result — mirrors
# compute_cc.py's "final synthesis" row (union across every agent, every round).
# ---------------------------------------------------------------------------

def accumulate_from_debate_history(pred: dspy.Prediction) -> tuple[list[dict], str]:
    """adversarial / choreographed / rotation all return debate_history: a flat list
    of per-round-per-agent entries with "triples" and "references" fields."""
    all_triples: list[dict] = []
    all_refs: list[str] = []
    for entry in pred.get("debate_history", []) or []:
        all_triples.extend(entry.get("triples") or [])
        refs = entry.get("references") or ""
        if refs:
            all_refs.append(refs)
    return all_triples, "\n".join(all_refs)


def accumulate_from_agent_hypotheses(pred: dspy.Prediction) -> tuple[list[dict], str]:
    """synthesis mode's result shape is {agent_name: {"triples":..., "references":...}}
    rather than a flat debate_history list."""
    all_triples: list[dict] = []
    all_refs: list[str] = []
    for data in (pred.get("agent_hypotheses", {}) or {}).values():
        all_triples.extend(data.get("triples") or [])
        refs = data.get("references") or ""
        if refs:
            all_refs.append(refs)
    return all_triples, "\n".join(all_refs)


_EXTRACTORS = {
    "synthesis": accumulate_from_agent_hypotheses,
}


def extract_for_mode(mode: str, pred: dspy.Prediction) -> tuple[list[dict], str]:
    return _EXTRACTORS.get(mode, accumulate_from_debate_history)(pred)


# ---------------------------------------------------------------------------
# GEPA feedback metric
# ---------------------------------------------------------------------------

def make_cc_metric(scorer: CreativityScorer, mode: str, scale: float = 150.0):
    """Build a GEPAFeedbackMetric for the given mode. GEPA calls this both at the
    program level (pred_name=None) and per-predictor during reflection (pred_name
    set) — we return the same whole-run creativity score either way, since CC is a
    property of the accumulated debate output, not attributable to a single call.

    GEPA's score MUST live in [0, perfect_score] (default perfect_score=1.0) — it
    uses score >= perfect_score as a "this example is already solved, stop trying to
    improve it" signal. Raw creativity is unbounded (grows with how much material
    accumulated), so a raw score of e.g. 185 makes GEPA think every rollout is
    already "perfect" and skip reflection entirely on iteration 0 — confirmed live,
    this silently wastes the whole optimization budget doing nothing. `scale` maps
    raw creativity through score = creativity / (creativity + scale), a monotonic
    squash into [0, 1) — ordering between candidates is preserved, GEPA just never
    sees a false "solved" signal. `scale` is the creativity value that maps to 0.5;
    default 150 is calibrated from one live rotation-mode rollout (creativity ~185)
    and should be revisited once you've seen scores from more modes/queries.
    """

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> ScoreWithFeedback:
        triples, refs = extract_for_mode(mode, pred)
        s = scorer.score(triples, refs)
        bounded_score = s["creativity"] / (s["creativity"] + scale) if s["creativity"] > 0 else 0.0

        feedback = (
            f"Combinatorial creativity = {s['creativity']:.2f} "
            f"(novelty={s['novelty']:.2f}, utility={s['utility']:.2f}, "
            f"{s['n_refs']} cited reference(s), {s['ref_grounded_frac']*100:.0f}% of them "
            f"actually grounded in the retrieved KG triples). Normalized score = {bounded_score:.3f}."
        )
        if not triples:
            feedback += " No knowledge-graph triples were retrieved/recorded at all — check grounding."
        elif s["n_refs"] > 0 and s["ref_grounded_frac"] < 0.5:
            feedback += (
                " Less than half of the cited references' KG nodes were actually touched "
                "by the retrieved argument — only cite papers whose specific content the "
                "argument draws on, and ground claims directly in the KG triples provided."
            )
        return ScoreWithFeedback(score=bounded_score, feedback=feedback)

    return metric
