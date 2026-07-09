#!/usr/bin/env python3
"""
optimization/gepa_optimize.py — GEPA-CC optimization for NeuMAD's modes.

Optimizes every DSPy signature in a NeuMAD chamber (all three SpecialistAgents'
predictors + the Mediator's) jointly, using GEPA's reflective evolutionary search,
scored by Combinatorial Creativity (novelty * utility — see eval/cc/compute_cc.py)
computed in-memory on each rollout's accumulated KG triples and citations.

Usage:
    python gepa_optimize.py --mode rotation --estimate-only
    python gepa_optimize.py --mode adversarial-nm --max-metric-calls 60
    python gepa_optimize.py --mode all --max-metric-calls 60 --num-threads 4

Modes: synthesis, adversarial, adversarial-nm, choreographed, choreographed-nm,
rotation, all (-nm = neuromorphic-mediated toggle, not a separate chamber).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "mad"))
sys.path.insert(0, str(_ROOT / "neukrag"))
sys.path.insert(0, str(_ROOT / "argora-public"))
sys.path.insert(0, str(Path(__file__).parent))

import dspy

from cc_metric import CreativityScorer, make_cc_metric
from programs import PROGRAM_FACTORIES, METRIC_MODE

DEFAULT_QUERIES_PATH = _ROOT / "eval" / "ashish_qs.txt"
RESULTS_DIR          = Path(__file__).parent / "results"

# For now both task and reflection LM are the same model per request; only the
# temperature differs, chosen for what each role actually does:
DEFAULT_MODEL = "openai/openai/gpt-oss-120b"

# Task LM executes the actual hypothesis/debate/mediation generation. The metric
# being optimized is COMBINATORIAL CREATIVITY (novelty x utility) — novelty
# specifically rewards drawing lower-probability (surprising) connections between
# KG concepts, so a high sampling temperature is what the objective wants: too low
# and the model just greedily restates the most obvious/high-probability claim,
# which directly suppresses novelty.
DEFAULT_TASK_TEMPERATURE = 1.0

# Reflection LM reads a batch of scored trajectories + feedback and proposes
# improved instructions for whichever predictor GEPA is currently mutating. This
# is part analytical (correctly diagnose *why* a prompt scored low from the
# feedback text) and part generative (propose a meaningfully different rewrite
# each iteration, not converge on minor rephrasings — GEPA's search is evolutionary,
# so proposal diversity matters). 0.7 sits below the pure-creative task setting to
# keep the diagnosis grounded, while staying well above a near-deterministic
# setting that would make successive mutations of the same predictor too similar.
DEFAULT_REFLECTION_TEMPERATURE = 0.7

# ---------------------------------------------------------------------------
# Per-rollout LLM call count estimates — derived by hand-tracing each chamber's
# code path (mad/chambers/*.py). These are NOT used by the metric; they only
# drive the --estimate-only cost projection so you can pick a sane budget before
# spending real API time. Rough by construction — actual counts vary with the
# adaptive early-stop in adversarial mode (this assumes it never fires, i.e. an
# upper bound) and with how many Γ+ε arguments/peer reactions the model actually
# returns (assumes both always succeed).
# ---------------------------------------------------------------------------

def estimate_calls_per_rollout(mode: str, max_rounds: int = 3, n_rotations: int = 1) -> int:
    base_mode = mode.removesuffix("-nm")
    is_nm = mode.endswith("-nm")

    if base_mode == "synthesis":
        # 3 agents x (1 initial_hypothesis + 1 parse + 2x(arg_miner + strength_attr))
        # + build_argument_graph (3x2 peer_elicit) + graph_synthesis + synthesis_provenance
        return 3 * 6 + 6 + 1 + 1

    if base_mode == "adversarial":
        opening  = 1 if is_nm else 3
        debaters = 2 if is_nm else 3
        calls = opening * 6  # hyp_predict + build_local_arguments (parse + 2x(miner+attr))
        for r in range(1, max_rounds + 1):
            calls += debaters * 1  # debate_response
            if is_nm and r == 1:
                calls += debaters * 5  # retroactive build_local_arguments (retrieve_context is no LLM call)
            else:
                calls += debaters * 2  # annotate_transitions: extract_main_claim + classify_transition
            calls += 1  # can_conclude
        calls += 6 + 1  # final: build_argument_graph peer_elicit (3x2) + extract_answer/mediate_as_agent
        return calls

    if base_mode == "choreographed":
        opening  = 1 if is_nm else 3
        debaters = 2 if is_nm else 3
        calls  = opening * 6                                   # round 1
        calls += debaters * 1                                  # round 2 debate_response
        calls += debaters * (5 if is_nm else 2)                # round 2: baseline QBAF build, or transitions
        calls += debaters * 1                                  # round 3 debate_response
        calls += debaters * 2                                  # round 3 annotate_transitions (always)
        calls += 6 + 1                                          # round 4: peer_elicit (3x2) + synthesis
        calls += debaters * 1                                  # round 5 review
        calls += debaters * 2                                  # round 5 annotate_transitions
        return calls

    if base_mode == "rotation":
        calls = 2  # initial_hypothesis + extract_main_claim (position_history seed)
        cycle_len = 2  # aiml, neuroscience
        for _ in range(n_rotations):
            calls += cycle_len * (1 + 2)  # edit_position + annotate_transitions (main_claim + classify)
            calls += 1 + 2                 # anchor's closing edit_position + annotate_transitions
        return calls

    raise ValueError(f"Unknown mode: {mode}")


def print_estimate(mode: str, n_train: int, n_val: int, max_metric_calls: int,
                    avg_call_seconds: float, num_threads: int,
                    max_rounds: int, n_rotations: int) -> None:
    calls_per_rollout = estimate_calls_per_rollout(mode, max_rounds, n_rotations)
    # max_metric_calls roughly bounds total rollouts across train batches + periodic
    # valset Pareto checks; treat it as the total rollout count for this projection.
    total_rollouts  = max_metric_calls
    total_llm_calls = total_rollouts * calls_per_rollout
    serial_seconds  = total_llm_calls * avg_call_seconds
    parallel_seconds = serial_seconds / max(1, num_threads)

    def fmt(seconds: float) -> str:
        h = seconds / 3600
        if h < 1:
            return f"{seconds/60:.0f} min"
        return f"{h:.1f} h"

    print(f"--- {mode} ---")
    print(f"  trainset={n_train}  valset={n_val}  max_metric_calls={max_metric_calls}")
    print(f"  ~{calls_per_rollout} LLM calls/rollout (max_rounds={max_rounds}, n_rotations={n_rotations})")
    print(f"  ~{total_llm_calls:,} total LLM calls projected")
    print(f"  ~{fmt(serial_seconds)} serial (num_threads=1)")
    print(f"  ~{fmt(parallel_seconds)} at num_threads={num_threads}  (assumes the endpoint tolerates that concurrency — unverified)")


# ---------------------------------------------------------------------------
# Trainset / valset
# ---------------------------------------------------------------------------

def load_queries(path: Path) -> list[str]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln]


def build_examples(queries: list[str]) -> list[dspy.Example]:
    return [dspy.Example(query=q).with_inputs("query") for q in queries]


def split_train_val(queries: list[str], train_frac: float) -> tuple[list[str], list[str]]:
    n_train = max(1, round(len(queries) * train_frac))
    n_train = min(n_train, len(queries) - 1) if len(queries) > 1 else len(queries)
    return queries[:n_train], queries[n_train:]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_one_mode(args, mode: str, scorer: CreativityScorer) -> None:
    queries = load_queries(args.queries)
    train_q, val_q = split_train_val(queries, args.train_frac)
    if not val_q:
        val_q = train_q  # tiny dataset fallback — still lets GEPA track a Pareto front

    print_estimate(
        mode, len(train_q), len(val_q), args.max_metric_calls,
        args.avg_call_seconds, args.num_threads, args.max_rounds, args.n_rotations,
    )
    if args.estimate_only:
        return

    trainset = build_examples(train_q)
    valset   = build_examples(val_q)

    factory = PROGRAM_FACTORIES[mode]
    program = factory()
    if hasattr(program, "max_rounds"):
        program.max_rounds = args.max_rounds
    if hasattr(program, "n_rotations"):
        program.n_rotations = args.n_rotations

    metric = make_cc_metric(scorer, METRIC_MODE[mode], scale=args.score_scale)

    task_lm = dspy.LM(
        args.task_model, api_base=args.api_base, api_key=args.api_key,
        temperature=args.task_temperature, cache=False,
    )
    reflection_lm = dspy.LM(
        args.reflection_model, api_base=args.api_base, api_key=args.api_key,
        temperature=args.reflection_temperature, cache=False,
    )
    dspy.configure(lm=task_lm)

    optimizer = dspy.GEPA(
        metric=metric,
        reflection_lm=reflection_lm,
        max_metric_calls=args.max_metric_calls,
        num_threads=args.num_threads,
        track_stats=True,
    )

    print(f"\nStarting GEPA optimization for mode={mode} ...")
    t0 = time.time()
    optimized = optimizer.compile(student=program, trainset=trainset, valset=valset)
    elapsed = time.time() - t0
    print(f"Done in {elapsed/3600:.2f}h ({elapsed:.0f}s)")

    out_dir = args.out_dir / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    optimized.save(str(out_dir / "optimized_program.json"))
    with open(out_dir / "run_info.json", "w") as f:
        json.dump({
            "mode": mode,
            "elapsed_seconds": elapsed,
            "max_metric_calls": args.max_metric_calls,
            "train_queries": train_q,
            "val_queries": val_q,
            "task_model": args.task_model,
            "task_temperature": args.task_temperature,
            "reflection_model": args.reflection_model,
            "reflection_temperature": args.reflection_temperature,
        }, f, indent=2)
    print(f"Saved optimized program to {out_dir}/optimized_program.json")


def main():
    p = argparse.ArgumentParser(description="GEPA-CC optimize a NeuMAD mode")
    p.add_argument("--mode", required=True,
                   choices=list(PROGRAM_FACTORIES.keys()) + ["all"])
    p.add_argument("--queries", type=Path, default=DEFAULT_QUERIES_PATH)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--max-metric-calls", type=int, default=60,
                   help="Rough rollout budget for GEPA (default is deliberately "
                        "conservative — see the printed estimate before raising it).")
    p.add_argument("--num-threads", type=int, default=1)
    p.add_argument("--max-rounds", type=int, default=3, help="adversarial only")
    p.add_argument("--n-rotations", type=int, default=1, help="rotation only")
    p.add_argument("--avg-call-seconds", type=float, default=15.0,
                   help="Assumed avg per-LLM-call latency, only for the time estimate.")
    p.add_argument("--estimate-only", action="store_true",
                   help="Print the projected call count/time and exit without running GEPA.")
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--task-model", default=DEFAULT_MODEL)
    p.add_argument("--reflection-model", default=DEFAULT_MODEL)
    p.add_argument("--task-temperature", type=float, default=DEFAULT_TASK_TEMPERATURE)
    p.add_argument("--reflection-temperature", type=float, default=DEFAULT_REFLECTION_TEMPERATURE)
    p.add_argument("--score-scale", type=float, default=150.0,
                   help="Raw creativity value mapped to a normalized score of 0.5 "
                        "(GEPA needs scores in [0,1]; raw creativity is unbounded). "
                        "Recalibrate after seeing scores from a real run.")
    p.add_argument("--api-base", default=None)
    p.add_argument("--api-key", default=None)
    args = p.parse_args()

    if args.api_base is None or args.api_key is None:
        from run_neukrag import OLLAMA_BASE_URL, OLLAMA_API_KEY
        args.api_base = args.api_base or OLLAMA_BASE_URL
        args.api_key  = args.api_key or OLLAMA_API_KEY

    modes = list(PROGRAM_FACTORIES.keys()) if args.mode == "all" else [args.mode]

    scorer = None
    if not args.estimate_only:
        print("Loading kg_all creative-landscape graph + DOI index (cached after first run)...")
        scorer = CreativityScorer()

    for mode in modes:
        run_one_mode(args, mode, scorer)


if __name__ == "__main__":
    main()
