# NeuMAD — Neuromorphic Multi-Agent Debate

NeuMAD is a multi-agent debate system for neuromorphic computing research. Three domain-specialist LLM agents — Neuroscience, AI/ML, and Neuromorphic Engineering — each query their own knowledge graph, generate hypotheses, and build formal argumentation structures. A mediator merges those structures, computes dialectical strengths, and synthesizes a final hypothesis grounded in the debate.

---

## System Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER QUERY                                 │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 ▼
       ┌────────────┐    ┌────────────┐    ┌────────────┐
       │Neuroscience│    │   AI/ML   │    │Neuromorphic│
       │   Agent   │    │   Agent   │    │   Agent   │
       └─────┬──────┘    └─────┬──────┘    └─────┬──────┘
             │                 │                 │
             ▼                 ▼                 ▼
       ┌────────────┐    ┌────────────┐    ┌────────────┐
       │KG Retrieval│    │KG Retrieval│    │KG Retrieval│
       │BFS k-hops  │    │BFS k-hops  │    │BFS k-hops  │
       │domain KG   │    │domain KG   │    │domain KG   │
       └─────┬──────┘    └─────┬──────┘    └─────┬──────┘
             │                 │                 │
             ▼                 ▼                 ▼
       ┌────────────┐    ┌────────────┐    ┌────────────┐
       │  Initial   │    │  Initial   │    │  Initial   │
       │ Hypothesis │    │ Hypothesis │    │ Hypothesis │
       └─────┬──────┘    └─────┬──────┘    └─────┬──────┘
             │                 │                 │
             ▼                 ▼                 ▼
       ┌────────────┐    ┌────────────┐    ┌────────────┐
       │  Γ + ε     │    │  Γ + ε     │    │  Γ + ε     │
       │ ArgLLMs    │    │ ArgLLMs    │    │ ArgLLMs    │
       │ local QBAF │    │ local QBAF │    │ local QBAF │
       └─────┬──────┘    └─────┬──────┘    └─────┬──────┘
             │                 │                 │
             └─────────────────┼─────────────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │       DEBATE CHAMBER           │
              │  (synthesis / adversarial /    │
              │   choreographed)               │
              └───────────────┬────────────────┘
                              │
                              ▼
              ┌────────────────────────────────┐
              │     MArgE GRAPH CONSTRUCTION   │
              │  merge local QBAFs + cross-    │
              │  agent peer reactions          │
              └───────────────┬────────────────┘
                              │
                              ▼
              ┌────────────────────────────────┐
              │   DFQuAD STRENGTH PROPAGATION  │
              │   ε (intrinsic) → σ (dialectic)│
              └───────────────┬────────────────┘
                              │
                              ▼
              ┌────────────────────────────────┐
              │     MEDIATOR SYNTHESIS         │
              │  graph-aware, <label> tags     │
              │  for provenance               │
              └───────────────┬────────────────┘
                              │
                              ▼
              ┌────────────────────────────────┐
              │          RESULT UI             │
              │  Plotly QBAF · hover cards    │
              │  MathJax · provenance labels  │
              └────────────────────────────────┘
```

---

## Debate Chambers

Each chamber controls how many rounds agents debate and what instructions they receive before the mediator constructs the argumentation graph.

### Synthesis (1 round)

The simplest mode. Agents generate their hypotheses and ArgLLMs structures in parallel — there are no debate rounds. The mediator immediately merges the three local QBAFs and synthesizes.

```
Agent 1 ──┐
Agent 2 ──┼──► MArgE ──► DFQuAD ──► Synthesis
Agent 3 ──┘
```

### Adversarial (1–5 rounds, adaptive)

Implements the MAD (Multi-Agent Debate) protocol with four tit-for-tat levels (0 = full consensus, 3 = forced disagreement). A discriminative mediator judge decides after each round whether the debate has reached a satisfactory answer, allowing early termination.

```
Round 0:  initial hypotheses + Γ+ε QBAFs
Round 1:  rebuttals (tit-for-tat level governs intensity)
Round 2:  rebuttals
  ...
Round N:  mediator judge: "concluded?" ──yes──► stop
                                        │
                                       no
                                        │
                                   next round
                                        │
                                (after max rounds)
                                        │
                                        ▼
                           MArgE ──► DFQuAD ──► Extract Answer
```

### Choreographed (5 fixed rounds)

A scripted arc that forces a specific conversational shape. Agent covariance (how much agents are expected to agree) is explicitly specified per round.

```
Round 1 — Establishing Positions   (covariance: moderate)
          initial hypotheses + Γ+ε QBAFs built here

Round 2 — Adversarial Challenge    (covariance: low)
          agents must disagree on every point

Round 3 — Finding Convergence      (covariance: high)
          agents seek common ground across domains

Round 4 — Mediator Synthesis       (covariance: none)
          MArgE ──► DFQuAD ──► graph-aware synthesis

Round 5 — Reviewing Synthesis      (covariance: moderate-high)
          each agent evaluates and accepts/rejects synthesis
```

---

## Argumentation Graph Construction

The graph is built using the ArgLLMs method (Çelik et al., AAAI 2025), extended to multiple agents via the MArgE aggregation scheme.

### Step 1 — Γ: Argument Generation (per agent)

Each SpecialistAgent generates two arguments about its own main claim, grounded in its KG triples: one supporting, one attacking. The LLM is instructed to return exactly `N/A` if no valid argument exists.

```
  main_claim (agent's hypothesis)
       │
       ├── Γ(supporting) ──► "Cortical column topology enables..."
       └── Γ(attacking)  ──► "Biological variability limits..."
```

DSPy signature: `AgentArgumentMiner`
- inputs: `query`, `agent_role`, `graph_context` (KG triples), `main_claim`, `polarity`
- output: `argument` (1–2 sentences, or "N/A")

### Step 2 — ε: Intrinsic Strength Attribution (per agent)

The same specialist LLM scores each generated argument on [0, 100]. This score reflects domain-calibrated confidence — the neuroscience agent only scores neuroscience arguments, the AI/ML agent only AI/ML arguments. The score is normalized to τ ∈ [0, 1] and stored as the node's base score in the QBAF.

```
  argument ──► ε("supporting", main_claim) ──► confidence: 73
                                                ↓
                                           τ = 0.73
```

DSPy signature: `ArgumentStrengthAttributor`
- inputs: `agent_role`, `argument`, `parent_claim`, `polarity`
- output: `confidence` (integer 0–100)

This is intentionally domain-scoped: cross-domain scoring would conflate domain expertise with argument quality, violating the locality assumption behind per-agent QBAFs.

### Step 3 — MArgE: Multi-Agent Graph Merge (mediator)

The mediator constructs a joint QBAF from the three local QBAFs, then adds cross-agent peer reactions.

```
  Local QBAFs (per-agent, τ = ε scores)
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │ Neuroscience │  │    AI/ML     │  │ Neuromorphic │
  │  main_claim  │  │  main_claim  │  │  main_claim  │
  │  support(τ)  │  │  support(τ)  │  │  support(τ)  │
  │  attack(τ)   │  │  attack(τ)   │  │  attack(τ)   │
  └──────────────┘  └──────────────┘  └──────────────┘
         │                 │                 │
         └─────────────────┼─────────────────┘
                           │  merge
                           ▼
              Joint QBAF (all 9 nodes)
                           │
                           │  + peer reactions
                           ▼
  For each (target_agent, peer_agent) pair:
  ┌────────────────────────────────────────────┐
  │ PeerArgumentElicitor asks peer_agent:      │
  │   "Do you agree with target_agent's claim?"│
  │                                            │
  │ stance = agree  ──► support edge, τ = 0.5 │
  │ stance = disagree ► attack  edge, τ = 0.5 │
  └────────────────────────────────────────────┘
```

Peer reactions use τ = 0.5 (neutral) because cross-domain confidence is not domain-expert-assessed; only the dialectical structure (agree vs. disagree) carries meaning.

DSPy signature: `PeerArgumentElicitor`
- inputs: `query`, `author_name`, `main_argument`, `peer_name`, `peer_hypothesis`
- outputs: `stance` (agree/disagree), `reasoning`

---

## Intrinsic vs. Dialectical Strength

Every node in the QBAF carries two scores that are displayed separately in the UI.

### ε — Intrinsic Strength (base score τ)

The LLM's raw confidence in an argument before considering how the rest of the graph reacts to it.

- Set once during Γ+ε construction, never modified
- Domain-scoped: only the owning specialist scores its own arguments
- Main claim nodes carry τ = 0.5 (neutral default; mains are not ε-scored)
- Range: [0, 1]

### σ — Dialectical Strength

Computed by DFQuAD (Discontinuity-Free Quantitative Argumentation Debate) after the full graph is assembled. It reflects how attackers and supporters across all three domains collectively modify each node's intrinsic strength.

DFQuAD evaluates nodes in topological order (leaves first). For each node with base score τ:

```
1. Collect σ of all attackers  {a₁, a₂, ...}
   Collect σ of all supporters {s₁, s₂, ...}

2. Product aggregation (signed):
   agg = Π(1 − aᵢ) − Π(1 − sᵢ)
         ────────────────────────
         attack product   support product

3. Linear influence:
   if agg > 0  (net attack):   σ = τ − τ · agg
   if agg < 0  (net support):  σ = τ + (1 − τ) · |agg|

   equivalently:
   σ = τ + (1−τ)·max(0, agg) − τ·max(0, −agg)   clipped to [0,1]
```

Intuition:
- A strong, uncontested supporter pushes σ toward 1
- A strong, uncontested attacker pushes σ toward 0
- Balanced attack and support leave σ close to τ
- A weak attacker on a node with τ = 0.8 barely moves σ
- Multiple moderate attackers compound via the product, producing a sharper drop than a single strong one

```
Example:
  main_claim  τ = 0.5
    ├── supporter  τ = 0.73  →  σ = 0.73  (leaf, σ = τ)
    └── attacker   τ = 0.61  →  σ = 0.61  (leaf, σ = τ)

  agg = (1 − 0.61) − (1 − 0.73)
      = 0.39 − 0.27
      = 0.12   (slight net attack)

  σ(main) = 0.5 − 0.5 · 0.12 = 0.44
```

Peer reactions (τ = 0.5) have weaker absolute influence than high-ε domain arguments, but they can tip a balanced case when two agents agree against one.

---

## Stack

| Component | Role |
|-----------|------|
| [DSPy](https://github.com/stanfordnlp/dspy) | LLM module declarations and typed I/O signatures |
| [ARGORA](argora-public/) | QBAF graph builder (`RoundGraph`) and `compute_strengths_single_pass` |
| [NeuKRAG](neukrag/) | KG loading, BFS subgraph retrieval, entity extraction |
| [Streamlit](ui/app.py) | Chat UI, Plotly graph, MathJax, provenance hover cards |
| Ollama | Local LLM serving (configurable model via `environment.json`) |

---

## Modes Summary

| Mode | Rounds | ArgLLMs | Debate | Adaptive break |
|------|--------|---------|--------|----------------|
| Synthesis | 1 | Round 1 | None | No |
| Adversarial | 1–5 | Round 0 | Tit-for-tat (levels 0–3) | Yes |
| Choreographed | 5 (fixed) | Round 1 | Scripted arc | No |
| NeuKRAG | 1 | None | None | No |
| NeuKRAG-inter | 1 | None | None | No |

---

## References

- Liang et al. (2023). *Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate* arXiv:2305.14325 — MAD protocol, tit-for-tat levels
- Çelik et al. (2025). *Argumentative Large Language Models for Explainable and Contestable Claim Verification* AAAI 2025 — Γ/ε/Σ pipeline, MArgE
- Kampik, Çyras, Ruiz Alarcón (2024). *Gradual Semantics for Weighted Bipolar Argumentation Frameworks.* IJAR — DFQuAD, aggregation-influence framework
