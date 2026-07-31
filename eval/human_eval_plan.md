# Human Evaluation Plan — Creativity Ranking Across NeuMAD Modes

## 1. Motivation

Combinatorial creativity scoring runs over the saved artifacts algorithmically: it is a fully objective, deterministic metric with no subjective judgment involved. Human evaluation is a deliberately separate, independent evaluation track: it exists to capture a subjective dimension of creativity that an algorithmic metric cannot, not to reproduce or validate the algorithmic score. This plan covers that human side: having neuromorphic computing experts rank outputs from the different NeuMAD modes for creativity, per research question.

## 2. Modes under evaluation

8 leaf conditions:

- NeuKRAG
  - NeuKRAG-base
  - NeuKRAG-inter
- Debate
  - Neuromorphic-mediated
    - Adversarial
    - Choreographed
  - Regular
    - Adversarial
    - Choreographed
- Synthesis
  - Rotation
  - Synthesis

## 3. Question set

16 questions total, 8 from each group: A and B:

- **Group A** — open-ended questions about the field in general. Agents don't necessarily even converge on a shared goal.
- **Group B** — design questions with a common goal fixed across agents. Agents share the goal but won't necessarily agree on the approach.

These are qualitatively different tasks (open-ended exploration vs. goal-directed design), so creativity may manifest differently under each. Group (A vs. B) is treated as a blocking factor in analysis rather than pooling all 16 questions together (see §9).

## 4. Two rating tracks

The evaluation has two independent tracks, scoring two different dimensions.

### Track 1 — Content creativity (blinded, `response.md`)

- Evaluation unit: the `response.md` artifact for each (question, mode) pair: final hypothesis text only, no debate/rotation history, no argumentation graph, no expanders.
- Per question, collect the 8 mode variants into one folder and rename them to opaque labels (`A.md` ... `H.md`). The label→mode mapping is randomized independently *per question* (not fixed across questions) and kept in a private key file, not shown to raters. This prevents raters from pattern-matching mode identity across questions after the first couple of items. Randomize the on-page order of A–H per rater per question, so list position carries no signal.
- Each rater covers 8 of the 16 questions (4 from group A + 4 from group B), not
  all 16 — see the balanced allocation in §6.
- Rank novelty and utility. Each rank is given a normalized score. First place gets one, last place gets zero, and the middle places are interpolated. The final creativity score is multiplied between novelty and utility.

### Track 2 — Explainability (unblinded, `response.html`)

- Evaluation unit: the full `response.html` artifact (debate tree, rotation trace,
  argumentation graph, hover cross-references, dropdowns included) — unblinded on
  purpose, since feature presence is mode-dependent (e.g. NeuKRAG-base has no
  debate trace) and that absence should be scored honestly, not hidden.
- Rating dimension: how well the interactive features let the rater verify *how*
  the answer was reached, as opposed to content creativity (that's Track 1).
- Structured rubric, not ranking (table below): explainability decomposes into
  gradable sub-features rather than a holistic aesthetic judgment.
- Same 8-question allocation as Track 1 (§6) — 4 from group A + 4 from group B,
  per rater — rather than a separately-chosen subset.

## 5. Rubric / calibration

**Note**: *This rubric is a work in progress. Needs a neuromorphic expert to review.*

<!-- - One shared calibration question, rated by all raters on both tracks, using the
  rubrics below; include one high-novelty/low-utility example and one
  low-novelty/high-utility example so raters see the two dimensions can diverge
  before scoring for real.
- Group discussion of discrepancies after calibration — the highest-leverage step
  for usable agreement numbers later. -->

### Track 1 rubric — Novelty & Utility (ranking)

| Score | Novelty anchor | Utility anchor | Score |
|---|---|---|---|
| 1 | First place | First place | 1.000 |
| 2 | | | 0.857 |
| 3 | | | 0.714 |
| 4 | | | 0.571 |
| 5 | | | 0.429 |
| 6 | | | 0.286 |
| 7 | | | 0.143 |
| 8 | Last place | Last place | 0.000 |

### Track 2 rubric — Explainability

Sub-feature checklist (per artifact) — quality, not just presence: 0 = absent,
1 = present but unhelpful, 5 = present and highly clear/useful:

| Sub-feature | Quality (0–5) | Notes |
|---|---|---|
| Traceable reasoning path (debate/rotation history) | | |
| Cross-referenced hover labels | | |
| Argumentation graph legibility | | |
Overall score:

| Score | Anchor |
|---|---|
| 1 | Can't tell how the answer was reached |
| 2 | |
| 3 | |
| 4 | |
| 5 | Fully traceable, clear how the conclusion was reached |

## 6. Raters

- 6 neuromorphic computing domain experts.
  - 3 is the practical floor, not a hard statistical minimum: at n=2, Kendall's W
    reduces to a rescaled Spearman correlation between the two rankings (Legendre,
    2005). No real agreement signal, and no way to tell which rater is the
    outlier.
    - Plain version: with 2 raters, "group agreement" is just whether those two
      people agree — the same information as a plain pairwise correlation, not a
      new measurement. And if they disagree, there's no way to tell who's the
      outlier vs. who's right. With 3+, a 2-1 split actually points to the outlier.
  - 6 is comfortably above that floor, an even number (clean for pairwise-agreement
    stats), and realistic to recruit in a specialized field.
- **Question allocation**: rather than every rater covering all 16 questions,
  split the 6 raters into two groups of 3 (R1, R2) and the 16 questions into two
  sets of 8 (4 from group A + 4 from group B each). R1 rates one set, R2 rates the
  other. Each rater does 4 questions per group (8 total) instead of 16, and every
  question is still seen by exactly 3 raters — `r = 6×8/16 = 3`, matching the
  practical floor above, so per-question agreement stays checkable even though no
  single rater sees the full set. Same 8-question split is used for both tracks
  for a given rater (Track 1 then Track 2, in that order — see below).
- Same 6 raters do both tracks (within-subject). Track 1 (blind) is fully
  completed and submitted/locked for a given rater *before* Track 2 (unblinded)
  opens for them, so the blind creativity rankings can't be contaminated by having
  already seen mode-revealing HTML.
  - Alternative: independent rater groups per track (e.g. 6 MD-only + 6 HTML-only)
    is methodologically cleaner (zero spillover risk) but doubles the recruiting
    burden. Flagged as a resourcing tradeoff, not decided here.

**Note**: Alternatively, we could get 12 raters addressing 4 questions each, giving us six data points per question.

## 7. Session logistics

- Track 1: 8 questions × 8 items × 2 scores (novelty, utility) = 128 atomic
  judgments per rater (per the §6 allocation, not all 16 questions). Reading the 8
  outputs remains the time bottleneck, not the scoring mechanics, ~20–25 minutes
  per question, ≈2.75–3.3 hours total. Split across 2 sessions of 4 questions
  each, question order randomized per rater independent of session split.
- Track 2: same 8 questions per rater, lighter rubric-based task per artifact;
  likely 1 session.
- Track 1 must fully close for a rater before their Track 2 sessions begin (§6).

**Note:** This can be executed through a html-based survey interface. I (Vikram) can keep the artifact results and assemble a survey interface.

<!-- ## 8. Open gaps

- Whether to run Track 1 and Track 2 with the same 6 raters (within-subject) or
  two independent groups of 6 (between-subject) — resourcing decision, not yet
  made (§6). -->

## 8. Analysis plan

1. **Inter-rater agreement**: Track 1's rater judgment is a ranking (novelty and
   utility each ranked 1–8), with the normalized/composite score a downstream
   transform of that ranking — so agreement should be computed on the raw ranks,
   not the derived scores. Primary: Krippendorff's α with ordinal weighting on the
   novelty and utility ranks separately; Kendall's W as a cross-check. Track 2
   explainability scores are genuine independent ratings (not rank-derived), so
   those use ICC as before. Computed separately for group A and group B. Target
   α/W/ICC ≥ 0.67 (acceptable), ≥ 0.8 (good).
2. **Human/computational comparison**: Spearman correlation between mean human
   composite score (Track 1) and the combinatorial creativity score, per question
   — this compares the two independent dimensions (subjective vs. algorithmic), it
   is not treated as validating either one against the other. If the
   computational metric itself decomposes into novelty and utility components,
   also correlate those against the matching human sub-scores separately — more
   informative than comparing composites alone, since it can reveal e.g. the
   algorithm tracking perceived novelty well while diverging on utility.
3. **Per-mode Track 1 vs. Track 2 view**: for each of the 8 modes, report mean
   creativity composite (Track 1) alongside mean explainability score (Track 2),
   side by side — descriptive, not a formal test. Shows whether explainability and
   creativity track together across modes or trade off (e.g. a mode that's highly
   traceable but not particularly creative, or vice versa).

<!-- ## 10. Status

Draft — pending revision. -->