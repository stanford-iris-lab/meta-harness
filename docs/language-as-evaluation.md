# Language as Evaluation — LLM-native selection for meta-harness evolution

*Working research framing. Status: draft / to-iterate.*

## 0. One-sentence thesis

Replace the **scalar + argmax** interface between *evaluation* and *evolution* with a
**language interface**: the evaluation product is a high-information-density
natural-language *report* per (harness, question), and *selection itself* is performed
by the LLM proposer reading those reports — not by a hard-coded ranking over aggregated
per-case scores.

## 1. Motivation

The evolve loop factorizes as `evolve = f(proposer, evaluation)`. The field pours effort
into the **proposer/mutation operator** and outsources **evaluation** to a scalar oracle.
Decompose what the loop actually needs from evaluation into three steps:

```
rich feedback (score / trace / logs …)
   │
   ├─ step 1  EVALUATION   which harnesses are good vs bad   (comparison, across candidates)
   ├─ step 2  ATTRIBUTION  why good / why bad → evolve insight (diagnosis, within a candidate)
   └─ step 3  EVOLVE       generate the next candidate from the insight
```

Two things we now hold as established:

- **(i) Scalar aggregation is lossy, and in our regime the scalar proxy barely tracks the
  objective.** Aggregating per-case results into one number "collapses multidimensional
  critiques into a single number and obscures the location of an error" (Text2Grad). On our
  retrosynthesis meta-harness, validation accuracy correlates with held-out test at only
  Spearman ≈ **+0.18** (top-1 mis-selected).
- **(ii) Language is a richer credit-assignment medium than scalar reward** — GEPA, TextGrad,
  Trace, Text2Grad, Feedback Descent all demonstrate this. **But every one of them uses
  language only at step 2 (mutation/attribution).** Selection (step 1) is still done on a
  raw scalar — even GEPA selects candidates by Pareto over a scalar metric.

**The gap:** step 1 is universally executed by the dumbest possible mechanism — "read the
number." Nobody makes *selection* language-native. A key realization: because the evaluation
product feeds an LLM proposer, **it does not need to be human-legible-at-a-glance; it needs
to be proposer-actionable.** That removes the reason we aggregate to a scalar in the first
place (humans wanted a sortable number).

## 2. Core idea — "Language as Evaluation"

- **Evaluation product = a per-(harness, question) natural-language REPORT, not a score.**
  Each report is a hybrid:
  - **(a) grounded multi-axis rubric comparison** — fixed, possibly hard-coded/verifiable
    axes (the leash that keeps the report tethered to reality);
  - **(b) free-form explanation** of why this harness did well/poorly on *this* item.
- **No hard-coded ranking of the archive.** The proposer/selector LLM reads the reports and
  performs **selection + breeding in one language-native step**.
- Net effect: the lossy `aggregate→argmax` bottleneck is replaced by an LLM that reasons over
  structured+free-form per-case evidence to decide what to keep and what to breed from.

This is precisely the GEPA move — "language is a richer learning medium than a scalar" —
**transported from step 2 to step 1.**

## 3. Research questions

- **RQ1 (does it select better).** At **equal evaluation budget**, does LLM selection over
  per-case language reports pick harnesses that generalize to **held-out test** better than
  scalar-argmax selection?
- **RQ2 (why, if it does).** Is any win from the **richness** of the per-case content
  (multi-axis + free-form) or from the **LLM's inferential aggregation** (reasoning across
  cases instead of averaging)? These are separable and must be ablated.
- **RQ3 (regime dependence).** Under what signal-to-noise regime does language-eval help?
  Our variance-capped result predicts: **no help when the true ranking is unrecoverable**
  (test CIs overlap). Where is the crossover?
- **RQ4 (gaming).** Folding selection into the LLM opens a Goodhart surface: does the proposer
  start producing harnesses that *read* well but *test* worse (BadScientist-style
  judge-gaming)? How do we detect and bound it?
- **RQ5 (aggregation / scale).** `N_harness × N_question` verbose reports do not fit in
  context. How do we aggregate language **without collapsing back to a scalar** and without
  blowing the context budget? (Linguistic aggregation is itself a sub-problem.)

## 4. Hypotheses

- **H1.** Language-eval selection **> scalar selection on held-out** *when a recoverable
  ranking exists* (moderate SNR), and **≈ scalar** when variance-capped.
- **H2.** Most of the gain is **inferential** (LLM reasoning across cases), not merely richer
  per-case content — i.e. feeding a per-axis *scalar vector* to argmax underperforms the same
  content read as language by an LLM.
- **H3.** Language-native selection is **more sample-efficient** (fewer cases to make the
  right keep/kill call), because it exploits within-case structure the scalar discards —
  the selection analogue of GEPA's ~35× rollout efficiency.
- **H4.** Without a grounding anchor (rubric axes / verifiable sub-signals), free-form
  selection is **hackable and drifts** — the free-form half needs the leash of the grounded
  half.

## 5. Design sketch

**Testbed first (this is the gating prerequisite).** Our retrosynthesis regime is the *worst*
place to test this: test itself can't rank the candidates, so no selector can. We need a task
where **test *can* rank but the cheap val scalar mis-ranks** — e.g. a higher-accuracy task, or
a controlled setup where we inject val↔test proxy gap on purpose (label noise / distribution
shift on val) so a ground-truth ranking provably exists.

**Arms (hold the proposer fixed; vary ONLY the evaluation→selection interface):**
- **A. scalar-argmax** selection on aggregated val accuracy — baseline.
- **B. multi-axis rubric vector** → simple/learned selector (rich but still numeric).
- **C. free-form language report** → LLM selector (language-native).
- **D. hybrid** (B + C) — grounded axes + free-form, LLM selector.

**Protocol:** budget-matched (identical # inner-loop evals per arm, so any win is
signal-quality not sample-count — the GEPA-style isolation). Measure held-out test of the
evolved frontier across iterations. **Gaming probe:** track report-predicted quality vs actual
test over generations; divergence = Goodhart onset.

## 6. Novelty / relationship to prior work

- **Extends GEPA/TextGrad** — "language > scalar" — from *mutation* (step 2) to *selection*
  (step 1). This is the specific unclaimed move.
- **Extends LLM-as-judge / Agent-as-a-Judge** from *scoring one output* to *selecting a
  policy/harness across a batch of cases inside an evolve loop*.
- **Contrasts Quality-Diversity** — QD uses behavioral descriptors for *diversity/coverage*;
  here language is used for *quality selection*, and it does not assume the axes predict the
  objective (which is exactly what our grounding experiment falsified when used as a selector).

## 7. Honest risks

1. **Variance doesn't vanish, it moves into the narrative.** A language report over noisy
   per-case outcomes can produce confident hallucinated stories over noise. Language ≠ immunity
   to variance; RQ3's testbed caveat still binds.
2. **Scale / context explosion** → forces linguistic aggregation, itself a research problem
   (RQ5).
3. **Circularity / gaming** when evaluator and proposer share a model family (RQ4).
4. **Verifiability** — free-form is untethered from gold at eval time (gold-free setting); the
   grounded rubric axes are the only leash.
