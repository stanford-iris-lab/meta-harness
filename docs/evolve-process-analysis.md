# Evochain Evolution Process Analysis — USPTO Online Mode

Covers three runs:
- `online_uspto_043436` — 50 train / 30 val / 100 test (full val)
- `mh-runB/online_uspto_081258` — 10 train / 4 val / 100 test (small val)
- `mh-runC/online_uspto_50_4` — 50 train / 4 val / 100 test (small val)

## Run Info (full-val run)

- **Run**: `online_uspto_043436`
- **Task**: USPTO retrosynthesis (product SMILES → reactant SMILES)
- **Mode**: Online (predict → learn_from_batch with real `was_correct`)
- **Evaluator**: Exact lowercased set-match on `.`-split reactants — **no RDKit canonicalization**
- **Data**: 50 train / 30 val / 100 test

---

## Proposer Input Signals

The Claude Code proposer (SKILL.md `meta-harness`) receives exactly:

| File | What it contains | How proposer uses it |
|---|---|---|
| `frontier_val.json` | Best system per task, val accuracy, ctx len | Identifies current frontier baseline |
| `evolution_summary.jsonl` | All past candidates: hypothesis, avg_val, outcome, components | Identifies which mechanism axes are exhausted |
| `logs/<task>/<memory>/<model>/log.jsonl` | Every training `step` (input, pred, tgt, ok, prompt, response) + every `val_step` | **Deeptraces failure patterns at the individual prediction level** |
| `logs/<task>/<memory>/<model>/memory.json` | Learned memory pool (examples + state) | Inspects what the system actually stored |
| `logs/<task>/<memory>/<model>/val.json` | Aggregated val accuracy | Quick number lookup |
| `config.yaml` | Dataset config, mode, splits | Confirms online vs offline, data size |

The proposer **does NOT** see `memory.json` during initial analysis — it reads it specifically to inspect what the best system stored (iteration 2 only).

---

## Iteration 1 (2026-06-22, ~646s, $2.96)

### Context at Start

- Only baselines exist: `no_memory` (10%), `fewshot_all` (6.7% — **regressed**, random demo selection hurts)
- No prior evolution history in this run
- Mode is **online** — `learn_from_batch` receives real `was_correct` flags (proposer corrects its offline assumption from prior memory)

### Proposer Tool Sequence

1. **Read** `config.yaml` → confirms online mode, USPTO 50/30/100 splits
2. **Bash** `ls agents/` → sees baseline files only
3. **Read** `frontier_val.json` + `evolution_summary.jsonl` → confirms only baselines
4. **Read** `fewshot_all.py`, `no_memory.py`, `memory_system.py` → understands interface
5. **Bash** (explore logs tree) → finds prior run `iter1_fixed_012127` with USPTO scores
6. **Bash** (inspect USPTO task) → dumps first `step` entry from `log.jsonl`: full prompt + response + pred vs tgt
7. **Bash** (read evaluators + loader) → finds evaluator is exact lowercased set-match
8. **Read** `data/loaders.py` → finds USPTO loader structure
9. **Bash** (prototype script 1: reaction-type buckets, recurring reagents, LCS ratio):
   - Reaction types are **perfectly balanced** (10 types × 5 train / 3 val)
   - **Boc2O** appears in **all 5** Protections train targets
   - **NBS** appears in 4/5 FGA train targets
   - 40% of reactions have main reactant copying >60% of product (LCS ratio)
   - Same-type 5 demos ≈ **496 chars** vs fewshot_all's **29,860 chars** (60× smaller)
10. **Bash** (prototype script 2: validate parser on full templated input) → parser works 50/50 on full prompt
11. **Write** `type_grouped_fewshot.py`
12. **Write** `reagent_vocab.py`
13. **Write** `error_diff_lessons.py`
14. **Bash** (validate imports + agent discovery)
15. **Bash** (smoke test with mock LLM)
16. **Write** `pending_eval.json`
17. **Bash** (final verification)

### Iteration 1 Hypotheses

| Candidate | Hypothesis | Mechanism Axis | Result |
|---|---|---|---|
| `type_grouped_fewshot` | Demos parsed by reaction-type bucket + char n-gram similarity ranking + strip boilerplate → compact demos (~500 chars) beat random 30k-char demos | **Selection algorithm** | 26.7% ✅ best |
| `reagent_vocab` | Extract recurring "added-component" fragments per category → inject as exact-string cheat-sheet + copy-backbone framing | **Memory content** | 23.3% |
| `error_diff_lessons` | Online error signal (pred vs gold diff) → LLM distills per-category correction rules (Reflexion-style) | **Learning trigger + LLM in learning** | 20.0% |

### Key Data-Analyzed Decisions

```
found: fewshot_all regressed to 6.7% (random demo selection actively hurts)
found: evaluator = exact lowercased set-match (string-convention problem, not just chemistry)
found: Boc2O recurs in ALL 5 Protections train targets
found: same-type demos 60× smaller than random demos, far more relevant
→ decision: category bucketing is the primary lever (type is stated in input)
```

---

## Iteration 2 (2026-06-22, ~776s, ~$3.55)

### Context at Start

- Frontier: `type_grouped_fewshot` 26.7% (8/30 correct), `reagent_vocab` 23.3%, `error_diff_lessons` 20.0%
- Baselines: `no_memory` 10%, `fewshot_all` 6.7%
- Reports dir: **does not exist** yet (iteration 1 proposer didn't create it — SKILL.md says to write post-eval reports but it skipped)

### Proposer Tool Sequence

1. **Read** `evolution_summary.jsonl`, `frontier_val.json`, `memory_system.py`
2. **Bash** (ls agents + reports dir) → `NO_REPORTS_DIR`, iter1 candidates visible
3. **Read** `evolution_summary.jsonl` + `frontier_val.json` → pulls exact numbers
4. **Read** `fewshot_all.py`, `memory_system.py`, `type_grouped_fewshot.py` (the best system)
5. **Bash** (explore logs tree) → confirms iter1 candidates have log.jsonl + memory.json
6. **Bash** (inspect log.jsonl structure) → finds: `meta`, `step`(50), `learn_batch`(50), `val_step`(30), `done`
7. **Read** `memory_system.py`, `type_grouped_fewshot.py` source
8. **Bash** (dump ALL 30 val predictions: pred vs gold for type_grouped_fewshot) → 8/30 correct
9. **Read** `val.json` (aggregated results)
10. **Bash** (inspect memory.json structure) → 50 train examples, `{cat, body, target}`
11. **Bash** (prototype: reagent vocab + analogy + backbone coverage):
    - **Mechanism A** (exact-reagent reranker): scores gold > pred in only **5/22** failures → too weak
    - **Mechanism B** (nearest-neighbor analogy): only **4/30** val cases share exact reagent with nearest same-category neighbor
    - Coverage: only **4/14** val golds have ≥1 added reagent seen verbatim in same-category train vocab
12. **Bash** (prototype: near-miss correction transfer train→val):
    - **13 near-miss corrections** found in val failures
    - **0 exact (wrong→gold) pairs** recurring train→val → error correction table does NOT transfer
    - **7/22** val failures are single-fragment swaps (potentially recoverable)
    - Wrong preds have **HIGHER** backbone-LCS to product (median 0.84) than gold (median 0.67) → model **under-disconnects** (keeps product nearly intact), gold splits into genuine precursors
13. **Bash** (prototype: product-reconstruction scorer discriminability):
    - Scorer(gold) > scorer(pred) in only **5/22** failures → deterministic reranker is a **bad discriminator**
    - Gold backbone-LCS median: 0.67; wrong-pred backbone-LCS median: **0.84**
    - Conclusion: backbone-conservation reranker would **pick the wrong answer**
14. **Write** `draft_verify_revise.py`
15. **Write** `consensus_fragment_voting.py`
16. **Write** `category_recipe.py`
17. **Bash** (validate imports, one subclass per file)
18. **Bash** (smoke test — first attempt failed due to PYTHONPATH)
19. **Bash** (smoke test retry with PYTHONPATH set):
    - `draft_verify_revise`: cold=1 call, warm=2 calls/predict, ctx=699
    - `consensus_fragment_voting`: cold=1 call, warm=3 calls/predict, ctx=566
    - `category_recipe`: cold=1 call, **learn=1 call** (recipe synthesis), warm=1 call/predict, ctx=366
20. **Write** `reports/iteration_1.md`
21. **Write** `pending_eval.json`
22. **Edit** `memory.md` with dead-lever findings

### Iteration 2 Hypotheses (Data-Driven Rejection → Pivot)

| Candidate | Hypothesis | Mechanism Axis | Result |
|---|---|---|---|
| `draft_verify_revise` | Two-pass LLM: draft → forward-reconstruction checklist self-verify → revise. Targets under-disconnection + exact-token errors | **Prompt architecture** | 20.0% |
| `consensus_fragment_voting` | Perturb demo context (jackknife) → multiple drafts → answer-set majority vote. Since temp=0 blocks resampling, manufactured diversity via context perturbation | **Ensemble + selection** | 26.7% |
| `category_recipe` ✅ | LLM abstracts **correct** examples per category → cached procedural recipe + verbatim recurring components. Different from error-distillation (which learned from failures) | **Learning strategy** | **30.0%** best |

### Key Data-Analyzed Decisions

```
Three dead ends ruled out by prototype data:
  1. Exact-reagent vocab: too sparse (4/14 val golds have verbatim reagent in train cat vocab)
  2. Error near-miss correction table: 0 transfer train→val (errors are molecule-specific, not systematic)
  3. Deterministic backbone-conservation reranker: ANTI-correlated with correctness
     (wrong preds keep product intact: backbone-LCS 0.84 vs gold 0.67)

Conclusion: bottleneck is model reasoning, not retrieval/vocab.
Three new mechanism families (all built on proven same-category retrieval):
  → draft_verify_revise: LLM self-check (not deterministic)
  → consensus_fragment_voting: ensemble diversity (not deterministic)
  → category_recipe: learn from CORRECT data (not from errors)
```

---

## Summary: How the Proposer Thinks

```
STEP 1 — Identify frontier + failure patterns
  frontier_val.json tells you what's best
  log.jsonl val_step traces tell you HOW it fails (not just accuracy)

STEP 2 — Hypothesis generation via data
  Proposer writes PROTOTYPE SCRIPTS against real data to TEST assumptions
  Before: "reagent vocab should work" → prototype says: only 4/14 coverage, too sparse
  Before: "error correction table should transfer" → prototype says: 0 train→val transfer
  Before: "backbone conservation reranker" → prototype says: ANTI-correlated (wrong preds have MORE backbone overlap)

STEP 3 — Pivot or refine
  Dead ends → ruled out definitively with data
  Surviving signals → 3 distinct mechanism families
  Each candidate MUST change predict() or learn_from_batch() logic (not just constants)

STEP 4 — Validate before writing
  Import check (one subclass per file)
  Full lifecycle smoke test (cold → learn → warm predict → state roundtrip)
  Context length sanity check (vs fewshot_all's 29k chars)
```

---

## Run B & C: Small Val Set — Credit Exhaustion / Auth Crash

Two runs with deliberately reduced validation size (`num_val=4`, each example = 25% accuracy):

| Run | Config | Train | Val | Test |
|---|---|---|---|---|
| `mh-runB/online_uspto_081258` | 10 + 4 + 100 | 10 | 4 | 100 |
| `mh-runC/online_uspto_50_4` | 50 + 4 + 100 | 50 | 4 | 100 |

### The `avg_val: 0` Problem

All candidates in both runs show `avg_val: 0`, `outcome: "failed"`, `delta: -50.0`.

**But this is NOT a mechanism failure.** The proposer investigations revealed the real cause:

- `mh-runB`: `no_memory` ran at 08:13 → succeeded (50% = 2/4 correct). All 3 iter1 candidates ran at 08:22–08:23 → **OpenRouter 401 auth crash** ("No cookie auth credentials found"). Never actually evaluated.
- `mh-runC`: Same pattern — baselines ran, candidates crashed on the same auth error.

The proposer correctly identified this from log inspection:
```
Each failed candidate has empty val.json + tiny log.jsonl (~215 bytes)
→ crashed at startup, before any prediction
→ .launcher/*.log tracebacks show: AuthenticationError 401
```

### How the Small Val Set Made This Worse

With only 4 val examples:
- Each example = 25 percentage points of accuracy
- `avg_val: 0` could mean: crashed / 0/4 / 1/4 / 2/4 — all indistinguishable from outside
- The proposer **cannot trust val scores as a signal** when all candidates show 0

This mirrors your question: when val is small, **does the proposer use val scores or actual traces?**

### Answer: The Proposer Uses Train Traces + Prototypes, Not Val Scores

With noisy/unreliable val scores, the proposer fell back to:

**1. Train prediction traces** (`log.jsonl` step entries):
```
From no_memory train run (14 examples = 10 train + 4 val):
- 2/4 val correct (FGA with NBS reagent, simple single-reactant)
- 2/4 val wrong (Acylation needs TMS-isocyanate — no same-type train demo → unlearnable;
                  FGA wrong reagent choice)
- Error taxonomy: convention errors (N vs [NH3+]), missing reagents (NBS/Boc2O/TMS),
                  mis-disconnections
```

**2. Mandatory prototype scripts against real data** (the only clean signal):
```
Prototype A: does mining reagent fragments per category surface the exact val reagent?
Result: FGA val reagent (NBS "O=C1CCC(=O)N1Br") appears 4× in same-category train
→ mechanism A validated

Prototype B: does conservation verifier pick correct over wrong on val failures?
Result: FGA 0.950 vs 0.773; Acylation 0.778 vs 0.474
→ mechanism B validated

Prototype C: does same-category exemplar carry the right reagent?
Result: FGA ✓, Acylation → "no reagent" (correct signal for single-reactant case)
→ mechanism C validated
```

**3. Rejected hypothesis by prototype, not by val score**:
```
Deterministic reranker / reagent vocab → ruled out by prototype on TRAIN DATA
(not by val score, because val has only 4 examples and is too noisy)
```

### Key Contrast with the Full-Val Run (online_uspto_043436)

| Signal Source | Full Val (30 examples) | Small Val (4 examples) |
|---|---|---|
| Val score reliability | High (each = 3.3%) | Near-zero (each = 25%) |
| Val trajectory (log.jsonl val_step) | 30 examples, rich | 4 examples, noisy |
| Proposer strategy | Val traces + train traces + prototypes | **Train traces + prototypes almost exclusively** |
| Hypothesis falsification | Prototype + val outcome | Prototype only (val all 0) |
| "avg_val: 0" meaning | Likely real failure | Could be crash, noise, or real failure — must investigate |

### What Happened: Credit Exhaustion

Even after the proposer correctly identified the crash pattern and re-proposed candidates in iter2/iter3, the runs hit **credit exhaustion** before any candidate could be properly evaluated:

```
iter1: candidates crash on auth → avg_val: 0 (crash, not failure)
iter2: re-proposed, but run out of credits before re-evaluation
iter3: same
```

The small val set amplified this: you can't distinguish a good candidate from a bad one when every candidate shows `avg_val: 0` for different reasons (crash vs. real failure vs. noise).

### Takeaway for Experiment Design

With `num_val=4`:
- **Val is essentially useless as a score** — too noisy to distinguish mechanisms
- **The proposer falls back to train traces + prototype scripting** — which works, but loses the "evolution" feedback loop
- **Credit/auth failures become catastrophic** — because you can't use val scores to confirm the candidate even ran

The full-val run (`online_uspto_043436`) avoided this: when candidates failed, the val scores (26.7%, 23.3%, 20.0%) were meaningful and the proposer could trace **which specific mechanisms** failed and why.

---

## Candidate Mechanism Map (Iterations 1-2)

```
BASELINES:
  no_memory        → 10%  (no retrieval)
  fewshot_all      → 6.7% (random demos, 30k chars, regressed)

ITERATION 1:
  type_grouped_fewshot    → 26.7%  [selection: category bucket + n-gram sim]
  reagent_vocab            → 23.3%  [memory content: fragment vocab cheat-sheet]
  error_diff_lessons       → 20.0%  [learning: error-triggered LLM rule distillation]

ITERATION 2 (3 dead ends found: reagent vocab too sparse, error transfer 0, backbone reranker anti-correlated):
  draft_verify_revise      → 20.0%  [prompt: two-pass self-verify, targets under-disconnection]
  consensus_fragment_voting→ 26.7%  [ensemble: demo-context perturbation + answer voting]
  category_recipe          → 30.0%  [learning: LLM abstracts CORRECT examples into per-category procedure]  ← BEST
```

---

## Hypothesis Quality Notes

- **`type_grouped_fewshot`** (26.7%): Well-motivated. Category is explicitly stated in input, same-type demos are 60× more compact, chemistry is correct but string conventions differ. The mechanism (category bucket + n-gram similarity) is genuinely different from random demo selection.

- **`error_diff_lessons`** (20.0%): Mechanistically plausible for online mode, but the error signal is too sparse and molecule-specific. Only 3/50 train steps correct, and near-miss corrections don't transfer train→val.

- **`reagent_vocab`** (23.3%): The insight about recurring reagents (Boc2O, NBS) is real, but coverage is too low (4/14 val cases) to be the primary lever. The pool of 50 unique products fundamentally caps memorization.

- **`category_recipe`** (30.0%, best): Key insight shift — instead of learning from errors (sparse, molecule-specific) or raw demos (convention-implicit), learn from **correct** examples by having LLM abstract the **procedure**. This makes the convention explicit rather than hoping the model re-infers it correctly each time.

- **`draft_verify_revise`** (20.0%): The diagnosis (model under-disconnects, keeps product intact) is correct from the backbone-LCS data. But two-pass LLM without a strong verifier signal doesn't reliably correct — the second pass has no more ground truth than the first.

- **`consensus_fragment_voting`** (26.7%): Self-consistency via demo-context perturbation is creative given temp=0 constraint, but the diversity from context perturbation alone may not be enough to overcome the fundamental task difficulty.
