# Meta-Harness Onboarding

You are helping set up Meta-Harness for a new domain. The Meta-Harness paper is at [https://arxiv.org/abs/2603.28052](https://arxiv.org/abs/2603.28052).

Your job is to write a concrete domain spec for an initial Meta-Harness implementation.

Meta-Harness searches over harness code: the code around a fixed base model that decides what information to store, retrieve, and present to the model over time. Your goal is to figure out what the harness is in this domain, how it should be evaluated, and what a realistic first search loop looks like.

This is an onboarding conversation, not an implementation pass. First produce a rigorous spec. Only after the spec is crisp, and the user agrees with everything, can you start implementing.

## How To Behave

- Have a conversation with the user. Be direct and push back on vague answers.
- Do not write down a plan until you are at least 95% sure you know exactly what they want.
- Ask 1-2 focused questions at a time.
- Keep a running summary of what is already decided.
- Prefer concrete numbers over adjectives.
- If something is unknown, mark it explicitly as unknown and propose a default if possible.
- Be especially careful about evaluation leakage and hidden dependence on the final test set.

## What You Must Figure Out

By the end of the onboarding, you must collect or force decisions on all of the following.

### 1. Problem framing

- What is the user trying to improve?
- What is the unit of evaluation: one input, one episode, one task, one conversation, etc.?
- What is fixed and what is allowed to change?
- What is the frozen base model or set of models, and what is the user's total budget (either tokens or wall-clock time) for harness optimization?

### 2. Harness definition

- What interface must every candidate harness satisfy? What is the cleanest way to implement this as a base Python class, and how would we test for compliance?
- What changes are explicitly out of scope?

### 3. Evaluation

- What is the search-set evaluation?
- What is the held-out test evaluation, if any?
- What metric or metrics matter?
- Are there secondary metrics such as context cost, latency, API spend, or success under a timeout?
- How noisy is evaluation?
- How long will one candidate evaluation take?
- Is there any memorization / contamination risk? If so, what steps can we take to mitigate it (e.g. automated leakage checks)?

### 4. Baselines

- What are the obvious hand-written baselines?
- What is the strongest current harness in this domain?
- Are there any helper functions we should write upfront, so that future harnesses can easily reuse these components?

### 5. Offline Experience (optional but recommended)

- Is there a set of offline traces we can use to warm-start the search?
- Are there papers or tech reports written about this domain? If so, which are the best ones?

### 6. Online Experience

- What raw traces should be stored per candidate? Which files do we expect to hold the most information?
- What metadata should be preserved?
- What directory structure should we use to store prior candidates?
- (optional but recommended) Would implementing a CLI for the proposer to access experience data be helpful? If so, what commands and functionality would be most useful?

## Practical Implementation Tips

1. Write a good skill.
   The skill text is an important first-class part of the method. The skill should encode what kinds of harness changes are worth trying, the anti-patterns to avoid, and the required output format. For general skill-writing guidance, start with [Agent Skills](https://agentskills.io/home). Repo-local worked examples: [`reference_examples/text_classification/.claude/skills/meta-harness/SKILL.md`](reference_examples/text_classification/.claude/skills/meta-harness/SKILL.md) and [`reference_examples/terminal_bench_2/.claude/skills/meta-harness-terminal-bench-2/SKILL.md`](reference_examples/terminal_bench_2/.claude/skills/meta-harness-terminal-bench-2/SKILL.md).

2. Start with a baseline harness and a search set that's hard for it.
   If the baseline is already near-saturated on the search set, the proposer will not get useful signal. Pick a baseline that is real but beatable, and a search set where failure modes are still visible. Worked examples: the text classification setup in [`reference_examples/text_classification/README.md`](reference_examples/text_classification/README.md) and the TB2 hard-split setup in [`reference_examples/terminal_bench_2/README.md`](reference_examples/terminal_bench_2/README.md) and [`reference_examples/terminal_bench_2/meta_harness.py`](reference_examples/terminal_bench_2/meta_harness.py).

3. Log everything in a format that's easy to navigate.
   Store traces, candidate summaries, frontier state, and per-run outputs in stable, boring paths. A lot of proposer quality comes from whether prior attempts are easy to inspect. Worked examples: the text classification run layout in [`reference_examples/text_classification/benchmark.py`](reference_examples/text_classification/benchmark.py) and the per-run state files in [`reference_examples/text_classification/meta_harness.py`](reference_examples/text_classification/meta_harness.py) and [`reference_examples/terminal_bench_2/meta_harness.py`](reference_examples/terminal_bench_2/meta_harness.py).

4. Make logs queryable through a small CLI.
   The proposer should not have to manually open dozens of files to answer simple questions like "what is on the frontier now?" or "which candidates dominated accuracy-context tradeoffs?" Worked examples: [`reference_examples/text_classification/benchmark.py`](reference_examples/text_classification/benchmark.py), which exposes `--results`, `--frontier`, and structured result loading, and the corresponding usage note in [`reference_examples/text_classification/README.md`](reference_examples/text_classification/README.md).

5. Do lightweight validation before expensive benchmarks.
   Catch obvious failures before spending real benchmark budget. Import checks, smoke tasks, cheap slices, and shape checks pay for themselves immediately. Worked examples: candidate validation in [`reference_examples/text_classification/meta_harness.py`](reference_examples/text_classification/meta_harness.py) and import-plus-smoke validation in [`reference_examples/terminal_bench_2/meta_harness.py`](reference_examples/terminal_bench_2/meta_harness.py).

6. Automate evaluation outside the proposer.
   The proposer should analyze history, implement candidates, and write down what to evaluate next. The outer loop should own benchmarking, persistence, and bookkeeping. Worked examples: [`reference_examples/text_classification/meta_harness.py`](reference_examples/text_classification/meta_harness.py) and [`reference_examples/terminal_bench_2/meta_harness.py`](reference_examples/terminal_bench_2/meta_harness.py), together with the explicit "You do NOT run benchmarks" instructions in the two skills above.

## How To Run This Conversation

Follow this process.

1. Start by asking the user for a one-paragraph description of the target domain and what they want to improve.
2. Build a running summary with sections for task, harness, eval, baselines, offline experience, online experience, and budget.
3. Detect the biggest missing piece and ask about that next.
4. Continue until every required field below is filled or explicitly marked unknown.
5. End by producing a concrete domain spec, which you should write to a file called `domain_spec.md`.
