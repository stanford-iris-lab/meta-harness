# SpreadsheetBench — Meta-Harness reference experiment

Agent-scaffold search on [SpreadsheetBench](https://github.com/RUCKBReasoning/SpreadsheetBench):
an agent reads an `.xlsx`, then writes a Python **solution program** over multiple rounds of
code execution; the program is re-run on ~3 hidden test-case inputs and scored OJ-style (all
must pass). The outer loop evolves the *scaffold* (prompt, exploration, execution-feedback,
self-checking) on the cheap `gpt-oss-120b` solver, so iteration is affordable.

Same Meta-Harness pattern as `terminal_bench_2` (scaffold search) but self-contained: no
Harbor, a lightweight base agent, subprocess code execution, and openpyxl value comparison.

## Architecture

- **Outer loop** `meta_harness.py`: propose (Claude Code subprocess) → validate (import +
  1-task smoke) → benchmark (`benchmark.py`) → update frontier. Per-run isolation under
  `logs/<run>/` with `pending_eval.json`, `frontier_val.json`, `evolution_summary.jsonl`.
- **Inner loop** `benchmark.py` + `inner_loop.py`: `benchmark.py` auto-discovers `agents/*.py`
  and runs `inner_loop.py` per (agent, task) with bounded concurrency; `inner_loop.py` runs one
  scaffold on one instruction and OJ-scores it.
- **Candidate = a scaffold**: `agents/<name>.py` with `class AgentHarness(SpreadsheetAgent)`
  (`agent.py`). Baselines: `baseline_react` (evolution parent), `baseline_single` (control).
- **Executor** `executor.py`: runs model code in a fresh workdir where the state is
  `input.xlsx` on disk; solutions write `output.xlsx`. **Evaluation** `evaluation.py` +
  `sheet_utils.py`: openpyxl `data_only` value comparison at `answer_position`.

## Setup

```bash
cd reference_examples/spreadsheet_bench
uv sync
export QNAIGC_API_KEY=sk-...        # .env is NOT auto-loaded; export it (see .env.example)
bash data/fetch_data.sh             # download + extract sample_data_200 into data/
```

This loop is API-bound (LLM calls) with light local code execution (openpyxl on small
spreadsheets) — run it directly where the solver endpoint is reachable, like the other
meta-harness experiments. **Reachability caveat:** Vulcan compute nodes reach the internet
only through a proxy that blocks `api.qnaigc.com` (it allows OpenRouter); the login node
reaches QNAIGC directly. So run against QNAIGC from where it's reachable, or switch the
solver to OpenRouter (see below). See `SETUP.md`.

## Running it — step by step

Everything is a `uv` project, so `uv run <cmd>` runs inside the project venv without
activating anything. If you prefer an activated shell instead, after `uv sync` do
`source .venv/bin/activate` and drop the `uv run` prefix from every command below.

### 0. Install + data (once)

```bash
cd reference_examples/spreadsheet_bench
uv sync                                   # creates .venv with openpyxl/pandas/litellm
export QNAIGC_API_KEY=sk-...              # solver key (see Model & keys); .env is NOT auto-loaded

# Fetch any of the three official datasets into data/ (all are gitignored):
bash data/fetch_data.sh                                   # sample_data_200  (200 tasks)
bash data/fetch_data.sh spreadsheetbench_912_v0.1         # full benchmark   (912 tasks)
bash data/fetch_data.sh spreadsheetbench_verified_400     # verified subset  (~394 tasks)
```

### 1. Sanity checks (no API)

```bash
uv run python executor.py                                       # subprocess exec self-test
uv run python -m data.loader --source sample_data_200 --list | head
uv run python -m data.loader --source spreadsheetbench_912_v0.1 --dev-size 30 --seed 42
```

### 2. Run one instruction with one scaffold

```bash
# task ids come from the --list command above
uv run python inner_loop.py --agent agents/baseline_react.py --task-id 31184
# prints: [PASS|fail] <id>  n_pass/n_cases  turns=..  tokens=..  <s>
# add --out result.json --log traj.jsonl to inspect the trajectory + per-case detail
```

### 3. Benchmark a scaffold (or all) over a dev subset

```bash
# dev-subset size: --dev-size N, or env MH_N_TASKS=N, or config dataset.dev_size (0 = all)
MH_N_TASKS=5 uv run bash scripts/run_eval.sh agents.baseline_single         # quick
uv run python benchmark.py --all --run-dir logs/adhoc --dev-size 30         # both baselines
uv run python benchmark.py --results  --run-dir logs/adhoc                  # print the table
cat logs/adhoc/frontier_val.json                                           # per-task best + Pareto
```

Per-task outputs land in `logs/<run>/<agent>/task_<id>.json` (+ `.jsonl` trajectory);
`summary.json` per agent; `frontier_val.json` for the run.

### 4. Run the evolution loop (proposer → benchmark → frontier)

```bash
# one iteration, fresh state, isolated under logs/my-run/
uv run python meta_harness.py --iterations 1 --fresh --run-name my-run
# a real search: more iterations, modest concurrency (solver throughput is the bottleneck)
uv run python meta_harness.py --iterations 10 --run-name my-run --concurrent 8
# optional final full-source eval of the winner:
uv run python meta_harness.py --iterations 10 --run-name my-run --full-eval
```

Resume: re-run with the same `--run-name` (it continues from the last iteration and skips
already-benchmarked tasks). Vary the dev-subset size across runs with `MH_N_TASKS`.

### Choosing the dataset

Edit `config.yaml` `dataset.source` (and `dev_size`) to point at whichever set you fetched,
e.g. `source: spreadsheetbench_912_v0.1` for the full benchmark. `dev_size: 0` uses the
entire source.

## Model & keys

Solver = `gpt-oss-120b` via QNAIGC (`https://api.qnaigc.com/v1`), key `QNAIGC_API_KEY`
(fallback `OPENAI_API_KEY`), configured in `config.yaml`. Same plumbing as
`text_classification_local`. The proposer (Claude Code) uses subscription auth — the outer
loop strips `ANTHROPIC_API_KEY` before launching it, so it is decoupled from the solver key.

**OpenRouter override (one-line swap).** To run where QNAIGC isn't reachable (e.g. a compute
node), set `model.name: openrouter/openai/gpt-oss-120b` and `model.api_base: null` in
`config.yaml` (or pass `--model openrouter/openai/gpt-oss-120b`), and export
`OPENROUTER_API_KEY`. This path is proven working (the pipeline was validated on it).

## Dev-subset size is the knob

`config.yaml dataset.dev_size` (default 30) sets how many instructions the search runs on;
override per-run with `MH_N_TASKS`. `dev_size: 0` uses the whole source.

## Evaluation fidelity

Default backend `value`: compare cached/literal values with openpyxl `data_only=True`. This
avoids LibreOffice **only if solutions write literal values** — the base prompts require that.
If a formula-output task is misjudged, wire up `evaluation.recalc_with_libreoffice` (soffice
inside an Apptainer container; the `apptainer` module is available on Vulcan) and set
`eval.backend: libreoffice`. Validate against real answer files before trusting new scores.
