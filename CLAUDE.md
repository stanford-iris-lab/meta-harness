# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Shape

This is a framework + reference experiments repo, **not** a single Python package. The top level has no `pyproject.toml`. Each reference experiment under `reference_examples/<name>/` is its own self-contained `uv` project with its own deps, scripts, agents, and outer-loop driver:

- `reference_examples/text_classification/` — memory-system search on MCE classification datasets.
- `reference_examples/terminal_bench_2/` — agent-scaffold search on Terminal-Bench 2.0 via Harbor.

Each experiment also ships its own `claude_wrapper.py` (identical between the two) and `.claude/skills/<skill>/SKILL.md` (the proposer prior). They are intentionally vendored per-experiment rather than shared — keep them in sync if you change one.

## Outer / Inner Loop Architecture

Both experiments follow the same Meta-Harness pattern:

- **Outer loop = `meta_harness.py`** (per experiment). Drives evolution: proposes candidates via the Claude Code CLI (`claude_wrapper.run(...)`), validates them, benchmarks them, updates the frontier, writes `evolution_summary.jsonl` / `frontier_val.json` / `pending_eval.json` under `logs/<run_name>/`.
- **Inner loop = the actual evaluator.** `benchmark.py` + `inner_loop.py` for text_classification, `scripts/run_eval.sh` → `uv run harbor run` for terminal_bench_2. The outer loop **does not** run benchmarks inside the proposer session — it dispatches them separately.
- **Proposer = a Claude Code subprocess.** `propose_claude` strips `ANTHROPIC_API_KEY` from env so the CLI uses subscription auth instead of the API key (which is reserved for the inner-loop solver). The proposer is bounded to `PROPOSER_ALLOWED_TOOLS` and loaded with the experiment's `.claude/skills/<skill>` directory.
- **Candidates live in `agents/`.** The proposer writes new `<name>.py` files there. The benchmark **auto-discovers** files in `agents/` — do not edit `config.yaml`'s `memory_systems.proposed` list just to register candidates. Baselines are kept under fixed filenames (see `BASELINE_FILES` in `meta_harness.py` for text_classification, `BASELINES` for terminal_bench_2) and must not be overwritten.
- **Per-run isolation.** Outputs go under `logs/<run_name>/` (auto-generated timestamp or `--run-name`). `--fresh` clears generated agents and prior logs.

## Common Commands

All commands assume you `cd` into the relevant `reference_examples/<name>/` first.

### text_classification

```bash
uv sync                                              # install deps
uv run python meta_harness.py --iterations 1         # one evolve iteration
uv run python meta_harness.py --iterations 10 --fresh --run-name my-run
uv run python benchmark.py --results                 # print benchmark summary
uv run python benchmark.py --memory <name>           # benchmark one system
uv run python benchmark.py --memory <name> --test    # held-out test eval
uv run python -m unittest tests.test_data            # run the data test suite
```

Single memory system on a single dataset — note the `PYTHONPATH=..` and module form, because `inner_loop.py` uses package-mode imports:

```bash
PYTHONPATH=.. uv run python -m text_classification.inner_loop \
  --memory fewshot_all --dataset Symptom2Disease
```

Candidate import-check (the same shape `validate_candidates` uses; must be run from the parent dir):

```bash
cd reference_examples
uv run --project text_classification python -c \
  "from text_classification.agents.<name> import *; print('OK')"
```

### terminal_bench_2

```bash
uv sync                                              # install deps (Python 3.12)
uv run python meta_harness.py --iterations 1        # one evolve iteration
uv run python meta_harness.py --iterations 1 --full-eval   # add 5-trial winner pass
uv run bash scripts/run_eval.sh agents.baseline_kira:AgentHarness full 1 1 -i extract-elf   # smoke
uv run bash scripts/run_eval.sh agents.baseline_kira:AgentHarness hard 1 50                 # cheap 30-task subset
```

`scripts/run_eval.sh` takes: `<agent_import_path> [hard|full] [runs] [n_concurrent] [extra_harbor_flags...]`. Extra flags are forwarded to `harbor run` (e.g. `-i <task>` to restrict to one task, `--job-name`, `--jobs-dir`). The script sources `.env` from `reference_examples/terminal_bench_2/.env` — put keys there, not only at the repo root.

Recommended bring-up order for any new TB2 idea (the full 89×2 default takes ~4–6h and ~$500 on Opus 4.6): `extract-elf` smoke → `hard` (30 tasks) → full search via `meta_harness.py`.

## Environment And Keys

`.env.example` at the repo root lists the keys both experiments may consume. The TB2 shell wrappers source `.env` from the **terminal_bench_2 directory**, not the repo root — duplicate `.env` there or export in your shell. TB2's shipped `runloop` path requires **both** `ANTHROPIC_API_KEY` and `RUNLOOP_API_KEY`.

Text-classification defaults to `openrouter/openai/gpt-oss-120b`; override `--model` (and optionally `--api-base`) or change `config.yaml`. The paper used a local vLLM `gpt-oss-120b` MXFP4 deployment with `max-model-len=32768` — API-backed runs may differ in quality.

The outer loop concurrency for both experiments is sensitive to API throughput. Many timeout-looking failures are actually rate-limit / throughput failures, not reasoning failures; sharing one API key across active projects slows runs noticeably.

## Constraints On Candidate Code

These come from the proposer skill files (`.claude/skills/.../SKILL.md`) and apply to any code you write into `agents/`:

- **text_classification:** every candidate must subclass `MemorySystem` (`memory_system.py`) and implement `predict`, `learn_from_batch`, `get_state`, `set_state`. Call the underlying LLM via `self.call_llm(prompt)` (it tracks per-thread last-prompt info for logging). The skill mandates 3 new systems per iteration and forbids dataset-specific hardcoding, dataset names in prompts/comments, and pure parameter-tweak variants of existing systems.
- **terminal_bench_2:** every candidate is a single file at `agents/<name>.py` with class `AgentHarness` subclassing `harbor.agents.terminus_2.terminus_2.Terminus2`. The agent is loaded by Harbor as `agents.<name>:AgentHarness`. Candidates may freely override Terminus2 methods (`_call_llm_with_tools`, `_parse_tool_calls`, `_execute_commands`, `_run_agent_loop`, `_summarize_context`, etc.) and may add a custom prompt template under `prompt-templates/<name>.txt`. Candidates must NOT modify other agent files, `meta_harness.py`, or `claude_wrapper.py`, and must not import from other candidate agents (copy any reused code).

## Data Layout (text_classification)

`reference_examples/text_classification/data/` is **whitelisted** in `.gitignore` (the global rule ignores all `**/data/` paths) — the MCE datasets are vendored and load locally via `load_mce_dataset`. The "transfer" datasets in `TRANSFER_TASKS` are pulled from HuggingFace at runtime; see `data/README.md` for provenance. Tests in `tests/test_data.py` pin the expected split sizes and prompt shapes for the vendored datasets — when touching loaders, run them.

## Onboarding A New Domain

`ONBOARDING.md` is a prompt for a coding assistant to interview a user and produce `domain_spec.md` for a new Meta-Harness application. It is intentionally rule-heavy (no `domain_spec.md` until every required field is filled or marked `unknown`, 1–2 focused questions at a time, hard-stop on evaluation-leakage risks). When asked to help adapt Meta-Harness to a new domain, follow `ONBOARDING.md` rather than improvising.
