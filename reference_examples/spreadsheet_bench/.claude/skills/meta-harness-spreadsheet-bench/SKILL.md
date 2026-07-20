---
name: meta-harness-spreadsheet-bench
description: Run one iteration of AgentHarness (scaffold) evolution for SpreadsheetBench.
---

# Meta-Harness: SpreadsheetBench scaffold evolution

You evolve the **agent scaffold** that drives a cheap LLM (gpt-oss-120b) to solve
SpreadsheetBench instructions. Each instruction gives an Excel file, an instruction, an
`instruction_type` (Cell-Level / Sheet-Level), and an `answer_position`. The agent writes a
Python **solution program** that reads `input.xlsx` and writes `output.xlsx` with literal
values at the answer positions. That program is re-run on ~3 hidden test-case inputs and
scored OJ-style (ALL test cases must match → the instruction passes).

**You do NOT run benchmarks.** You analyze results + failed trajectories, form a hypothesis,
implement ONE new scaffold variant, and import-check it. The outer loop (`meta_harness.py`)
validates, smoke-tests, benchmarks, and updates the frontier.

## Hard constraints
- Produce **exactly ONE** new scaffold variant this iteration. Never write "the frontier is
  optimal" or stop early.
- **No task-specific hints.** Never mention a dataset/task name, and never hardcode column
  letters, row counts, answer values, or dataset-specific heuristics in code or prompts. The
  scaffold must generalize across arbitrary spreadsheet instructions.
- **One mechanism per candidate.** Change a real mechanism, not just a constant. A variant
  that only bumps `MAX_TURNS` or reworders the prompt is not acceptable on its own.

## Search space — subclass `SpreadsheetAgent`
Your candidate is a single file `agents/<name>.py` defining `class AgentHarness(SpreadsheetAgent)`
(import it: `from agent import SpreadsheetAgent`). The base class (`agent.py`) runs a ReAct
loop; override any of these to change behavior:

- `build_system_prompt()` / `SYSTEM_TEMPLATE` — the protocol prompt (you may add a new
  `prompt-templates/<name>.txt` and point `SYSTEM_TEMPLATE` at it).
- `build_user_prompt(task, primary_input)` — how the instruction + sheet preview are shown.
- `parse_action(response)` — how you extract the action (`explore`/`solve`) and code.
- `observe(res, kind)` — how execution feedback (stdout/stderr) is fed back.
- `run_loop(task, primary_input, workdir)` — the whole control loop (exploration budget,
  when to accept a solution, self-verification, retries, voting, etc.).
- `summarize_context(messages)` — context management for long trajectories.
- `call_llm(messages)` — e.g. sampling strategy.
- Knobs: `MAX_TURNS`, `CODE_TIMEOUT`, `PREVIEW_ROWS`.

Ideas that tend to matter: better exploration before solving (inspect shapes, headers,
dtypes, edge rows); using execution feedback to self-check the produced `output.xlsx`
against the instruction before accepting; making the model prove its program is general
(not overfit to the visible rows); robust parsing; light self-consistency.

Reusable execution: `from executor import run_code` (runs code with a fresh `input.xlsx`
and captures stdout/stderr; `expect_output=True` also requires `output.xlsx`). Sheet preview:
`from sheet_utils import preview_spreadsheet`. Do NOT read the answer files or the scoring
code to shape a solution — that is evaluation leakage.

## Candidate rules
- CAN: edit your new `agents/<name>.py` and add `prompt-templates/<name>.txt`.
- MUST name the class `AgentHarness` (the loop normalizes the import path to
  `agents.<name>:AgentHarness`).
- MUST NOT modify other agent files, `agent.py`, `executor.py`, `evaluation.py`,
  `sheet_utils.py`, `benchmark.py`, `meta_harness.py`, `inner_loop.py`, or `claude_wrapper.py`,
  and MUST NOT import from other candidate agents (copy any reused code).
- The solution the model writes must produce literal values (not formulas) — keep that
  requirement in any prompt you author.

## Workflow
1. Launch ONE general-purpose `Agent` subagent to read the run state: `evolution_summary.jsonl`
   and `frontier_val.json`, plus several **failed** trajectories under
   `logs/<run>/<agent>/task_<id>.jsonl` (look at where solutions crash, produce wrong values,
   or overfit the visible rows). Read `agent.py` and the current frontier agent. Have it
   return: STATE (what works / fails), HYPOTHESIS (one falsifiable claim), CANDIDATE (the
   concrete mechanism to implement).
2. Implement: copy the chosen parent to `agents/<name>.py`, make the ONE targeted change,
   self-critique for leakage/overfitting, and import-check:
   `uv run python -c "from agents.<name> import *; print('OK')"`.
3. Write `logs/<run>/pending_eval.json` and print `CANDIDATES: <name>`.

### pending_eval.json schema
```json
{
  "iteration": <N>,
  "candidates": [
    {
      "name": "<snake_case_name>",
      "import_path": "agents.<name>:AgentHarness",
      "hypothesis": "<one falsifiable claim>",
      "changes": "<what mechanism changed vs the parent>"
    }
  ]
}
```
