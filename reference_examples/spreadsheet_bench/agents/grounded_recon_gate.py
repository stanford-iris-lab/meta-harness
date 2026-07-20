"""Grounded-reconnaissance gate.

Parent: baseline_react. ONE new mechanism in ``run_loop``: a solve is not accepted
until the model has actually OBSERVED the data -- i.e. at least one prior explore
turn ran without error AND produced non-empty stdout. A crashed or empty
exploration is explicitly NOT treated as grounding, which blocks the observed
"I imagined the printed output and solved from it" failure. If the model tries to
solve before it has grounded, the loop withholds execution and requires a concrete
(task-agnostic) reconnaissance pass; once grounded, acceptance is identical to the
baseline (first runnable solve wins), so no extra LLM call is spent on the common
accept path.

Distinct from ``recon_profile`` (which injected a deterministic profile the model
raced past -- here the model must WRITE, RUN and READ its own reconnaissance) and
from the post-hoc output gates (readback_verify / answer_type_gate /
code_answer_guard, which judged the produced output) -- this acts BEFORE the solve,
on whether the data was seen at all, not on the artifact that was written.
"""

from __future__ import annotations

from pathlib import Path

from agent import PASSTHROUGH_SOLUTION, SpreadsheetAgent
from executor import run_code


# Task-agnostic reconnaissance the model must perform before it may solve.
RECON_REQUEST = (
    "STOP -- do not solve yet. You have not actually inspected this workbook, so a "
    "solution now would be guesswork. First send `ACTION: explore` with code that "
    "PRINTS, for EVERY sheet in input.xlsx:\n"
    "  - the sheet name and its (max_row, max_column);\n"
    "  - the header row, and for each column the Python type(s) of its data values;\n"
    "  - the CURRENT contents of the answer_position cells, loaded BOTH with "
    "`data_only=True` (cached values) AND `data_only=False` (raw). If they differ, the "
    "cells are formula-backed: recompute the value in Python -- never copy the formula "
    "string or a blank;\n"
    "  - a few rows sampled from the TOP and the BOTTOM of the used range;\n"
    "  - note whether any other sheet illustrates the expected result shape.\n"
    "Next turn, let the PRINTED output -- not your assumptions -- drive the solution."
)

EMPTY_RECON_NOTE = (
    "\n\nNOTE: that exploration produced no usable output (it printed nothing or "
    "crashed), so you still have NOT seen the data. Do not solve from assumptions or "
    "pretend you saw values -- send a working `ACTION: explore` that actually prints "
    "what you need."
)


class AgentHarness(SpreadsheetAgent):
    # Base 6 + headroom for the forced reconnaissance turn(s); the gate is the
    # mechanism, this bump only funds it.
    MAX_TURNS = 8
    # After this many "solve-before-grounded" nudges, stop gating so a determined
    # solver is never blocked -> the loop can never livelock, never worse than base.
    MAX_RECON_NUDGES = 2

    def run_loop(self, task, primary_input: Path, workdir: Path) -> dict:
        messages = [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": self.build_user_prompt(task, primary_input)},
        ]
        solution_code = None
        produced = False
        turns = 0
        code = None
        grounded = False   # a real, non-empty exploration has been observed
        nudges = 0

        for turn in range(self.MAX_TURNS):
            turns = turn + 1
            messages = self.summarize_context(messages)
            try:
                response = self.call_llm(messages)
            except Exception:
                # Transient backend failure (e.g. a 502): degrade to baseline
                # acceptance (best-so-far / passthrough) rather than aborting the run.
                break
            messages.append({"role": "assistant", "content": response})

            action, code = self.parse_action(response)
            if code is None:
                messages.append({
                    "role": "user",
                    "content": (
                        "No code block found. Respond with an ACTION line and one fenced "
                        "```python``` block."
                    ),
                })
                continue

            run_dir = workdir / f"turn_{turns}"
            if action == "solve":
                # Gate: withhold the solve until the data has actually been observed.
                if not grounded and nudges < self.MAX_RECON_NUDGES:
                    nudges += 1
                    messages.append({"role": "user", "content": RECON_REQUEST})
                    continue
                res = run_code(
                    code, primary_input, run_dir,
                    timeout=self.CODE_TIMEOUT, expect_output=True,
                    python_exe=self.python_exe,
                )
                if res.ok:
                    solution_code = code
                    produced = True
                    break  # accept the first runnable solution (once grounded)
                messages.append({"role": "user", "content": self.observe(res, "solve")})
            else:  # explore
                res = run_code(
                    code, primary_input, run_dir,
                    timeout=self.CODE_TIMEOUT, expect_output=False,
                    python_exe=self.python_exe,
                )
                feedback = self.observe(res, "explore")
                if res.ok and (res.stdout or "").strip():
                    grounded = True   # a genuine, non-empty observation exists
                else:
                    feedback += EMPTY_RECON_NOTE  # crash / empty is NOT grounding
                messages.append({"role": "user", "content": feedback})

        if solution_code is None:
            # No runnable solution; fall back so scoring still runs (and fails cleanly).
            solution_code = code if code else PASSTHROUGH_SOLUTION
        return {
            "solution_code": solution_code,
            "n_turns": turns,
            "produced": produced,
            "trajectory": messages,
        }
