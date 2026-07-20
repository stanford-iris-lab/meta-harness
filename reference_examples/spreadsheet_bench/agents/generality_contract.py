"""Plan-gated "generality contract": commit a concrete plan before any solution is accepted.

The baseline ReAct loop lets the model jump straight to a solution and accepts the first
program that merely *runs*. But the loop only ever shows ONE input (`test_cases[0]`) while
the program is later graded on several hidden inputs that differ in size, ordering, and the
exact cell values. With no pressure to generalize, the model bakes incidental properties of
the single visible instance into its code -- hardcoded row windows, case-normalized lookups,
and literal English-question misreads (writing prose/code text into the answer cells) -- which
pass the visible input yet fail the hidden ones. Notably the base system prompt ALREADY says
"write general code, never hardcode row counts", and it is ignored; and a post-hoc read-back
value check (iteration 1) recovered nothing because it only re-reads the visible output and
colludes with a misread the model has already committed.

ONE mechanism is added here: a mandatory PLAN phase enforced by the control loop. Before any
`ACTION: solve` is accepted, the model must first emit `ACTION: plan` and fill a fixed,
task-agnostic contract -- what concrete data (type/shape/meaning) the answer cells must hold
(translating a question-style instruction into output VALUES), how every extent and column is
derived FROM THE DATA rather than hardcoded, that looked-up values are transcribed faithfully
(case/spacing/type preserved), and how empty/duplicate/boundary rows are handled. Because the
commit *precedes* code generation, it reframes the task up front instead of rubber-stamping a
solution already written. Acceptance is otherwise identical to the baseline (the first runnable
solution wins) -- no verification, voting, or perturbation is added. A bounded nudge stops
gating after one reminder so a determined solver is never blocked (regression guard). No ground
truth, scoring code, or task-specific values are ever used.

MAX_TURNS is lifted 6->8 solely to fund the extra plan turn; the mechanism is the plan gate,
not the larger budget.
"""

from __future__ import annotations

import re
from pathlib import Path

from agent import PASSTHROUGH_SOLUTION, SpreadsheetAgent
from executor import run_code

_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_ACTION_RE = re.compile(r"ACTION:\s*(plan|explore|solve)", re.IGNORECASE)


class AgentHarness(SpreadsheetAgent):
    # +2 turns solely to fund the mandatory plan phase (the mechanism is the plan gate).
    MAX_TURNS = 8
    SYSTEM_TEMPLATE = "generality_contract.txt"
    # After this many "plan first" nudges, stop gating solves (avoid livelock / regression).
    MAX_PLAN_NUDGES = 1

    def build_user_prompt(self, task, primary_input: Path) -> str:
        base = super().build_user_prompt(task, primary_input)
        return base + "\nBegin with ACTION: plan (commit the contract), then explore/solve.\n"

    def parse_action(self, response: str):
        """Extend the base parser to recognize ACTION: plan (which carries no code block)."""
        blocks = _CODE_FENCE_RE.findall(response or "")
        code = blocks[-1].strip() if blocks else None
        m = _ACTION_RE.search(response or "")
        action = m.group(1).lower() if m else None
        if action is None and code is not None:
            # Infer as before: writing output.xlsx implies a solution, else exploration.
            action = "solve" if "output.xlsx" in code else "explore"
        return action, code

    def run_loop(self, task, primary_input: Path, workdir: Path) -> dict:
        messages = [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": self.build_user_prompt(task, primary_input)},
        ]
        solution_code = None
        produced = False
        turns = 0
        code = None
        plan_committed = False
        plan_nudges = 0

        for turn in range(self.MAX_TURNS):
            turns = turn + 1
            messages = self.summarize_context(messages)
            response = self.call_llm(messages)
            messages.append({"role": "assistant", "content": response})

            action, code = self.parse_action(response)

            # ── PLAN: record the contract; nothing is executed this turn ────────
            if action == "plan":
                plan_committed = True
                messages.append({
                    "role": "user",
                    "content": (
                        "Plan committed. Verify its assumptions against the real data with "
                        "ACTION: explore if useful, then submit ACTION: solve. Your program "
                        "must satisfy every point of your plan."
                    ),
                })
                continue

            if code is None:
                messages.append({
                    "role": "user",
                    "content": (
                        "No code block found. Respond with an ACTION line and one fenced "
                        "```python``` block (or ACTION: plan with your written plan)."
                    ),
                })
                continue

            run_dir = workdir / f"turn_{turns}"
            if action == "solve":
                # Gate: require a committed plan before the first accepted solution. The
                # nudge is bounded so a model that refuses to plan is never blocked forever.
                if not plan_committed and plan_nudges < self.MAX_PLAN_NUDGES:
                    plan_nudges += 1
                    messages.append({
                        "role": "user",
                        "content": (
                            "Before solving, commit your plan first. Reply with ACTION: plan "
                            "and fill the contract (OUTPUT / LOCATE / VALUES / EDGES / "
                            "LITERALS) for this task, then submit your solution."
                        ),
                    })
                    continue
                res = run_code(
                    code, primary_input, run_dir,
                    timeout=self.CODE_TIMEOUT, expect_output=True,
                    python_exe=self.python_exe,
                )
                if res.ok:
                    solution_code = code
                    produced = True
                    break  # accept the first runnable solution (base behavior)
                messages.append({"role": "user", "content": self.observe(res, "solve")})
            else:  # explore
                res = run_code(
                    code, primary_input, run_dir,
                    timeout=self.CODE_TIMEOUT, expect_output=False,
                    python_exe=self.python_exe,
                )
                messages.append({"role": "user", "content": self.observe(res, "explore")})

        if solution_code is None:
            solution_code = code if code else PASSTHROUGH_SOLUTION
        return {
            "solution_code": solution_code,
            "n_turns": turns,
            "produced": produced,
            "trajectory": messages,
        }
