"""Single-round control baseline.

One generation, no exploration and no execution feedback: present the instruction plus a
preview of the input, take the single solution program, and return it. Establishes the
floor that multi-round scaffolds must beat.
"""

from pathlib import Path

from agent import PASSTHROUGH_SOLUTION, SpreadsheetAgent
from executor import run_code


class AgentHarness(SpreadsheetAgent):
    SYSTEM_TEMPLATE = "single.txt"
    MAX_TURNS = 1

    def run_loop(self, task, primary_input: Path, workdir: Path) -> dict:
        messages = [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": self.build_user_prompt(task, primary_input)},
        ]
        response = self.call_llm(messages)
        messages.append({"role": "assistant", "content": response})

        _, code = self.parse_action(response)
        if not code:
            code = PASSTHROUGH_SOLUTION

        # Best-effort single execution just to record whether it runs; we return the
        # program regardless (OJ scoring re-runs it on every test case).
        res = run_code(
            code, primary_input, workdir / "turn_1",
            timeout=self.CODE_TIMEOUT, expect_output=True, python_exe=self.python_exe,
        )
        return {
            "solution_code": code,
            "n_turns": 1,
            "produced": res.ok,
            "trajectory": messages,
        }
