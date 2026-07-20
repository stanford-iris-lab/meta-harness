"""Base spreadsheet agent scaffold.

`SpreadsheetAgent` drives a multi-round ReAct loop: the model explores the input,
then submits a Python *solution program* that reads `input.xlsx` and writes
`output.xlsx` with literal values in the answer positions. The loop debugs the program
against the primary (first) test-case input only — answers are never shown. The returned
`solution_code` is later re-run on every OJ test case by `evaluation.score_instruction`.

This is the search surface for the meta-harness. Candidate agents subclass this and
override methods (`build_user_prompt`, `parse_action`, `run_loop`, `observe`,
`call_llm`, `summarize_context`) or tune the knobs (`MAX_TURNS`, `CODE_TIMEOUT`,
`PREVIEW_ROWS`). They must NOT modify shared modules.
"""

from __future__ import annotations

import re
from pathlib import Path

from executor import run_code
from sheet_utils import preview_spreadsheet

EVOLVE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = EVOLVE_DIR / "prompt-templates"

# A last-resort solution when the model never produces a runnable program: copy the
# input through unchanged so scoring produces a (failing) comparison instead of crashing.
PASSTHROUGH_SOLUTION = (
    "import shutil\n"
    "shutil.copyfile('input.xlsx', 'output.xlsx')\n"
)

_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_ACTION_RE = re.compile(r"ACTION:\s*(explore|solve)", re.IGNORECASE)


class SpreadsheetAgent:
    # Tunable knobs (overridable on subclasses or via config).
    MAX_TURNS = 6
    CODE_TIMEOUT = 60
    PREVIEW_ROWS = 20
    PREVIEW_COLS = 20
    SYSTEM_TEMPLATE = "react.txt"

    def __init__(self, llm, config: dict | None = None, python_exe: str | None = None):
        self.llm = llm
        self.python_exe = python_exe
        cfg = config or {}
        # config.agent knobs override class defaults when present
        self.MAX_TURNS = int(cfg.get("max_turns", self.MAX_TURNS))
        self.CODE_TIMEOUT = int(cfg.get("code_timeout_s", self.CODE_TIMEOUT))
        self.PREVIEW_ROWS = int(cfg.get("preview_rows", self.PREVIEW_ROWS))

    # ── prompt construction ────────────────────────────────────────────────
    def _template_path(self) -> Path:
        return TEMPLATES_DIR / self.SYSTEM_TEMPLATE

    def build_system_prompt(self) -> str:
        return self._template_path().read_text().format(max_turns=self.MAX_TURNS)

    def build_user_prompt(self, task, primary_input: Path) -> str:
        preview = preview_spreadsheet(
            primary_input, max_rows=self.PREVIEW_ROWS, max_cols=self.PREVIEW_COLS
        )
        sheet_line = ""
        if getattr(task, "answer_sheet", ""):
            sheet_line = f"### answer_sheet\n{task.answer_sheet}\n\n"
        return (
            f"### instruction\n{task.instruction}\n\n"
            f"### instruction_type\n{task.instruction_type}\n\n"
            f"### answer_position\n{task.answer_position}\n\n"
            f"{sheet_line}"
            f"### spreadsheet_content (preview of input.xlsx)\n{preview}\n"
        )

    # ── LLM + parsing ──────────────────────────────────────────────────────
    def call_llm(self, messages: list[dict]) -> str:
        return self.llm.chat(messages)

    def parse_action(self, response: str) -> tuple[str | None, str | None]:
        """Return (action, code). action in {"explore","solve",None}."""
        blocks = _CODE_FENCE_RE.findall(response or "")
        code = blocks[-1].strip() if blocks else None
        m = _ACTION_RE.search(response or "")
        action = m.group(1).lower() if m else None
        if action is None and code is not None:
            # Infer: writing output.xlsx implies a solution.
            action = "solve" if "output.xlsx" in code else "explore"
        return action, code

    # ── observations ───────────────────────────────────────────────────────
    def observe(self, res, kind: str) -> str:
        parts = []
        if res.stdout:
            parts.append(f"[stdout]\n{res.stdout}")
        if res.stderr:
            parts.append(f"[stderr]\n{res.stderr}")
        body = "\n".join(parts) if parts else "(no output)"
        if kind == "solve":
            if res.ok:
                return f"Your solution ran successfully and wrote output.xlsx.\n{body}"
            return (
                "Your solution FAILED (no output.xlsx or it errored). Fix the code and "
                f"resubmit with ACTION: solve.\n{body}"
            )
        return f"Execution result:\n{body}"

    def summarize_context(self, messages: list[dict]) -> list[dict]:
        """Hook for context management. Base keeps the full history."""
        return messages

    # ── main loop ──────────────────────────────────────────────────────────
    def run_loop(self, task, primary_input: Path, workdir: Path) -> dict:
        messages = [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": self.build_user_prompt(task, primary_input)},
        ]
        solution_code = None
        produced = False
        turns = 0
        code = None

        for turn in range(self.MAX_TURNS):
            turns = turn + 1
            messages = self.summarize_context(messages)
            response = self.call_llm(messages)
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
                res = run_code(
                    code, primary_input, run_dir,
                    timeout=self.CODE_TIMEOUT, expect_output=True,
                    python_exe=self.python_exe,
                )
                if res.ok:
                    solution_code = code
                    produced = True
                    break  # base scaffold accepts the first runnable solution
                messages.append({"role": "user", "content": self.observe(res, "solve")})
            else:  # explore
                res = run_code(
                    code, primary_input, run_dir,
                    timeout=self.CODE_TIMEOUT, expect_output=False,
                    python_exe=self.python_exe,
                )
                messages.append({"role": "user", "content": self.observe(res, "explore")})

        if solution_code is None:
            # No runnable solution; fall back so scoring still runs (and fails cleanly).
            solution_code = code if code else PASSTHROUGH_SOLUTION
        return {
            "solution_code": solution_code,
            "n_turns": turns,
            "produced": produced,
            "trajectory": messages,
        }

    def solve(self, task, workdir) -> dict:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        return self.run_loop(task, task.primary_input, workdir)
