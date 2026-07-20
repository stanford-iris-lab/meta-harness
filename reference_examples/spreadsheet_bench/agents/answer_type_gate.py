"""Deterministic answer-cell type-validity gate before acceptance.

Parent: ``baseline_react`` (a bare ``SpreadsheetAgent``). The baseline accepts the FIRST
program that merely *runs* (``if res.ok: break``) — it never looks at what the program
wrote, so a solve that executes cleanly but puts an impossibly-typed value into an answer
cell is accepted as-is. On the dev set this is exactly how an instruction *misread* slips
through: when the request is phrased as a question, the model answers it in prose and writes
a long text string into answer cells whose column is otherwise entirely numeric — the
program runs, so the baseline keeps it.

ONE new mechanism is added here: a **deterministic** post-solve gate. After a solution runs,
the scaffold itself (not the model) checks each written answer cell against a hard, task-
agnostic validity rule derived *only from the input's own column types* — no answer key, no
model self-judgment:

  * a **formula string** in any answer cell (openpyxl stores ``"=..."`` with no cached value,
    which the scorer reads back as empty → guaranteed 0), or
  * **free text** written into a cell whose column is strongly numeric/date in the input.

Only when such an objective violation exists does the loop withhold acceptance and feed back
the specific offending cell(s), asking for a corrected solve. When the output is type-clean
(the common case, including both currently-passing guardrail tasks) the gate fires nothing:
no extra LLM call, behaviour identical to the baseline. That property is deliberate — it is
why this cannot regress a passing task and why it does not raise per-call infra-flake exposure.

Distinct from the two tested post-baseline candidates: ``readback_verify`` showed the written
values back and let the *model* decide "VERIFY: pass/revise" (it rubber-stamped its own output
and tied); ``recon_profile`` injected column types *before* the model acted (passive context
the model raced past, and it regressed). Here the reject/accept decision is made
deterministically by the scaffold, and only *after* a concrete violation is measured, so it
is neither collusible nor a passive prompt dump. The first runnable program is always retained
as a fallback, so the gate can only replace it with a later runnable one — never do worse than
the baseline would.
"""

from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter, range_boundaries

from agent import PASSTHROUGH_SOLUTION, SpreadsheetAgent
from executor import run_code
from sheet_utils import parse_answer_position


def _kind(v) -> str:
    """Coarse, scorer-aligned value category for one cell.

    ``numeric`` folds ints/floats and numeric-looking text (the scorer compares numeric
    strings as numbers). ``formula`` is an openpyxl formula string (``"=..."``) read back
    without ``data_only`` — it has no cached value and scores as empty.
    """
    if v is None or v == "":
        return "blank"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (datetime, date, time)):
        return "date"
    if isinstance(v, (int, float)):
        return "numeric"
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("="):
            return "formula"
        t = s.replace(",", "")
        if t.endswith("%"):
            t = t[:-1]
        try:
            float(t)
            return "numeric"  # numeric-looking text; scorer treats it as a number
        except ValueError:
            return "text"
    return "text"


def _resolve_ws(wb, sheet_name):
    if sheet_name is None:
        return wb.active
    if sheet_name in wb.sheetnames:
        return wb[sheet_name]
    return None


class AgentHarness(SpreadsheetAgent):
    # MAX_TURNS is lifted only to fund the single correction round the gate may request;
    # the mechanism is the deterministic type gate, not the larger budget.
    MAX_TURNS = 7
    MAX_FIX_ROUNDS = 1        # at most one gate-triggered correction per instruction

    # Column-typing is intentionally strict so the gate never second-guesses a valid solve.
    TYPE_SCAN_ROWS = 3000     # rows scanned to characterise a column (bounded on big sheets)
    TYPE_MIN_CELLS = 4        # need this many typed cells before a column is called "numeric"
    TYPE_DOMINANCE = 0.9      # and this fraction sharing one numeric/date category
    MAX_VIOLATIONS = 8        # cap the cells reported back

    # ── deterministic type check ───────────────────────────────────────────
    def _column_is_numeric(self, ws, col_idx: int, answer_rows: set[int]) -> bool:
        """True iff column ``col_idx`` is strongly numeric/date in the INPUT.

        Rows inside the answer range and the (usually textual) header row are excluded so we
        judge the column by its real data, never by the answer cells we are about to check.
        """
        cap = min(ws.max_row or 0, self.TYPE_SCAN_ROWS)
        if cap < 2:
            return False
        counts: dict[str, int] = {}
        total = 0
        for r, (val,) in enumerate(
            ws.iter_rows(
                min_row=2, max_row=cap, min_col=col_idx, max_col=col_idx, values_only=True
            ),
            start=2,
        ):
            if r in answer_rows:
                continue
            k = _kind(val)
            if k in ("blank", "formula"):
                continue
            counts[k] = counts.get(k, 0) + 1
            total += 1
        if total < self.TYPE_MIN_CELLS:
            return False
        numeric_like = counts.get("numeric", 0) + counts.get("date", 0)
        return numeric_like / total >= self.TYPE_DOMINANCE

    def _type_violations(self, task, output_path) -> list[tuple[str, object, str]]:
        """List (loc, written_value, expected) answer cells that cannot be correctly typed.

        Reads the produced output with ``data_only=False`` so formula strings are visible,
        and the input with ``data_only=True`` for its literal column types. Any failure
        degrades to an empty list → the baseline accept-first behaviour.
        """
        try:
            in_wb = load_workbook(task.primary_input, data_only=True)
            out_wb = load_workbook(output_path, data_only=False)
        except Exception:
            return []
        default_sheet = getattr(task, "answer_sheet", "") or None
        violations: list[tuple[str, object, str]] = []
        try:
            for sheet, rng in parse_answer_position(task.answer_position):
                sheet = sheet or default_sheet
                in_ws = _resolve_ws(in_wb, sheet)
                out_ws = _resolve_ws(out_wb, sheet)
                if in_ws is None or out_ws is None:
                    continue
                try:
                    min_c, min_r, max_c, max_r = range_boundaries(rng)
                except Exception:
                    continue
                answer_rows = set(range(min_r, max_r + 1))
                for c in range(min_c, max_c + 1):
                    numeric = self._column_is_numeric(in_ws, c, answer_rows)
                    for r in range(min_r, max_r + 1):
                        val = out_ws.cell(row=r, column=c).value
                        k = _kind(val)
                        loc = f"{get_column_letter(c)}{r}"
                        if k == "formula":
                            violations.append(
                                (loc, val, "a literal value (a formula stores no cached "
                                            "value and is scored as empty)")
                            )
                        elif numeric and k == "text":
                            violations.append(
                                (loc, val, f"a number/date — column {get_column_letter(c)} "
                                           "holds numeric data in the input")
                            )
                        if len(violations) >= self.MAX_VIOLATIONS:
                            return violations
        except Exception:
            return []
        return violations

    def _fix_prompt(self, violations) -> str:
        lines = []
        for loc, val, expected in violations:
            shown = repr(val)
            if len(shown) > 60:
                shown = shown[:57] + "...'"
            lines.append(f"  - {loc}: you wrote {shown}, but this cell must hold {expected}.")
        body = "\n".join(lines)
        return (
            "Automated type check (deterministic — it uses only the input's own column "
            "types, never an answer key). Your program ran, but these answer cells hold a "
            "value whose form cannot be correct and will be scored wrong:\n"
            f"{body}\n\n"
            "This is the signature of a misread request — e.g. answering/describing the task "
            "in prose instead of writing the computed data values, or emitting an Excel "
            "formula instead of a literal. Re-read the instruction, compute the intended "
            "literal values from the data, and resubmit the complete program:\n"
            "ACTION: solve\n"
            "```python\n"
            "# read input.xlsx, compute, write literal values into the answer cells, "
            "save output.xlsx\n"
            "```"
        )

    # ── main loop (baseline structure + the deterministic type gate) ────────
    def run_loop(self, task, primary_input: Path, workdir: Path) -> dict:
        messages = [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": self.build_user_prompt(task, primary_input)},
        ]
        solution_code = None  # first runnable solution kept as fallback
        produced = False
        turns = 0
        code = None
        fix_used = 0

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
                    # Deterministic gate: withhold acceptance only for an objective,
                    # measured type violation, and only once. A clean output is accepted
                    # immediately with no extra LLM call (identical to the baseline).
                    if fix_used < self.MAX_FIX_ROUNDS:
                        violations = self._type_violations(task, res.output_path)
                        if violations:
                            messages.append(
                                {"role": "user", "content": self._fix_prompt(violations)}
                            )
                            fix_used += 1
                            continue
                    break  # accept: type-clean, or the one correction round is spent
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
