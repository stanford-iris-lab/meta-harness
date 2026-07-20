"""Deterministic generality gate on the SOURCE CODE: reject constants coupled to the one
visible instance's data extent and force ONE data-derived re-solve.

Parent: ``baseline_react`` (a bare ``SpreadsheetAgent``). The loop debugs the program on
exactly ONE input instance (``test_cases[0]``); the accepted program is then re-run,
unchanged, on ~3 hidden inputs that are structurally the same but a DIFFERENT SIZE, and the
instruction passes only if ALL of them match. The single failure family that every prior
candidate is structurally blind to is **overfitting to that one visible instance**: the model
bakes a size read off the current sheet — the number of data rows/columns (e.g.
``LAST_DATA_COL = 21``, the last column that actually holds data) — into the program as a
constant, which makes the VISIBLE case pass and silently under/over-reads on the
differently-sized hidden cases. Every tested post-solve gate
(readback_verify / answer_type_gate / code_answer_guard) inspects the produced OUTPUT, which is
*correct on the visible case*, so it can never see this bug; and passive prompt context
(recon_profile / formula_transplant) and the base prompt's own "never hardcode row counts" line
get raced past. The observed signature is idx0(visible)=pass, idx1+(hidden)=fail with an
off-by-a-row/column value error.

ONE new mechanism: a **deterministic** pre-acceptance gate that reads the generated PROGRAM's
own constants. After a solution runs, the scaffold parses ``solution_code`` (``ast``) and flags
any integer literal ``L`` (``L >= 3``, so ubiquitous 0/1/2 indexing is ignored) that (a) equals a
size of THIS input instance — the last row/column that actually holds data, a sheet ``max_row``/
``max_column``, or the non-empty-row count, all computed by scanning ``primary_input`` — and
(b) is NOT a fixed bound named by
``answer_position`` (those are legitimately constant across instances and are whitelisted, with
±1 to allow ``range(start, end+1)`` idioms). On a hit the loop withholds acceptance ONCE, feeds
back the exact offending line(s) and the concrete instruction ("that constant equals this
instance's max_column; derive it from the data — ws.max_column / len(...) — because the program
is re-run on differently-sized inputs"), and requests a corrected solve. A constant-clean
program fires nothing: no extra LLM call, byte-for-byte the baseline. This is why it cannot
regress a currently-passing task — both passers already derive their extents dynamically
(36764 iterates to ``ws2.max_row``; 547-18 to ``src.max_row``/``src.max_column`` and its only
``>2`` literals ``7``/``4`` are its A1:D7 answer bounds → whitelisted) so the gate never fires on
them — and why it adds no per-call 502 exposure on tasks that pass.

Distinct from every prior candidate:
  * readback_verify / answer_type_gate / code_answer_guard judged the produced OUTPUT (correct
    on the visible case → blind to instance-overfit); this reads the generated SOURCE for a
    constant that couples the program to one instance — the direct signature of the bug.
  * recon_profile / formula_transplant / value_read_solve changed the prompt CONTEXT (raced
    past); this confronts the concrete emitted constant, the lever code_answer_guard showed is
    what actually moves a small model off a wrong artifact.
  * grounded_recon_gate gated on whether the data was seen at all; this gates on whether the
    program encodes a size it should have derived.

No task-specific hints: it compares the program's literals only against the input's OWN
dimensions (never an answer key, never scoring internals, no hardcoded column/row/value/rule);
the correction is a general Excel-generality principle. The first runnable program is always
retained as the fallback, and the gate only fires on a program that is already certain to
misgeneralize, so it can only replace it with a later runnable solve — never do worse than the
baseline. ``call_llm`` is wrapped so a transient backend 502 degrades to best-so-far acceptance
(as in grounded_recon_gate) instead of aborting the run.
"""

from __future__ import annotations

import ast
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.utils import range_boundaries

from agent import PASSTHROUGH_SOLUTION, SpreadsheetAgent
from executor import run_code
from sheet_utils import parse_answer_position


class AgentHarness(SpreadsheetAgent):
    # Lifted only to fund the single correction round the gate may request; the mechanism is
    # the deterministic source-code extent gate, not the larger budget.
    MAX_TURNS = 7
    MAX_FIX_ROUNDS = 1        # at most one gate-triggered correction per instruction
    MIN_LITERAL = 3           # ignore 0/1/2 — ubiquitous indexing, never a meaningful extent
    SCAN_ROWS = 5000          # bounded row scan so extent stats stay cheap on large sheets
    SCAN_COLS = 60            # bounded column scan
    MAX_VIOLATIONS = 6        # cap the literals reported back

    # ── deterministic reconnaissance of the one visible instance's sizes ────
    def _instance_magnitudes(self, primary_input) -> dict[int, str]:
        """Map each instance-size integer -> a short label describing which dimension it is.

        These are the numbers a model reads off the preview and might bake in. Crucially we
        include the REAL data extent (the last row/column that actually contains a value), not
        just ``openpyxl``'s ``max_row``/``max_column`` — those can be inflated by trailing blank
        cells (e.g. a sheet whose data ends at column U=21 but whose ``max_column`` is 25), while
        the model hard-codes the *data* extent it sees in the preview. Any failure returns an
        empty map → the baseline accept-first behaviour.
        """
        mags: dict[int, str] = {}

        def _add(v, label: str) -> None:
            if isinstance(v, int) and not isinstance(v, bool) and v >= self.MIN_LITERAL:
                mags.setdefault(v, label)

        try:
            wb = load_workbook(primary_input, data_only=True)
        except Exception:
            return mags
        try:
            for ws in wb.worksheets:
                max_row = ws.max_row or 0
                max_col = ws.max_column or 0
                _add(max_row, f"{ws.title!r} max_row (openpyxl extent)")
                _add(max_col, f"{ws.title!r} max_column (openpyxl extent)")
                # Scan for the true data extents and non-empty-row count.
                scan_rows = min(max_row, self.SCAN_ROWS)
                scan_cols = min(max_col, self.SCAN_COLS)
                last_data_row = 0
                last_data_col = 0
                n_nonempty = 0
                for r, row in enumerate(
                    ws.iter_rows(min_row=1, max_row=scan_rows, max_col=scan_cols,
                                 values_only=True),
                    start=1,
                ):
                    row_has_data = False
                    for i, v in enumerate(row, start=1):
                        if v is not None and v != "":
                            row_has_data = True
                            if i > last_data_col:
                                last_data_col = i
                    if row_has_data:
                        n_nonempty += 1
                        last_data_row = r
                _add(last_data_row, f"{ws.title!r} last row that contains data")
                _add(last_data_col, f"{ws.title!r} last column that contains data")
                _add(n_nonempty, f"{ws.title!r} number of non-empty rows")
                _add(n_nonempty - 1, f"{ws.title!r} data-row count (rows minus a header)")
        except Exception:
            return mags
        return mags

    def _answer_whitelist(self, task) -> set[int]:
        """Integers that are legitimately fixed by the task (answer-range bounds), so a program
        may hold them as constants across instances. Includes ±1 for ``range(a, b+1)`` idioms."""
        wl: set[int] = set()
        try:
            for _sheet, rng in parse_answer_position(task.answer_position):
                try:
                    min_c, min_r, max_c, max_r = range_boundaries(rng)
                except Exception:
                    continue
                if None in (min_c, min_r, max_c, max_r):
                    continue
                height = max_r - min_r + 1
                width = max_c - min_c + 1
                for base in (min_c, min_r, max_c, max_r, height, width):
                    for d in (-1, 0, 1):
                        wl.add(base + d)
        except Exception:
            return wl
        return wl

    def _extent_violations(self, solution_code: str, task, primary_input) -> list[tuple[int, int, str]]:
        """List (lineno, literal, matched-dimension-label) instance-coupled constants.

        A violation is an int literal in the program that equals a size of the visible instance
        and is not an answer-range bound. Any parse/scan failure → [] (baseline accept).
        """
        mags = self._instance_magnitudes(primary_input)
        if not mags:
            return []
        wl = self._answer_whitelist(task)
        try:
            tree = ast.parse(solution_code or "")
        except Exception:
            return []
        out: list[tuple[int, int, str]] = []
        seen: set[tuple[int, int]] = set()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, int)
                    and not isinstance(node.value, bool)):
                continue
            v = node.value
            if v < self.MIN_LITERAL or v in wl or v not in mags:
                continue
            key = (getattr(node, "lineno", 0), v)
            if key in seen:
                continue
            seen.add(key)
            out.append((getattr(node, "lineno", 0), v, mags[v]))
            if len(out) >= self.MAX_VIOLATIONS:
                break
        return out

    def _fix_prompt(self, violations: list[tuple[int, int, str]]) -> str:
        lines = [
            f"  - line {lineno}: the literal {v} equals this input's {label}."
            for lineno, v, label in violations
        ]
        body = "\n".join(lines)
        return (
            "Automated generality check (deterministic — it inspects only the CONSTANTS in your "
            "program against the dimensions of the single input instance you can see, never an "
            "answer key). Your program ran, but it hard-codes number(s) that equal this "
            "instance's own size:\n"
            f"{body}\n\n"
            "The exact same program is re-run on several hidden inputs that are structurally the "
            "same but a DIFFERENT SIZE (more or fewer rows/columns), and it must pass ALL of "
            "them. A constant read off the current sheet's extent will be silently wrong there — "
            "it passes now only because you are debugging on this one instance. Replace each such "
            "constant with a value DERIVED FROM THE DATA at run time (e.g. ws.max_row, "
            "ws.max_column, len(list(...)), or a scan for the last non-empty cell). Do not hard-"
            "code a size you read off the preview. (A bound explicitly named by answer_position "
            "is fine to keep.) Resubmit the complete program:\n"
            "ACTION: solve\n"
            "```python\n"
            "# read input.xlsx, derive every range/size from the data, compute and write literal "
            "values into the answer cells, save output.xlsx\n"
            "```"
        )

    # ── main loop (baseline structure + the deterministic source-code gate) ──
    def run_loop(self, task, primary_input: Path, workdir: Path) -> dict:
        messages = [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": self.build_user_prompt(task, primary_input)},
        ]
        solution_code = None   # first runnable solution kept as the fallback
        produced = False
        turns = 0
        code = None
        fix_used = 0

        for turn in range(self.MAX_TURNS):
            turns = turn + 1
            messages = self.summarize_context(messages)
            try:
                response = self.call_llm(messages)
            except Exception:
                # Transient backend failure (e.g. a 502): degrade to best-so-far acceptance
                # (fallback / passthrough) rather than aborting the whole run.
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
                res = run_code(
                    code, primary_input, run_dir,
                    timeout=self.CODE_TIMEOUT, expect_output=True,
                    python_exe=self.python_exe,
                )
                if res.ok:
                    solution_code = code
                    produced = True
                    # Deterministic gate: withhold acceptance only when the program hard-codes a
                    # size of this instance, and only once. A constant-clean program is accepted
                    # immediately with no extra LLM call (identical to the baseline).
                    if fix_used < self.MAX_FIX_ROUNDS:
                        try:
                            violations = self._extent_violations(code, task, primary_input)
                        except Exception:
                            violations = []
                        if violations:
                            messages.append(
                                {"role": "user", "content": self._fix_prompt(violations)}
                            )
                            fix_used += 1
                            continue
                    break  # accept: constant-clean, or the one correction round is spent
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
