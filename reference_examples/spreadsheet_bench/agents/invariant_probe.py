"""Predict-then-perturb invariance probe.

Parent: baseline_react. ONE new mechanism in ``run_loop``: a runnable solution is not
accepted until the model's own *declared* behavioural invariants have been checked
MECHANICALLY, by re-running the same program on structurally perturbed copies of
``input.xlsx`` and diffing only the ``answer_position`` cells against an unperturbed run.

Why this and not another output check. Every previous gate judged the artifact the
program wrote for the one visible input -- but on the failures that matter the visible
output is *correct*; the program is simply tuned to an incidental property of the rows in
front of it (keys happen to be unique, rows happen to be in the right order, capitalisation
happens to match, one sheet happens to already contain the answer). No amount of looking at
that output can reveal this, and asking the model to judge its own output invites it to
agree with itself. Perturbing the *input* creates evidence that does not exist in the
visible instance, and the verdict is a diff, not an opinion.

The declaration is carried in the solve message itself, so the common "prediction matches
observation" path costs ZERO extra LLM calls; a single revision turn is spent only when the
check finds a concrete contradiction. Acceptance is monotone: the first runnable solution is
retained and a revision replaces it only if the revision also runs, so this can never do
worse than the parent's "first runnable solve wins".

Distinct from readback_verify / answer_type_gate / code_answer_guard (post-hoc judgements of
the produced output), from grounded_recon_gate / recon_profile (pre-solve reconnaissance of
the given data) and from instance_extent_gate (static AST inspection of the source): here the
program is *executed on inputs it has never seen* and graded against a contract it wrote.
"""

from __future__ import annotations

import re
from pathlib import Path

import openpyxl

from agent import PASSTHROUGH_SOLUTION, SpreadsheetAgent
from executor import run_code
from sheet_utils import cells_in_range, parse_answer_position

# Declaration lines the model emits alongside a solve, parsed from the prose that
# precedes the code fence.
_DECL_RE = re.compile(
    r"^\s*(DEPENDS|DUP_ROW|ROW_ORDER|CASE)\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_INVARIANCE_KEYS = ("DUP_ROW", "ROW_ORDER", "CASE")
# A sheet the answer supposedly ignores is probed with the first perturbation that
# applies; row reversal leads because it leaves no trace in values that are merely
# copied through, so any movement it causes is a genuine read of that sheet.
_SPURIOUS_KINDS = ("ROW_ORDER", "DUP_ROW", "CASE")
_MUTATION_LABEL = {
    "CASE": "flipping the letter-case of the text",
    "DUP_ROW": "appending a duplicate of the last data row",
    "ROW_ORDER": "reversing the order of the data rows",
}

_REPORT_HEADER = (
    "INVARIANCE CHECK -- your program was re-run on modified copies of input.xlsx and the "
    "answer_position cells were compared against the unmodified run. The following "
    "observations CONTRADICT the invariants you declared:\n\n"
)

_REPORT_FOOTER = (
    "\nThese modifications are structure-preserving: the hidden inputs this program will be "
    "re-run on differ from the one you saw in exactly these ways. Decide which side of each "
    "contradiction is wrong.\n"
    "- If the PROGRAM is wrong, resubmit a corrected `ACTION: solve` with a fresh INVARIANTS "
    "block.\n"
    "- If your PREDICTION was wrong -- the task really does require that behaviour -- "
    "resubmit the same program with a corrected INVARIANTS block and say in one line why.\n"
    "Do not change anything else about the program."
)


def _last_data_row(ws) -> int:
    """Largest row index holding any non-blank cell (openpyxl's max_row over-counts)."""
    last = 0
    for row in ws.iter_rows():
        for cell in row:
            v = cell.value
            if v is not None and not (isinstance(v, str) and not v.strip()):
                last = max(last, cell.row)
                break
    return last


def _write_mutation(src: Path, dst: Path, sheet_name: str, kind: str) -> bool:
    """Write a perturbed copy of `src` to `dst`. False if the perturbation does not apply.

    Perturbations never insert or delete rows: they only rewrite cells inside the used
    block or fill the first entirely-blank row beneath it, so cell addresses -- and hence
    the answer_position range -- never shift.
    """
    wb = openpyxl.load_workbook(src)
    if sheet_name not in wb.sheetnames:
        return False
    ws = wb[sheet_name]
    last = _last_data_row(ws)
    width = ws.max_column or 0
    changed = False

    if kind == "IDENTITY":
        changed = True
    elif kind == "CASE":
        for row in ws.iter_rows():
            for cell in row:
                v = cell.value
                if isinstance(v, str) and v.strip() and not v.startswith("="):
                    swapped = v.swapcase()
                    if swapped != v:
                        cell.value = swapped
                        changed = True
    elif kind == "DUP_ROW":
        # Copy the last data row into the blank row directly beneath it.
        if last >= 2 and width:
            for col in range(1, width + 1):
                ws.cell(row=last + 1, column=col).value = ws.cell(row=last, column=col).value
            changed = True
    elif kind == "ROW_ORDER":
        # Reverse the data rows in place, treating row 1 as a header.
        if last >= 4 and width:
            block = [
                [ws.cell(row=r, column=c).value for c in range(1, width + 1)]
                for r in range(2, last + 1)
            ]
            if any(vals != block[-1 - i] for i, vals in enumerate(block)):
                for i, vals in enumerate(reversed(block)):
                    for j, v in enumerate(vals):
                        ws.cell(row=2 + i, column=j + 1).value = v
                changed = True

    if not changed:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    wb.save(dst)
    return True


def _values_equal(a, b, ignore_case: bool = False) -> bool:
    if isinstance(a, str) and not a.strip():
        a = None
    if isinstance(b, str) and not b.strip():
        b = None
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))
    if ignore_case and isinstance(a, str) and isinstance(b, str):
        # Under a case-flip the answer is only required to be stable up to letter-case:
        # text merely carried out of the mutated sheet changes case for correct programs
        # too, and only a change in WHICH value was selected is evidence of a bug.
        return a.lower() == b.lower()
    return a == b


class AgentHarness(SpreadsheetAgent):
    SYSTEM_TEMPLATE = "invariant_probe.txt"

    # Bound the probe so a wide workbook cannot blow up the turn's wall-clock.
    MAX_PROBE_SHEETS = 3
    MAX_PROBE_RUNS = 8
    MAX_REPORTED = 4

    # ── probe machinery ────────────────────────────────────────────────────
    def parse_declaration(self, response: str) -> dict:
        """Pull the INVARIANTS block out of the prose preceding the code fence."""
        head = (response or "").split("```")[0]
        decl: dict = {}
        for m in _DECL_RE.finditer(head):
            key, raw = m.group(1).upper(), m.group(2).strip()
            if key == "DEPENDS":
                names = [p.strip().strip("'\"`") for p in raw.split(",")]
                decl["DEPENDS"] = [n for n in names if n and n.lower() != "none"]
            else:
                low = raw.lower()
                if "same" in low or "unchanged" in low:
                    decl[key] = "same"
                elif "diff" in low or "change" in low:
                    decl[key] = "different"
        return decl

    def _answer_specs(self, task, path: Path):
        """[(resolved_sheet_title, [coords])] for the answer range(s), or None."""
        wb = openpyxl.load_workbook(path)
        try:
            specs = []
            for sheet, rng in parse_answer_position(getattr(task, "answer_position", "")):
                name = sheet or getattr(task, "answer_sheet", "") or None
                ws = wb[name] if (name and name in wb.sheetnames) else wb.active
                specs.append((ws.title, cells_in_range(rng)))
            return specs or None
        finally:
            wb.close()

    def _read_answer(self, path: Path, specs) -> dict:
        wb = openpyxl.load_workbook(path)
        try:
            return {
                (title, coord): (wb[title][coord].value if title in wb.sheetnames else None)
                for title, coords in specs
                for coord in coords
            }
        finally:
            wb.close()

    def _run_variant(self, code, src: Path, run_dir: Path, specs):
        """Run `code` on `src`; return the answer-cell map, or None if it did not run."""
        res = run_code(
            code, src, run_dir, timeout=self.CODE_TIMEOUT,
            expect_output=True, python_exe=self.python_exe,
        )
        if not res.ok or res.output_path is None:
            return None
        return self._read_answer(res.output_path, specs)

    def _diff(self, base: dict, other: dict, kind: str, positional: bool = True) -> list[str]:
        """Describe how the answer moved, under the comparison `kind` makes meaningful."""
        if kind == "ROW_ORDER" and not positional:
            # The answer region sits inside the block whose rows were reversed, so its
            # cells are positionally realigned by construction: a correct program is only
            # required to produce the same answers, not in the same places.
            before = sorted(repr(v) for v in base.values())
            after = sorted(repr(other.get(k)) for k in base)
            if before == after:
                return []
            gained = sorted(set(after) - set(before))[:2]
            return [
                "the set of answer values changed"
                + (f" (now includes {', '.join(gained)})" if gained else "")
            ]
        out = []
        for key, bval in base.items():
            oval = other.get(key)
            if not _values_equal(bval, oval, ignore_case=(kind == "CASE")):
                title, coord = key
                out.append(f"{title}!{coord}: unmodified={bval!r} -> modified={oval!r}")
        return out

    def probe(self, code: str, decl: dict, task, primary_input: Path, probe_dir: Path):
        """Return a list of human-readable contradictions (possibly empty)."""
        specs = self._answer_specs(task, primary_input)
        if not specs:
            return []

        wb = openpyxl.load_workbook(primary_input)
        all_sheets = list(wb.sheetnames)
        wb.close()
        if not all_sheets:
            return []
        sheets = all_sheets[: self.MAX_PROBE_SHEETS]

        ident = probe_dir / "identity.xlsx"
        if not _write_mutation(primary_input, ident, all_sheets[0], "IDENTITY"):
            return []
        # Baseline is the identity round-trip, so openpyxl's own re-save artefacts are
        # differenced out and only the perturbation's effect remains.
        baseline = self._run_variant(code, ident, probe_dir / "identity", specs)
        if baseline is None:
            return []

        declared = {d.strip().lower() for d in decl.get("DEPENDS", [])}
        answer_sheets = {title for title, _ in specs}
        findings, runs = [], 0

        for sheet in sheets:
            is_dep = (not declared) or (sheet.strip().lower() in declared)
            # A sheet the answer supposedly ignores only needs one probe to expose a
            # spurious dependency; a sheet it reads gets the full invariance battery.
            kinds = _INVARIANCE_KEYS if is_dep else _SPURIOUS_KINDS
            for kind in kinds:
                if runs >= self.MAX_PROBE_RUNS or len(findings) >= self.MAX_REPORTED:
                    return findings
                tag = f"{sheet}_{kind}".replace("/", "_").replace(" ", "_")
                mutated = probe_dir / f"{tag}.xlsx"
                if not _write_mutation(primary_input, mutated, sheet, kind):
                    continue
                runs += 1
                got = self._run_variant(code, mutated, probe_dir / tag, specs)

                if got is None:
                    findings.append(
                        f"- {kind} on sheet {sheet!r}: your program CRASHED or wrote no "
                        f"output.xlsx on this input, though it ran on the original. A "
                        f"program that only survives the exact rows you saw is not general."
                    )
                    continue

                positional = sheet not in answer_sheets
                diff = self._diff(baseline, got, kind, positional=positional)
                label = _MUTATION_LABEL[kind]
                evidence = "; ".join(diff[:2])
                if positional or kind != "ROW_ORDER":
                    evidence = f"moved {len(diff)} answer cell(s): {evidence}"

                if not is_dep:
                    if diff:
                        findings.append(
                            f"- You declared DEPENDS without sheet {sheet!r}, but {label} "
                            f"on {sheet!r} {evidence}"
                        )
                    break  # one applicable perturbation settles an undeclared sheet

                predicted = decl.get(kind)
                if predicted == "same" and diff:
                    findings.append(
                        f"- You declared {kind}: same, but {label} on sheet {sheet!r} "
                        f"{evidence}"
                    )
                elif predicted == "different" and not diff:
                    findings.append(
                        f"- You declared {kind}: different, but {label} on sheet {sheet!r} "
                        f"left every answer cell identical -- your program is not reading "
                        f"what you think it is."
                    )
        return findings

    # ── main loop ──────────────────────────────────────────────────────────
    def run_loop(self, task, primary_input: Path, workdir: Path) -> dict:
        messages = [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": self.build_user_prompt(task, primary_input)},
        ]
        solution_code = None      # accepted
        best_runnable = None      # runnable but not yet accepted (never regress past this)
        produced = False
        probed = False
        turns = 0
        code = None

        for turn in range(self.MAX_TURNS):
            turns = turn + 1
            messages = self.summarize_context(messages)
            try:
                response = self.call_llm(messages)
            except Exception:
                # Transient backend failure (e.g. a 502): degrade to the best solution
                # found so far rather than aborting the run.
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
                if not res.ok:
                    messages.append({"role": "user", "content": self.observe(res, "solve")})
                    continue

                best_runnable = code
                if probed or turn == self.MAX_TURNS - 1:
                    solution_code = code
                    produced = True
                    break

                try:
                    findings = self.probe(
                        code, self.parse_declaration(response), task,
                        primary_input, run_dir / "probe",
                    )
                except Exception:
                    findings = []  # probe machinery must never cost a working solution
                probed = True

                if not findings:
                    solution_code = code   # declared behaviour survived perturbation
                    produced = True
                    break
                messages.append({
                    "role": "user",
                    "content": _REPORT_HEADER
                    + "\n".join(findings[: self.MAX_REPORTED])
                    + "\n"
                    + _REPORT_FOOTER,
                })
            else:  # explore
                res = run_code(
                    code, primary_input, run_dir,
                    timeout=self.CODE_TIMEOUT, expect_output=False,
                    python_exe=self.python_exe,
                )
                messages.append({"role": "user", "content": self.observe(res, "explore")})

        if solution_code is None:
            # Prefer a program already proven to run over the last thing the model typed.
            solution_code = best_runnable or code or PASSTHROUGH_SOLUTION
            produced = produced or (best_runnable is not None)
        return {
            "solution_code": solution_code,
            "n_turns": turns,
            "produced": produced,
            "trajectory": messages,
        }
