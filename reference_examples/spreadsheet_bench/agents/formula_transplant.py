"""Surface the input's own Excel formulas at the answer region (the "formula transplant").

Parent: ``baseline_react`` (a bare ``SpreadsheetAgent``). The sheet preview the model is shown
is built with ``data_only=True`` (``sheet_utils.preview_spreadsheet``), which renders every
input formula cell as its *cached* value — and for a workbook that was never opened in a
spreadsheet app that cached value is absent, so a formula-backed answer column shows up
**blank** in the preview. The model then cannot see the transformation the sheet actually
encodes, so it *reverse-engineers* the formula from scratch and gets the semantics wrong (e.g. a
nested-IF / parity rule, or a hand-rolled Excel financial function that degenerates on an edge
input). This wrong-formula family is the single largest source of failures on the dev set, and
no post-hoc check can catch it because the answer key is hidden from the loop — which is exactly
why every verification/gate candidate so far has tied or regressed. The leverage is in *what the
model sees before it generates*, not in checking what it produced.

ONE new mechanism is added here: a **deterministic** reconnaissance step in ``build_user_prompt``.
It reads the INPUT with ``data_only=False`` and, for the answer region only, extracts the *actual
Excel formula text* living in (or in the same column as, within a bounded window of) the answer
cells, then appends a labelled section instructing the model to replicate that exact logic and
write LITERAL values. It surfaces information that is genuinely present in the input but
structurally hidden by the value preview — never the answer key, never a hardcoded
column/value/rule; the formula is task content the model could itself have read with
``data_only=False`` in an explore turn.

When the answer region contains no formulas (the common case, including both currently-passing
guardrail tasks whose answers are filtered/aggregated data, not a template formula) nothing is
appended and the user prompt is byte-identical to the baseline: **no extra LLM call**, no change
to ``run_loop`` / ``parse_action`` / ``observe`` / ``MAX_TURNS``, so it cannot regress a
formula-free task and adds no per-call 502 exposure.

Distinct from the prior candidates: ``recon_profile`` injected only a *count* of formula cells
across the whole workbook (a passive stat the model raced past — it regressed a task); here we
transplant the *formula string itself*, scoped to the answer region, replacing error-prone
guessing with the ground-truth transformation. ``answer_type_gate`` / ``readback_verify`` acted
*after* a solve on the produced output; this acts *before* generation on the input.
"""

from __future__ import annotations

from pathlib import Path

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter, range_boundaries

from agent import SpreadsheetAgent
from sheet_utils import parse_answer_position


def _resolve_ws(wb, sheet_name):
    if sheet_name is None:
        return wb.active
    if sheet_name in wb.sheetnames:
        return wb[sheet_name]
    return None


class AgentHarness(SpreadsheetAgent):
    # How far above/below the answer rows to look, within the answer column(s), for a
    # representative template formula when the answer cells themselves hold none (many tasks
    # keep the formula in template rows the answer region extends).
    FORMULA_SCAN_PAD = 200
    MAX_FORMULAS = 8          # cap distinct formulas surfaced
    MAX_FORMULA_LEN = 240     # cap each formula string shown

    def _answer_region_formulas(self, task, input_path) -> list[tuple[str, str]]:
        """Return [(location, formula_text)] found in the INPUT at/near the answer region.

        Reads the input with ``data_only=False`` so ``=...`` strings are visible. Scans the
        answer cells themselves first, then the surrounding rows of the same answer column(s)
        within a bounded window. Deduped by a digit-stripped skeleton so a uniform column (the
        same formula shifted per row) contributes a single representative. Any failure degrades
        to an empty list → the baseline prompt.
        """
        try:
            wb = load_workbook(input_path, data_only=False)
        except Exception:
            return []
        default_sheet = getattr(task, "answer_sheet", "") or None
        found: list[tuple[str, str]] = []
        seen: set[str] = set()
        try:
            for sheet, rng in parse_answer_position(task.answer_position):
                ws = _resolve_ws(wb, sheet or default_sheet)
                if ws is None:
                    continue
                try:
                    min_c, min_r, max_c, max_r = range_boundaries(rng)
                except Exception:
                    continue
                if None in (min_c, min_r, max_c, max_r):
                    continue
                lo = max(1, min_r - self.FORMULA_SCAN_PAD)
                hi = min(ws.max_row or max_r, max_r + self.FORMULA_SCAN_PAD)
                for c in range(min_c, max_c + 1):
                    # answer cells first, then the surrounding template rows of this column
                    ordered_rows = list(range(min_r, max_r + 1))
                    ordered_rows += [
                        r for r in range(lo, hi + 1) if not (min_r <= r <= max_r)
                    ]
                    for r in ordered_rows:
                        val = ws.cell(row=r, column=c).value
                        if not (isinstance(val, str) and val.lstrip().startswith("=")):
                            continue
                        text = val.strip()
                        skel = " ".join(
                            "".join(ch for ch in text.upper() if not ch.isdigit()).split()
                        )
                        if skel in seen:
                            continue
                        seen.add(skel)
                        if len(text) > self.MAX_FORMULA_LEN:
                            text = text[: self.MAX_FORMULA_LEN - 3] + "..."
                        found.append((f"{get_column_letter(c)}{r}", text))
                        if len(found) >= self.MAX_FORMULAS:
                            return found
        except Exception:
            return found
        return found

    def build_user_prompt(self, task, primary_input: Path) -> str:
        base = super().build_user_prompt(task, primary_input)
        formulas = self._answer_region_formulas(task, primary_input)
        if not formulas:
            return base  # byte-identical to the baseline → cannot regress formula-free tasks
        listing = "\n".join(f"  - {loc}: {text}" for loc, text in formulas)
        return base + (
            "\n### formulas already present in the input at the answer region\n"
            "The preview above shows cached values and HIDES formulas — in a workbook that was "
            "never opened in a spreadsheet app, formula cells render blank. The input actually "
            "contains the Excel formula(s) below in (or in the same column as) the answer cells; "
            "they encode the exact transformation the answer requires. Replicate their logic in "
            "Python and write the resulting LITERAL values into the answer cells (never a "
            "formula string). Do not reverse-engineer the rule from the visible values:\n"
            f"{listing}\n"
        )
