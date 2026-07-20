"""Align the solution program's input-read path with the exploration read path.

Parent: ``baseline_react`` (a bare ``SpreadsheetAgent``). ONE mechanism is changed: the
demonstrated ``ACTION: solve`` code skeleton, via a new system template
(``prompt-templates/value_read.txt``; ``SYSTEM_TEMPLATE`` is the sanctioned override point
already used by ``baseline_single`` / ``generality_contract`` — no shared-module edits).

Why this is the lever. In the base ``react.txt`` the *explore* skeleton reads with
``data_only=True`` (so the model inspects correct **cached** values), but the *solve*
skeleton reads with a plain ``load_workbook("input.xlsx")`` (``data_only=False``) and uses
that same handle to write and save. So the generated program re-reads any formula-backed
input column as its raw formula **text** (``"=..."``), which the model's own numeric guards
coerce to ``0``/``None`` — a silent wrong value even when the transformation logic is
correct. This is the single most-provable dev-set failure: e.g. an "answer = other-column
when <condition>" task where that other column is itself a formula column — the model's
branch logic is right, but it reads the formula string and writes ``0`` instead of the
cached value ``2`` it had already printed while exploring.

The fix changes the generated program's read/write *architecture* from one default-mode
workbook to a dual-workbook split: read every input value from a ``data_only=True`` book
(the same computed values seen during exploration) and write/save the normal book (so all
other cells' formulas and formatting survive the save). The template also states the
underlying rule in one sentence so the model applies it beyond the skeleton.

Properties:
- ONE mechanism (generated-program input-read path); nothing else touched — same
  ``run_loop`` / ``parse_action`` / ``observe`` / ``MAX_TURNS`` as the parent.
- ZERO extra LLM calls, no gate, no added turn → no increase in 502-flake exposure
  (respects the prior "prefer zero-extra-call mechanisms" lesson).
- No task-specific hint: a general Excel-correctness pattern + a data-derived rule; no
  hardcoded columns/rows/values, and it never touches the answer key. Targets GENERATION
  quality (the program the model writes), not post-hoc verification (which colludes).
- Cannot regress the two current passers: their answers read literal-valued columns, so
  ``data_only=True`` returns byte-identical values. The template diff vs ``react.txt`` is
  scoped to the solve skeleton + one rationale section to minimise prompt perturbation.

Distinct from prior candidates: ``formula_transplant`` surfaced the answer region's formula
*text* and told the model to replicate the logic — the model did, then still read the source
column with ``data_only=False`` and wrote ``0`` (it fixed logic, not the read path). This
candidate fixes the read path itself, which ``formula_transplant`` left broken.
"""

from agent import SpreadsheetAgent


class AgentHarness(SpreadsheetAgent):
    SYSTEM_TEMPLATE = "value_read.txt"
