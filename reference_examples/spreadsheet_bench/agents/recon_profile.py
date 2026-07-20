"""ReAct baseline + deterministic workbook reconnaissance.

Parent: ``baseline_react`` (a bare ``SpreadsheetAgent``). ONE new mechanism: the initial
user prompt is augmented with a deterministic, task-agnostic STRUCTURAL PROFILE of the
input workbook, computed by scanning it with openpyxl *before* the model acts.

Motivation. The baseline shows the model only a value-only 20x20 preview. That preview
structurally hides the properties whose absence drives the observed generation errors:

  - the true data extent per sheet (so the model hardcodes ranges from the few visible
    rows and overfits to the debug case),
  - per-column data types,
  - which columns hold FORMULAS in the input -- uncached formula cells read back as blank
    under ``data_only=True`` (which the preview uses), so they are *invisible* in the
    preview and get silently read as None,
  - which columns contain DUPLICATE keys or BLANK cells inside the data extent.

Surfacing these lets the model derive answer ranges and edge-case handling from the data
rather than from the visible rows. The profile is computed deterministically (no extra LLM
call), so this mechanism cannot collude with the generator and cannot be killed by
per-call infra flakiness. Everything else -- system prompt, ``parse_action``, ``observe``,
acceptance (first runnable solution wins), ``MAX_TURNS`` -- is identical to the baseline.
"""

from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

from agent import SpreadsheetAgent


def _cell_kind(v) -> str:
    """Coarse, scorer-aligned value category for one cell."""
    if v is None or v == "":
        return "blank"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (datetime, date, time)):
        return "date"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        s = v.strip().replace(",", "")
        if s.endswith("%"):
            s = s[:-1]
        try:
            float(s)
            return "numstr"  # numeric-looking text; the scorer compares it as a number
        except ValueError:
            return "text"
    return "other"


class AgentHarness(SpreadsheetAgent):
    # Bounded, task-agnostic caps so profiling stays cheap on large workbooks.
    PROFILE_MAX_ROWS = 4000   # rows scanned per sheet for type/dup/blank stats
    PROFILE_MAX_COLS = 40     # columns profiled per sheet
    PROFILE_SAMPLES = 5       # distinct sample values shown per column

    def build_user_prompt(self, task, primary_input: Path) -> str:
        base = super().build_user_prompt(task, primary_input)
        profile = self._workbook_profile(Path(primary_input))
        if not profile:
            return base  # any failure degrades cleanly to the baseline prompt
        return (
            f"{base}\n"
            "### workbook_profile\n"
            "Deterministic scan of input.xlsx (per-column types, data extents, blank and "
            "duplicate cells, and formula cells). Derive answer ranges and edge-case "
            "handling from the data described here, not from the few preview rows above; "
            "note that uncached formula cells appear blank in that preview.\n"
            f"{profile}\n"
        )

    # -- deterministic reconnaissance --------------------------------------
    def _workbook_profile(self, path: Path) -> str:
        try:
            wb_v = load_workbook(path, data_only=True)
        except Exception:
            return ""
        try:
            wb_f = load_workbook(path, data_only=False)  # exposes formula strings
        except Exception:
            wb_f = None
        try:
            blocks = [self._sheet_profile(ws, wb_f) for ws in wb_v.worksheets]
        except Exception:
            return ""
        return "\n".join(b for b in blocks if b)

    def _sheet_profile(self, ws, wb_f) -> str:
        max_row = ws.max_row or 0
        max_col = ws.max_column or 0
        n_cols = min(max_col, self.PROFILE_MAX_COLS)
        scan_rows = min(max_row, self.PROFILE_MAX_ROWS)
        if n_cols == 0 or scan_rows == 0:
            return f"- Sheet {ws.title!r}: {max_row} rows x {max_col} cols (empty)"

        # Per-column accumulators (index 0 == column A within the scanned window).
        kinds: list[dict] = [dict() for _ in range(n_cols)]
        seen: list[set] = [set() for _ in range(n_cols)]
        samples: list[list] = [[] for _ in range(n_cols)]
        n_blank = [0] * n_cols
        n_formula = [0] * n_cols
        dup = [False] * n_cols
        first_nb: list[int | None] = [None] * n_cols
        last_nb: list[int | None] = [None] * n_cols

        for r, row in enumerate(
            ws.iter_rows(min_row=1, max_row=scan_rows, max_col=n_cols, values_only=True),
            start=1,
        ):
            for i, v in enumerate(row):
                k = _cell_kind(v)
                kinds[i][k] = kinds[i].get(k, 0) + 1
                if k == "blank":
                    n_blank[i] += 1
                    continue
                if first_nb[i] is None:
                    first_nb[i] = r
                last_nb[i] = r
                key = str(v)
                if key in seen[i]:
                    dup[i] = True
                elif len(samples[i]) < self.PROFILE_SAMPLES:
                    samples[i].append(key)
                seen[i].add(key)

        # Formula cells are only visible without data_only; scan the same window.
        if wb_f is not None and ws.title in wb_f.sheetnames:
            ws_f = wb_f[ws.title]
            for row in ws_f.iter_rows(
                min_row=1, max_row=scan_rows, max_col=n_cols, values_only=True
            ):
                for i, fv in enumerate(row):
                    if isinstance(fv, str) and fv.startswith("="):
                        n_formula[i] += 1

        lines = [f"- Sheet {ws.title!r}: {max_row} rows x {max_col} cols"]
        if scan_rows < max_row:
            lines.append(f"  (column stats sampled over the first {scan_rows} rows)")
        if max_col > n_cols:
            lines.append(f"  (profiling first {n_cols} of {max_col} columns)")

        for i in range(n_cols):
            if first_nb[i] is None:
                continue  # fully blank column within the scan window
            typed = {k: n for k, n in kinds[i].items() if k != "blank"}
            types = ", ".join(
                f"{k}x{n}" for k, n in sorted(typed.items(), key=lambda kv: -kv[1])[:3]
            )
            tags = [f"types {types}", f"rows {first_nb[i]}-{last_nb[i]}"]
            if n_blank[i]:
                tags.append(f"{n_blank[i]} blank")
            if dup[i]:
                tags.append("has duplicates")
            if n_formula[i]:
                tags.append(f"{n_formula[i]} formula cells")
            col = get_column_letter(i + 1)
            sample_str = ", ".join(
                s if len(s) <= 24 else s[:21] + "..." for s in samples[i]
            )
            lines.append(f"  col {col}: {'; '.join(tags)} | e.g. {sample_str}")
        return "\n".join(lines)
