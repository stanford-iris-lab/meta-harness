"""OJ-style scoring for SpreadsheetBench.

A candidate produces ONE solution program (Python that reads `input.xlsx` and writes
`output.xlsx`). We run that same program against every test-case input and compare the
produced output to the test case's answer at the instruction's answer positions.

`metric="hard"` (default, the paper's headline metric): the instruction passes iff ALL
test cases pass. `metric="soft"`: fraction of test cases that pass.

Eval backend:
  - "value"       : compare cached/literal values via openpyxl(data_only=True). No
                    LibreOffice needed as long as the solution writes literal values.
  - "libreoffice" : recalc formulas first (stub; see recalc_with_libreoffice).
"""

from __future__ import annotations

from pathlib import Path

from executor import run_code
from sheet_utils import compare_workbooks


def recalc_with_libreoffice(xlsx_path: Path) -> None:
    """Recalculate formulas in-place so openpyxl(data_only=True) can read values.

    Not enabled by default. Vulcan has no system LibreOffice, but `apptainer` is
    available as a module, so this can shell out to soffice inside a container, e.g.:

        apptainer exec libreoffice.sif soffice --headless --calc \\
            --convert-to xlsx:"Calc MS Excel 2007 XML" --outdir <tmp> <xlsx_path>

    then copy the converted file back over xlsx_path. Implement when a formula-output
    task is found that the value backend misjudges (see WALKTHROUGH / plan step 4).
    """
    raise NotImplementedError(
        "libreoffice eval backend not wired up; use backend='value' (agent writes "
        "literal values) or add an Apptainer soffice call here."
    )


def score_instruction(
    solution_code: str,
    task,
    workdir: str | Path,
    timeout: int = 60,
    tol: float = 1e-6,
    metric: str = "hard",
    backend: str = "value",
    python_exe: str | None = None,
) -> dict:
    """Run `solution_code` on every test case of `task` and score OJ-style.

    Returns {passed: bool, score: float, per_case: [...], n_cases: int}.
    """
    workdir = Path(workdir)
    per_case = []
    for case in task.test_cases:
        case_dir = workdir / f"case_{case.index}"
        res = run_code(
            solution_code,
            case.input_path,
            case_dir,
            timeout=timeout,
            expect_output=True,
            python_exe=python_exe,
        )
        entry = {"index": case.index, "ok": False, "detail": ""}
        if not res.ok:
            entry["detail"] = (
                f"execution failed (rc={res.returncode}, timeout={res.timed_out}): "
                + (res.stderr or res.stdout)[-600:]
            )
            per_case.append(entry)
            continue

        if backend == "libreoffice":
            recalc_with_libreoffice(res.output_path)

        ok, detail = compare_workbooks(
            case.answer_path, res.output_path, task.answer_position, tol,
            default_sheet=getattr(task, "answer_sheet", "") or None,
        )
        entry["ok"] = ok
        entry["detail"] = detail
        per_case.append(entry)

    n = len(per_case)
    n_pass = sum(1 for e in per_case if e["ok"])
    if metric == "soft":
        score = n_pass / n if n else 0.0
        passed = score >= 1.0
    else:  # hard
        passed = n > 0 and n_pass == n
        score = 1.0 if passed else 0.0
    return {
        "passed": passed,
        "score": score,
        "n_cases": n,
        "n_pass": n_pass,
        "per_case": per_case,
    }
