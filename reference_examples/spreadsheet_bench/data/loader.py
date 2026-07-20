"""SpreadsheetBench dataset loader.

Parses a SpreadsheetBench `dataset.json` and resolves the per-instruction test-case
spreadsheet files. Layout (after `fetch_data.sh`):

    data/<source>/dataset.json
    data/<source>/spreadsheet/<id>/{1,2,3}_{id}_input.xlsx
    data/<source>/spreadsheet/<id>/{1,2,3}_{id}_answer.xlsx

Each instruction JSON object has: id, instruction, instruction_type
("Cell-Level Manipulation" | "Sheet-Level Manipulation"), answer_position, and
optionally spreadsheet_path (a folder path). We resolve the spreadsheet folder from
that field when present, else fall back to `<dataset dir>/spreadsheet/<id>`.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent


@dataclass
class TestCase:
    index: int  # 1-based No. prefix
    input_path: Path
    answer_path: Path


@dataclass
class Task:
    id: str
    instruction: str
    instruction_type: str  # "Cell-Level Manipulation" | "Sheet-Level Manipulation"
    answer_position: str
    spreadsheet_dir: Path
    test_cases: list[TestCase] = field(default_factory=list)
    answer_sheet: str = ""  # default sheet for positions with no "Sheet!" qualifier (verified_400)

    @property
    def is_cell_level(self) -> bool:
        return "cell" in self.instruction_type.lower()

    @property
    def primary_input(self) -> Path:
        """Input the agent debugs against (first test case)."""
        return self.test_cases[0].input_path


def _find_index_file(source: str) -> Path:
    """Locate the instruction index (dataset.json or *.jsonl), tolerating one nesting."""
    base = DATA_ROOT / source
    patterns = ["dataset.json", "*.jsonl", "*/dataset.json", "*/*.jsonl"]
    for pat in patterns:
        matches = sorted(base.glob(pat)) if "*" in pat else [base / pat]
        for c in matches:
            if c.exists():
                return c
    raise FileNotFoundError(
        f"No dataset.json or *.jsonl found under {base}. Did you run data/fetch_data.sh?"
    )


def _read_index(path: Path) -> list[dict]:
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    raw = json.loads(text)
    if isinstance(raw, dict):
        return raw.get("data", list(raw.values()))
    return raw


def _resolve_spreadsheet_dir(data_dir: Path, task_id: str, spreadsheet_path) -> Path:
    """Resolve the folder holding a task's input/answer xlsx files."""
    if spreadsheet_path:
        p = Path(spreadsheet_path)
        # Try as given, relative to the dataset dir, and by basename under spreadsheet/.
        for cand in (p, data_dir / p, data_dir / "spreadsheet" / p.name):
            if cand.is_dir():
                return cand
    default = data_dir / "spreadsheet" / task_id
    return default


# Input/answer naming conventions across SpreadsheetBench releases:
#   sample_data_200 / 912 : {n}_{id}_input.xlsx  + {n}_{id}_answer.xlsx
#   verified_400          : {n}_{id}_init.xlsx   + {n}_{id}_golden.xlsx
_NAMING = [("_input.xlsx", "_answer.xlsx"), ("_init.xlsx", "_golden.xlsx")]


def _collect_test_cases(folder: Path, task_id: str) -> list[TestCase]:
    for in_suf, ans_suf in _NAMING:
        cases: list[TestCase] = []
        for inp in sorted(folder.glob(f"*{in_suf}")):
            m = re.match(r"(\d+)_", inp.name)
            idx = int(m.group(1)) if m else len(cases) + 1
            answer = folder / inp.name.replace(in_suf, ans_suf)
            if answer.exists():
                cases.append(TestCase(index=idx, input_path=inp, answer_path=answer))
        if cases:
            cases.sort(key=lambda c: c.index)
            return cases
    return []


def load_tasks(source: str, require_files: bool = True) -> list[Task]:
    """Load all instructions for a dataset source, resolving test-case files."""
    ds_json = _find_index_file(source)
    data_dir = ds_json.parent
    raw = _read_index(ds_json)

    tasks: list[Task] = []
    for obj in raw:
        tid = str(obj["id"])
        folder = _resolve_spreadsheet_dir(data_dir, tid, obj.get("spreadsheet_path"))
        cases = _collect_test_cases(folder, tid) if folder.is_dir() else []
        if require_files and not cases:
            continue
        tasks.append(
            Task(
                id=tid,
                instruction=obj["instruction"],
                instruction_type=obj.get("instruction_type", ""),
                answer_position=obj.get("answer_position", ""),
                spreadsheet_dir=folder,
                test_cases=cases,
                answer_sheet=str(obj.get("answer_sheet", "") or ""),
            )
        )
    return tasks


def select_dev_subset(tasks: list[Task], dev_size: int, seed: int) -> list[Task]:
    """Deterministically pick `dev_size` tasks. dev_size<=0 or >= len returns all."""
    ordered = sorted(tasks, key=lambda t: t.id)
    if dev_size is None or dev_size <= 0 or dev_size >= len(ordered):
        return ordered
    rng = random.Random(seed)
    picked = rng.sample(ordered, dev_size)
    picked.sort(key=lambda t: t.id)
    return picked


def get_task(source: str, task_id: str) -> Task:
    for t in load_tasks(source):
        if t.id == task_id:
            return t
    raise KeyError(f"task {task_id} not found in source {source}")


def _main() -> None:
    ap = argparse.ArgumentParser(description="SpreadsheetBench loader utilities")
    ap.add_argument("--source", default="sample_data_200")
    ap.add_argument("--list", action="store_true", help="list tasks + test-case counts")
    ap.add_argument("--dev-size", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tasks = load_tasks(args.source)
    if args.dev_size:
        tasks = select_dev_subset(tasks, args.dev_size, args.seed)
    print(f"{len(tasks)} tasks from source={args.source}")
    if args.list:
        for t in tasks:
            print(
                f"  {t.id}  [{t.instruction_type}]  n_cases={len(t.test_cases)}  "
                f"pos={t.answer_position!r}  dir={t.spreadsheet_dir}"
            )


if __name__ == "__main__":
    _main()
