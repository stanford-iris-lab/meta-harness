from __future__ import annotations

import chz
from typing_extensions import override

from agents.baseline_basic_agent import STRICT_REPLICATION_NOTE
from agents.baseline_basic_agent import BasicAgentSolver as _Base

EXECUTION_GATE_NOTE = """
EXECUTION GATE — DO NOT FINISH UNTIL reproduce.sh RUNS CLEAN
---
The grading reproduction step will execute `reproduce.sh` from scratch in a fresh
environment. Code that imports cleanly is NOT enough. Before you stop:

1. Run `bash reproduce.sh` end-to-end at least once from the submission root and
   confirm exit code 0. If it fails, fix the cause and rerun. Do not stop on a
   failed reproduce.sh.
2. After it succeeds, list the result artifacts it produced (CSVs, plots, JSON
   metrics, tables) and confirm each one is on disk and non-empty. Cross-check
   that every paper figure/table you claim to reproduce has a corresponding
   artifact under the submission directory.
3. Common, expensive failure modes to actively defend against:
   - argparse mismatches: every `args.<name>` access must correspond to an
     `add_argument("--<name>")` (or `dest=`) in the SAME parser. Grep for
     `args\\.` and confirm. A single typo (e.g. `args.method` vs `args.methods`)
     will zero out Code Execution and Result Analysis.
   - relative paths assumed from a working directory that reproduce.sh does not
     `cd` into.
   - missing data downloads, missing `mkdir -p` for output directories,
     missing `set -e` causing silent partial failures.
   - python module imports that work in your interactive shell but fail under
     `bash reproduce.sh` because PYTHONPATH or cwd differs.
4. If the full experiment cannot finish inside the runtime budget, reproduce.sh
   must still complete a labelled smoke run end-to-end (writing real, smaller
   artifacts) and the full-scale path must be the documented default invocation.
   Never leave reproduce.sh in a state where it crashes.

Budget guidance: spend the FIRST third of your time getting an end-to-end
reproduce.sh skeleton that runs (even if it only trains for a few steps) before
expanding scope. A working narrow pipeline beats a broad pipeline that crashes.
The judge cannot give Code Execution or Result Analysis credit for code that
never ran.
"""


@chz.chz
class BasicAgentSolver(_Base):
    """BasicAgent variant with a hard execution gate around reproduce.sh."""

    extra_instructions: str = f"{STRICT_REPLICATION_NOTE.strip()}\n\n{EXECUTION_GATE_NOTE.strip()}\n"

    @override
    def shortname(self) -> str:
        return "mh_exec_gate_basicagent"
