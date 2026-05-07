from __future__ import annotations

import chz
from typing_extensions import override

from agents.baseline_basic_agent import STRICT_REPLICATION_NOTE
from agents.baseline_basic_agent import BasicAgentSolver as _Base

METRICS_BREADTH_NOTE = """
RESULT METRICS COVERAGE
---
Result Analysis credit depends on a final metrics file written by
reproduce.sh (e.g. `results/metrics.csv`) with one row per
(task, method, simulation_budget, seed) combination. The judge inspects
per-task rows across the FULL benchmark task suite the paper evaluates,
so missing tasks zero out their analysis criteria.

When building reproduce.sh:
- Iterate over EVERY task the paper benchmarks, not a convenient subset.
  If runtime is tight, reduce rounds/seeds/budgets uniformly rather than
  dropping whole tasks; a row with a noisy real C2ST is gradable, a
  missing row is not.
- Append rows incrementally as each (task, method) evaluation finishes
  so the CSV has partial coverage even if reproduce.sh is interrupted
  before the full sweep completes.
- Every row must come from a real experiment run on real synthetic data
  (consistent with the strict replication requirement above). Do not
  fabricate, copy from the paper, or hand-edit metric values into the CSV.

Keep reproduce.sh itself committed at /home/submission/reproduce.sh; this
note is about what reproduce.sh produces, not a substitute for it.
"""


@chz.chz
class BasicAgentSolver(_Base):
    """Baseline plus a narrow nudge for broad task coverage in the recorded metrics."""

    extra_instructions: str = f"{STRICT_REPLICATION_NOTE.strip()}\n\n{METRICS_BREADTH_NOTE.strip()}\n"

    @override
    def shortname(self) -> str:
        return "mh_metrics_breadth_basicagent"
