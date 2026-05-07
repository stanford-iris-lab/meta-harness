from __future__ import annotations

import chz
from typing_extensions import override

from agents.baseline_basic_agent import STRICT_REPLICATION_NOTE
from agents.baseline_basic_agent import BasicAgentSolver as _Base

SMOKE_RUN_NOTE = """
REPRODUCE.SH SMOKE-RUN REMINDER
---
Two short rules, in addition to the replication requirement above:

1. Keep `/home/submission/reproduce.sh` present and runnable from the very
   first commit. Even before any real experiment code exists, the submission
   directory must already contain a `reproduce.sh` (use `set -euo pipefail`
   and `cd "$(dirname "$0")"` at the top). Expand it as your code grows;
   never delete it or move it out of `/home/submission/`. The grader will
   give 0 if `reproduce.sh` is missing from the submission.

2. Before you call end_task, run `bash /home/submission/reproduce.sh` once
   from a clean shell and confirm it exits 0 and writes its claimed result
   artifacts. If it fails or produces nothing, fix it and rerun. A
   reproduce.sh that completes a real (or clearly-labelled smoke) end-to-end
   pipeline beats a longer pipeline that crashes partway through.
"""


@chz.chz
class BasicAgentSolver(_Base):
    """BasicAgent variant that nudges the agent to keep reproduce.sh runnable."""

    extra_instructions: str = f"{STRICT_REPLICATION_NOTE.strip()}\n\n{SMOKE_RUN_NOTE.strip()}\n"

    @override
    def shortname(self) -> str:
        return "mh_smoke_run_basicagent"
