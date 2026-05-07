from __future__ import annotations

import chz
from typing_extensions import override

from agents.baseline_basic_agent import BasicAgentSolver as _Base
from agents.baseline_basic_agent import STRICT_REPLICATION_NOTE

REPRO_HYGIENE_NOTE = """
REPRODUCTION HYGIENE
---
Keep the baseline strategy: build as much real replication code as possible. In
addition, keep `/home/submission` clean and runnable:

- Commit all source code and `reproduce.sh`; never submit a placeholder script.
- Put virtual environments, package caches, and bulky temporary downloads outside
  `/home/submission` (for example under `/tmp`), so the final tarball contains
  code, configs, manifests, and generated result artifacts, not dependency caches.
- Before final submit, run `git status --short` and ensure important source files
  are tracked. It is okay for generated outputs/logs to be untracked if
  `reproduce.sh` regenerates them.
"""


@chz.chz
class BasicAgentSolver(_Base):
    """Baseline plus a narrow nudge for clean, committed reproduction artifacts."""

    extra_instructions: str = f"{STRICT_REPLICATION_NOTE.strip()}\n\n{REPRO_HYGIENE_NOTE.strip()}\n"

    @override
    def shortname(self) -> str:
        return "mh_clean_repro_basicagent"
