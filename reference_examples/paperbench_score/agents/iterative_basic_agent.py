from __future__ import annotations

import chz
from typing_extensions import override

from agents.baseline_basic_agent import BasicAgentSolver as _Base


@chz.chz
class BasicAgentSolver(_Base):
    """Baseline instructions with PaperBench's iterative BasicAgent tool profile."""

    iterative_agent: bool = True

    @override
    def shortname(self) -> str:
        return "mh_iterative_basicagent"
