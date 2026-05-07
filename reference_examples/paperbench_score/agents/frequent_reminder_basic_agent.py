from __future__ import annotations

import chz
from typing_extensions import override

from agents.baseline_basic_agent import BasicAgentSolver as _Base


@chz.chz
class BasicAgentSolver(_Base):
    """Baseline with a more frequent periodic state/time/commit reminder."""

    reminder_freq: int = 3

    @override
    def shortname(self) -> str:
        return "mh_frequent_reminder_basicagent"
