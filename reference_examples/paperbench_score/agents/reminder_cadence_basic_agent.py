from __future__ import annotations

import chz
from typing_extensions import override

from agents.baseline_basic_agent import BasicAgentSolver as _Base


@chz.chz
class BasicAgentSolver(_Base):
    """Baseline with a less frequent periodic reminder to reduce context noise."""

    reminder_freq: int = 10

    @override
    def shortname(self) -> str:
        return "mh_reminder_cadence_basicagent"
