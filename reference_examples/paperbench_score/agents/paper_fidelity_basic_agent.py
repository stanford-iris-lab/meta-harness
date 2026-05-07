from __future__ import annotations

import chz
from typing_extensions import override

from agents.baseline_basic_agent import STRICT_REPLICATION_NOTE
from agents.baseline_basic_agent import BasicAgentSolver as _Base

PAPER_FIDELITY_NOTE = """
PAPER SPECIFICATION FIDELITY
---
Many graded checks in this evaluation key on specific values and method
choices the paper explicitly fixes: numerical solver choice, sample
counts, schedule constants, embedding/network widths, number of
training rounds, and the named methods that are compared. When the
paper (including its appendix tables / hyperparameter section) states
an exact value or a named procedure, implement that value or procedure
literally rather than silently substituting a generic ML default
because it is more convenient. Examples of the shape of such specs
(not literal targets): "solver X is used, not Y", "exactly N samples
are drawn, not a smaller default", "constant c is set to v for the
listed tasks", "the loss is the Monte Carlo estimate of <formula>".

This is a final correctness pass, not a reason to shrink scope. The
central deliverable is unchanged: a runnable reproduce.sh committed at
/home/submission/reproduce.sh that produces the result artifacts from
real experiment runs. Get that end-to-end pipeline working first, and
only then audit it against the paper's stated constants and named
methods before you stop.
"""


@chz.chz
class BasicAgentSolver(_Base):
    """Baseline plus a narrow nudge to honor paper-stated constants and named methods."""

    extra_instructions: str = f"{STRICT_REPLICATION_NOTE.strip()}\n\n{PAPER_FIDELITY_NOTE.strip()}\n"

    @override
    def shortname(self) -> str:
        return "mh_paper_fidelity_basicagent"
