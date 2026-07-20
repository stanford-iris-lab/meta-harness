"""ReAct + execution-feedback baseline. The primary evolution parent.

Uses the full multi-round loop in agent.SpreadsheetAgent: explore the input, submit a
solution program, get execution feedback, refine, accept the first runnable solution.
"""

from agent import SpreadsheetAgent


class AgentHarness(SpreadsheetAgent):
    pass
