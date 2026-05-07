# PaperBench Score Meta-Harness Proposer

You are proposing candidate PaperBench BasicAgent harness variants.

## Objective

Maximize the real PaperBench overall score for `sequential-neural-score-estimation`. The evaluator runs the candidate through Modal/BareRuntime, executes `reproduce.sh`, and grades with PaperBench. Do not optimize for static rubric appearance; optimize for faithful, executable paper replication.

## Candidate Interface

Create one or more Python files in `agents/`. Each candidate file must export `BasicAgentSolver`.

The safest pattern is:

```python
from agents.baseline_basic_agent import BasicAgentSolver as _Base

class BasicAgentSolver(_Base):
    ...
```

Candidates may change prompt guidance, planning policy, time allocation instructions, or other solver subclass behavior. They must not change the judge, paper split, PaperBench score computation, Modal runtime, or evaluator configuration.

## Required Output

Write `logs/<run-name>/pending_eval.json` at the path given in the task prompt.

Use this shape:

```json
{
  "candidates": [
    {
      "name": "candidate_file_stem",
      "hypothesis": "Why this should improve PaperBench score",
      "axis": "prompting|planning|provenance|execution|other"
    }
  ]
}
```

## Guidance

- Preserve artifact provenance: final CSVs, plots, metrics, and reports must be produced by `reproduce.sh`, not pre-baked.
- Help the agent prioritize a runnable end-to-end pipeline before broad paper coverage.
- Prefer small, auditable changes over large rewrites.
- Use prior score breakdowns and logs to target the weakest categories.
- Avoid instructions that encourage fake results, toy proxies presented as full experiments, or judge gaming.
