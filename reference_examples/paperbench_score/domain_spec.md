# Domain Spec: PaperBench Single-Paper Score Optimization

## Domain Summary

The task is to improve a PaperBench replication agent on one paper at a time. One evaluation unit is a full PaperBench rollout for `sequential-neural-score-estimation` using the current Modal/BareRuntime runner, including agent authoring, `reproduce.sh`, remote judging, and final `grade.json` scoring.

The fixed base model is the PaperBench BasicAgent completer model configured by the Modal runner (`gpt-5.5`). The searchable harness is the Python module that exports `BasicAgentSolver`; initial search should focus on prompt policy, time allocation, planning structure, and anti-fakery/provenance guidance, not on changing the judge, paper, model, Modal runtime, or PaperBench scoring.

## Harness and Search Plan

Every candidate is a Python file under `agents/` that exports `BasicAgentSolver`. The benchmark copies that file into the Modal image as `/work/papers/basic_agent.py`, where PaperBench imports `basic_agent:BasicAgentSolver`.

Start from `agents/baseline_basic_agent.py`. Candidate files should be small, import-checkable, and should preserve the PaperBench solver interface. The proposer writes `pending_eval.json` with candidate names and hypotheses, then `benchmark.py` evaluates each candidate.

## Evaluation Plan

The search-set evaluation is the single PaperBench paper `sequential-neural-score-estimation` with `600s` agent timeout and `600s` reproduction timeout. The primary metric is `grade.json["score"]`.

Feedback to the proposer should include:

- Overall score.
- Category aggregate scores for Code Development, Code Execution, and Result Analysis.
- Whether submission, reproduction metadata, and judge output existed.
- Paths to the Modal/eval logs and any parsed result JSON.

This is noisy and expensive: one candidate can take 10-25 minutes plus judge time. There is leakage risk because the optimizer sees the same single-paper feedback repeatedly, so this should be treated as a one-datapoint hill climb rather than a generalization claim.

## Experience and Logging

Store each candidate evaluation under `logs/<run-name>/<candidate>/`, including `modal.log`, `result.json`, raw parsed score data, proposer session logs, `frontier_val.json`, and `evolution_summary.jsonl`.

The highest-signal feedback is the compact score block:

```text
Overall: 0.4358
Code Development: 34/81
Code Execution: 6/7
Result Analysis: 13/20
```

## Open Questions and Unknowns

Held-out evaluation is unknown. The first pass intentionally optimizes a single live datapoint, so improvements should later be checked on other PaperBench papers before claiming robustness.
