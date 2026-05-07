# PaperBench Score Meta-Harness

This example hill-climbs a PaperBench BasicAgent variant on a single PaperBench paper. Each candidate is a Python module in `agents/` that exports `BasicAgentSolver`.

## Smoke Checks

```bash
python benchmark.py --agent baseline_basic_agent --run-name smoke
python meta_harness.py --iterations 1 --run-name paperbench-score-smoke --skip-baseline
```

## Objective

The scalar objective is the real PaperBench overall score from `grade.json`. The proposer feedback includes the compact category breakdown:

```text
Overall: <score>
Code Development: <got>/<weight>
Code Execution: <got>/<weight>
Result Analysis: <got>/<weight>
```

The Modal evaluator uses the same bare-runtime PaperBench path as `/Users/emmett/Projects/papers/run_paperbench_modal_bare.py`, but parameterizes the agent file so Meta-Harness can evaluate generated candidates.
