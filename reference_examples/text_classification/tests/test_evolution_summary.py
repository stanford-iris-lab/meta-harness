from __future__ import annotations

import argparse
import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import meta_harness


class EvolutionSummaryTests(unittest.TestCase):
    def test_deltas_use_pre_iteration_frontier(self) -> None:
        scores = {"winner": 75.0, "improved": 70.0, "tied": 60.0, "regressed": 55.0}
        deltas = {"winner": 15.0, "improved": 10.0, "tied": 0.0, "regressed": -5.0}
        candidates = [{"name": name} for name in scores]
        config = {
            "datasets": ["unit-dataset"],
            "models": [{"model": "provider/unit-model"}],
            "memory_systems": {"baselines": []},
        }
        results = {
            ("unit-model", "unit-dataset", name): {"accuracy": score / 100}
            for name, score in scores.items()
        }
        args = argparse.Namespace(
            model="provider/unit-model",
            run_name="test-run",
            fresh=False,
            skip_baseline=True,
            iterations=1,
            propose_timeout=1,
            test=False,
        )

        for valid in (True, False):
            with self.subTest(valid=valid), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run_dir = root / "logs" / "test-run"
                run_dir.mkdir(parents=True)
                frontier = run_dir / "frontier_val.json"
                frontier.write_text(
                    json.dumps({"_pareto": [{"val_accuracy": 60.0}]})
                )

                def propose(*_args, **_kwargs):
                    meta_harness.PENDING_EVAL.write_text(
                        json.dumps({"candidates": candidates})
                    )
                    return True

                def benchmark(command):
                    if "--frontier" in command:
                        frontier.write_text(
                            json.dumps({"_pareto": [{"val_accuracy": 75.0}]})
                        )
                    return subprocess.CompletedProcess(command, 0, "", "")

                with (
                    patch.multiple(
                        meta_harness,
                        EVOLVE_DIR=root,
                        AGENTS_DIR=root / "agents",
                        LOGS_DIR=run_dir,
                        PENDING_EVAL=run_dir / "pending_eval.json",
                        FRONTIER_VAL=frontier,
                        EVOLUTION_SUMMARY=run_dir / "evolution_summary.jsonl",
                        RESULTS_DIR=run_dir / "results",
                        FINALIZED=run_dir / "finalized.json",
                        _interrupted=False,
                    ),
                    patch.object(meta_harness.yaml, "safe_load", return_value=config),
                    patch.object(meta_harness, "propose_claude", side_effect=propose),
                    patch.object(
                        meta_harness,
                        "validate_candidates",
                        return_value=candidates if valid else [],
                    ),
                    patch.object(meta_harness, "run_benchmark", side_effect=benchmark),
                    patch.object(meta_harness, "load_results", return_value=results),
                    redirect_stdout(io.StringIO()),
                ):
                    meta_harness.run_evolve(args)

                rows = [
                    json.loads(line)
                    for line in (run_dir / "evolution_summary.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertEqual(len(rows), len(candidates))
                for row in rows:
                    name = row["system"]
                    with self.subTest(candidate=name):
                        self.assertEqual(row["delta"], deltas[name] if valid else -60.0)
                        self.assertEqual(
                            row["outcome"],
                            f"{scores[name]:.1f}% ({deltas[name]:+.1f})"
                            if valid
                            else "failed",
                        )


if __name__ == "__main__":
    unittest.main()
