from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import meta_harness


class EvolutionSummaryTests(unittest.TestCase):
    def test_deltas_use_pre_iteration_frontier(self) -> None:
        rewards = {
            "winner": [1.0] * 6 + [0.0] * 2,
            "improved": [1.0] * 5 + [0.0] * 3,
            "tied": [1.0] * 4 + [0.0] * 4,
            "regressed": [1.0] * 2 + [0.0] * 6,
        }
        deltas = {"winner": 0.25, "improved": 0.125, "tied": 0.0, "regressed": -0.25}
        candidates = [
            {"name": name, "import_path": f"agents.{name}:AgentHarness"}
            for name in rewards
        ]
        args = argparse.Namespace(
            run_name="test-run",
            fresh=False,
            skip_baseline=True,
            iterations=1,
            trials=8,
            concurrent=1,
            propose_timeout=1,
            skip_smoke=True,
            full_eval=False,
        )

        for valid in (True, False):
            with self.subTest(valid=valid), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run_dir = root / "logs" / "test-run"
                run_dir.mkdir(parents=True)
                frontier = run_dir / "frontier_val.json"
                frontier.write_text(
                    json.dumps({"_best": {"agent": "previous", "avg_pass_rate": 0.5}})
                )

                def propose(*_args, **_kwargs):
                    meta_harness.PENDING_EVAL.write_text(
                        json.dumps({"candidates": candidates})
                    )
                    return True

                with (
                    patch.multiple(
                        meta_harness,
                        EVOLVE_DIR=root,
                        AGENTS_DIR=root / "agents",
                        JOBS_DIR=root / "jobs" / "test-run",
                        LOGS_DIR=run_dir,
                        PENDING_EVAL=run_dir / "pending_eval.json",
                        FRONTIER_VAL=frontier,
                        EVOLUTION_SUMMARY=run_dir / "evolution_summary.jsonl",
                        _interrupted=False,
                    ),
                    patch.object(meta_harness, "propose_claude", side_effect=propose),
                    patch.object(
                        meta_harness, "validate_candidate", return_value=valid
                    ),
                    patch.object(
                        meta_harness, "harbor_run", return_value=(root / "job", True)
                    ),
                    patch.object(
                        meta_harness,
                        "parse_job_results",
                        side_effect=[
                            {"unit-task": values} for values in rewards.values()
                        ],
                    ),
                    patch.object(meta_harness, "parse_trial_metrics", return_value={}),
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
                    name = row["agent"]
                    with self.subTest(candidate=name):
                        self.assertEqual(row["delta"], deltas[name] if valid else -0.5)
                        self.assertEqual(
                            row["outcome"],
                            f"{sum(rewards[name]) / 8:.1%} ({deltas[name]:+.1%})"
                            if valid
                            else "failed",
                        )


if __name__ == "__main__":
    unittest.main()
