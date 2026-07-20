"""Held-out test evaluation on tasks DISJOINT from the dev/val subset.

Reconstructs the exact val subset the evolution used
(`select_dev_subset(all_tasks, dev_size, seed)`), samples `--test-size` tasks from the
REMAINING tasks (a separate seed), and benchmarks the given agents on that test set.
Outputs land under `<run-dir>/test/` so they never collide with the val logs.

    uv run python test_eval.py --run-dir logs/overnight \
        --agents agents.baseline_single:AgentHarness agents.baseline_react:AgentHarness \
                 agents.<best>:AgentHarness
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

import yaml

import benchmark
from data.loader import load_tasks, select_dev_subset

EVOLVE_DIR = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((EVOLVE_DIR / "config.yaml").read_text())


def test_task_ids(source, dev_size, dev_seed, test_size, test_seed) -> list[str]:
    all_tasks = load_tasks(source)
    val = {t.id for t in select_dev_subset(all_tasks, dev_size, dev_seed)}
    remaining = [t for t in sorted(all_tasks, key=lambda t: t.id) if t.id not in val]
    rng = random.Random(test_seed)
    picked = rng.sample(remaining, min(test_size, len(remaining)))
    picked.sort(key=lambda t: t.id)
    return [t.id for t in picked]


def main() -> None:
    ap = argparse.ArgumentParser(description="Held-out test eval (disjoint from val)")
    ap.add_argument("--agents", nargs="+", required=True, help="agent import paths")
    ap.add_argument("--run-dir", required=True, help="logs/<run> dir of the evolution")
    ap.add_argument("--source", default=None)
    ap.add_argument("--test-size", type=int, default=200)
    ap.add_argument("--test-seed", type=int, default=2024)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--model", default=None)
    ap.add_argument("--api-base", default=None)
    args = ap.parse_args()

    source = args.source or CONFIG["dataset"]["source"]
    dev_size = int(CONFIG["dataset"]["dev_size"])
    dev_seed = int(CONFIG["dataset"]["seed"])
    ids = test_task_ids(source, dev_size, dev_seed, args.test_size, args.test_seed)

    test_dir = Path(args.run_dir) / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / "test_task_ids.json").write_text(json.dumps(ids, indent=2))
    print(f"test set: {len(ids)} tasks (disjoint from {dev_size} val), source={source}")

    # One event loop for ALL agents: a fresh asyncio.run() per agent tears the loop down
    # while subprocess transports are still finalizing ("Event loop is closed").
    async def run_all():
        out = {}
        for imp in args.agents:
            name = benchmark.agent_name_of(imp)
            print(f"eval {name} on {len(ids)} test tasks "
                  f"(concurrency={args.concurrency})...", flush=True)
            summary = await benchmark.run_agent(
                imp, ids, test_dir, source, args.concurrency, args.model, args.api_base
            )
            out[name] = {
                "pass_rate": summary["pass_rate"],
                "n_pass": summary["n_pass"],
                "n_tasks": summary["n_tasks"],
                "n_missing": summary.get("n_missing", 0),
                "mean_turns": summary.get("mean_turns"),
                "mean_tokens": summary.get("mean_tokens"),
            }
            print(f"  {name}: test pass_rate={summary['pass_rate']:.1%} "
                  f"({summary['n_pass']}/{summary['n_tasks']}, "
                  f"missing={summary.get('n_missing', 0)})", flush=True)
            # Persist incrementally so a crash never loses completed agents.
            (test_dir / "test_summary.json").write_text(json.dumps(out, indent=2))
        return out

    asyncio.run(run_all())
    print("wrote", test_dir / "test_summary.json")


if __name__ == "__main__":
    main()
