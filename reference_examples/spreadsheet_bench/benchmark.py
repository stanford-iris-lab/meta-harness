"""Benchmark sweep: run agent scaffolds over a SpreadsheetBench dev subset.

Auto-discovers `agents/*.py`, runs `inner_loop.py` as a subprocess per (agent, task)
with bounded concurrency, aggregates the OJ pass rate, and writes per-agent summaries and
a frontier file.

    uv run python benchmark.py --agent agents.baseline_react --run-dir logs/adhoc
    uv run python benchmark.py --all --run-dir logs/adhoc
    uv run python benchmark.py --results --run-dir logs/adhoc
    uv run python benchmark.py --frontier --run-dir logs/adhoc

Dev-subset size comes from --dev-size, else env MH_N_TASKS, else config dataset.dev_size.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import yaml

from data.loader import load_tasks, select_dev_subset

EVOLVE_DIR = Path(__file__).resolve().parent
AGENTS_DIR = EVOLVE_DIR / "agents"
CONFIG = yaml.safe_load((EVOLVE_DIR / "config.yaml").read_text())

_SKIP_AGENT_FILES = {"__init__"}
# Retries for a task whose worker produced no record (OOM kill / transient crash).
_TASK_RETRIES = int(os.environ.get("MH_TASK_RETRIES", "2"))


def agent_name_of(import_path: str) -> str:
    name = import_path.split(":", 1)[0]
    if name.startswith("agents."):
        name = name[len("agents.") :]
    return name


def discover_agents() -> list[str]:
    out = []
    for f in sorted(AGENTS_DIR.glob("*.py")):
        if f.stem in _SKIP_AGENT_FILES:
            continue
        out.append(f"agents.{f.stem}")
    return out


def resolve_dev_size(arg_dev_size) -> int:
    if arg_dev_size is not None:
        return arg_dev_size
    env = os.environ.get("MH_N_TASKS")
    if env:
        return int(env)
    return int(CONFIG.get("dataset", {}).get("dev_size", 30))


def dev_task_ids(source: str, dev_size: int, seed: int) -> list[str]:
    tasks = load_tasks(source)
    subset = select_dev_subset(tasks, dev_size, seed)
    return [t.id for t in subset]


async def _run_task(sem, agent_import, task_id, agent_dir, source, model, api_base):
    out_file = agent_dir / f"task_{task_id}.json"
    log_file = agent_dir / f"task_{task_id}.jsonl"
    if out_file.exists():  # resume: skip completed tasks
        return
    cmd = [
        sys.executable, "inner_loop.py",
        "--agent", agent_import,
        "--task-id", task_id,
        "--source", source,
        "--out", str(out_file),
        "--log", str(log_file),
    ]
    if model:
        cmd += ["--model", model]
    if api_base is not None:
        cmd += ["--api-base", api_base]

    # Transient kills (OOM under the login-node cgroup cap, teardown) must be retried,
    # not recorded as failures — a missing record is scored 0 and silently deflates a run.
    last_err = ""
    for attempt in range(_TASK_RETRIES + 1):
        async with sem:
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=str(EVOLVE_DIR),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
        if out_file.exists():
            return
        last_err = (stderr or b"").decode("utf-8", "replace")[-1000:]
        killed = proc.returncode is not None and proc.returncode < 0
        if attempt < _TASK_RETRIES:
            print(f"    retry {task_id} (attempt {attempt + 1}, "
                  f"{'signal ' + str(proc.returncode) if killed else 'no output'})", flush=True)
            await asyncio.sleep(3 * (attempt + 1))

    # Retries exhausted: record a real failure so aggregation counts it as 0.
    out_file.write_text(json.dumps({
        "task_id": task_id, "agent": agent_import, "passed": False, "score": 0.0,
        "n_cases": 0, "n_pass": 0, "per_case": [], "n_turns": 0, "produced": False,
        "llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
        "cost_usd": 0.0, "runtime_s": 0.0,
        "error": last_err,
        "solution_code": "",
    }, indent=2))


async def run_agent(agent_import, task_ids, run_dir, source, concurrency, model, api_base):
    agent_dir = run_dir / agent_name_of(agent_import)
    agent_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    await asyncio.gather(*[
        _run_task(sem, agent_import, tid, agent_dir, source, model, api_base)
        for tid in task_ids
    ])
    return aggregate(run_dir, agent_import, task_ids)


def aggregate(run_dir, agent_import, task_ids) -> dict:
    name = agent_name_of(agent_import)
    agent_dir = run_dir / name
    per_task, costs, turns, tokens = {}, [], [], []
    missing = 0
    for tid in task_ids:
        f = agent_dir / f"task_{tid}.json"
        if not f.exists():
            per_task[tid] = 0.0
            missing += 1
            continue
        r = json.loads(f.read_text())
        per_task[tid] = 1.0 if r.get("passed") else 0.0
        if r.get("cost_usd") is not None:
            costs.append(r["cost_usd"])
        if r.get("n_turns"):
            turns.append(r["n_turns"])
        if r.get("total_tokens"):
            tokens.append(r["total_tokens"])
    n = len(task_ids)
    n_pass = int(sum(per_task.values()))
    summary = {
        "agent": name,
        "import_path": f"agents.{name}:AgentHarness",
        "source": None,
        "n_tasks": n,
        "n_pass": n_pass,
        "pass_rate": round(n_pass / n, 4) if n else 0.0,
        "per_task": per_task,
        "mean_cost_usd": round(sum(costs) / len(costs), 4) if costs else None,
        "mean_turns": round(sum(turns) / len(turns), 2) if turns else None,
        "mean_tokens": round(sum(tokens) / len(tokens), 1) if tokens else None,
        "n_missing": missing,
        "task_ids": list(task_ids),
    }
    if missing:
        # Never let truncated coverage masquerade as a real score.
        print(f"  WARNING: {name}: {missing}/{n} task records MISSING — scored 0. "
              f"pass_rate is a LOWER BOUND, not a valid measurement.", flush=True)
    (agent_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_summaries(run_dir: Path) -> list[dict]:
    out = []
    for sf in sorted(run_dir.glob("*/summary.json")):
        out.append(json.loads(sf.read_text()))
    return out


def write_frontier(run_dir: Path) -> dict:
    summaries = load_summaries(run_dir)
    if not summaries:
        return {}
    # Per-task best across agents.
    frontier: dict = {}
    for s in summaries:
        for task, rate in s["per_task"].items():
            if rate > frontier.get(task, {}).get("pass_rate", -1):
                frontier[task] = {"best_agent": s["agent"], "pass_rate": rate}
    # Pareto over (pass_rate up, mean_tokens down).
    pareto = sorted(summaries, key=lambda s: (-s["pass_rate"], s.get("mean_tokens") or 0))
    frontier["_pareto"] = [
        {"agent": s["agent"], "pass_rate": s["pass_rate"], "mean_tokens": s.get("mean_tokens")}
        for s in pareto
    ]
    best = pareto[0]
    frontier["_best"] = {"agent": best["agent"], "pass_rate": best["pass_rate"]}
    (run_dir / "frontier_val.json").write_text(json.dumps(frontier, indent=2))
    return frontier


def print_results(run_dir: Path) -> None:
    summaries = load_summaries(run_dir)
    if not summaries:
        print(f"(no summaries under {run_dir})")
        return
    summaries.sort(key=lambda s: -s["pass_rate"])
    width = max((len(s["agent"]) for s in summaries), default=10)
    print(f"{'agent':<{width}}  pass_rate  n_pass/n   mean_turns  mean_tokens")
    for s in summaries:
        print(
            f"{s['agent']:<{width}}  {s['pass_rate']:>8.1%}  "
            f"{s['n_pass']:>3}/{s['n_tasks']:<3}    "
            f"{(s.get('mean_turns') or 0):>6.1f}     {(s.get('mean_tokens') or 0):>10.0f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="SpreadsheetBench benchmark sweep")
    ap.add_argument("--agent", default=None, help="single agent import (agents.<name>)")
    ap.add_argument("--all", action="store_true", help="run all discovered agents")
    ap.add_argument("--source", default=None)
    ap.add_argument("--dev-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--run-dir", default=str(EVOLVE_DIR / "logs" / "adhoc"))
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--results", action="store_true", help="print existing summaries")
    ap.add_argument("--frontier", action="store_true", help="(re)write frontier_val.json")
    ap.add_argument("--model", default=None)
    ap.add_argument("--api-base", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.results:
        print_results(run_dir)
        return
    if args.frontier and not (args.agent or args.all):
        write_frontier(run_dir)
        print(f"wrote {run_dir / 'frontier_val.json'}")
        return

    source = args.source or CONFIG.get("dataset", {}).get("source", "sample_data_200")
    dev_size = resolve_dev_size(args.dev_size)
    seed = args.seed if args.seed is not None else int(CONFIG.get("dataset", {}).get("seed", 42))
    concurrency = args.concurrency or int(CONFIG.get("benchmark", {}).get("concurrency", 16))
    task_ids = dev_task_ids(source, dev_size, seed)

    agents = [args.agent] if args.agent else discover_agents() if args.all else None
    if not agents:
        ap.error("specify --agent <import> or --all")

    print(
        f"benchmark: agents={len(agents)} tasks={len(task_ids)} source={source} "
        f"concurrency={concurrency} run_dir={run_dir}"
    )
    for agent_import in agents:
        summary = asyncio.run(
            run_agent(agent_import, task_ids, run_dir, source, concurrency,
                      args.model, args.api_base)
        )
        summary["source"] = source
        (run_dir / summary["agent"] / "summary.json").write_text(json.dumps(summary, indent=2))
        print(
            f"  {summary['agent']}: pass_rate={summary['pass_rate']:.1%} "
            f"({summary['n_pass']}/{summary['n_tasks']})  "
            f"mean_turns={summary.get('mean_turns')}  mean_tokens={summary.get('mean_tokens')}"
        )

    write_frontier(run_dir)
    print_results(run_dir)


if __name__ == "__main__":
    main()
