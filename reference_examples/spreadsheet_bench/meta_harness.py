"""Autonomous evolution loop for agent scaffolds on SpreadsheetBench.

Starts from agents/baseline_react.py and evolves improvements on a config-driven dev
subset of SpreadsheetBench, scored OJ-style on the cheap gpt-oss-120b solver (QNAIGC).

    uv run python meta_harness.py --iterations 5
    uv run python meta_harness.py --iterations 10 --fresh --run-name my-run
    MH_N_TASKS=10 uv run python meta_harness.py --iterations 3   # smaller dev subset

The proposer (Claude Code) analyzes results + failed trajectories and writes ONE new
scaffold per iteration into agents/. This driver validates, smoke-tests, benchmarks it via
benchmark.py, and updates the frontier. It does NOT run benchmarks inside the proposer.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

# ── ANSI colors ──────────────────────────────────────────────
_USE_COLOR = sys.stdout.isatty()


def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _bold(t):
    return _c("1", t)


def _dim(t):
    return _c("2", t)


def _green(t):
    return _c("32", t)


def _red(t):
    return _c("31", t)


def _yellow(t):
    return _c("33", t)


def _cyan(t):
    return _c("36", t)


def _ts():
    return _dim(datetime.now().strftime("[%H:%M:%S]"))


def _elapsed(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def _rate_str(rate):
    s = f"{rate:.0%}"
    if rate >= 0.5:
        return _green(s)
    elif rate >= 0.2:
        return _yellow(s)
    return _red(s)


import claude_wrapper  # noqa: E402

EVOLVE_DIR = Path(__file__).parent
CONFIG = yaml.safe_load((EVOLVE_DIR / "config.yaml").read_text())
AGENTS_DIR = EVOLVE_DIR / "agents"

# Per-run paths (rebound in run_evolve).
LOGS_DIR = EVOLVE_DIR / "logs"
PENDING_EVAL = LOGS_DIR / "pending_eval.json"
FRONTIER_VAL = LOGS_DIR / "frontier_val.json"
EVOLUTION_SUMMARY = LOGS_DIR / "evolution_summary.jsonl"

BASELINES = [
    ("baseline_react", "agents.baseline_react:AgentHarness"),
    ("baseline_single", "agents.baseline_single:AgentHarness"),
]
BASELINE_AGENT_NAME = BASELINES[0][0]
BASELINE_FILES = {"__init__.py", "baseline_react.py", "baseline_single.py"}

SOURCE = CONFIG.get("dataset", {}).get("source", "sample_data_200")
DEV_SIZE = int(os.environ.get("MH_N_TASKS", CONFIG.get("dataset", {}).get("dev_size", 30)))
DEFAULT_CONCURRENCY = int(CONFIG.get("benchmark", {}).get("concurrency", 16))

PROPOSER_ALLOWED_TOOLS = ["Read", "Glob", "Grep", "Agent", "Write", "Edit", "Bash"]

_interrupted = False


def _handle_signal(signum, frame):
    global _interrupted
    _interrupted = True
    print("\nInterrupted, finishing current step...", flush=True)


def agent_name_of(import_path: str) -> str:
    name = import_path.split(":", 1)[0]
    return name[len("agents.") :] if name.startswith("agents.") else name


def run_cmd(cmd, timeout=7200, cwd=None):
    try:
        return subprocess.run(
            cmd, cwd=cwd, timeout=timeout, capture_output=True, text=True,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, "", f"Timed out after {timeout}s")


# ── benchmarking via benchmark.py ────────────────────────────────────────────
def benchmark_run(import_path, dev_size, concurrency, timeout=14400):
    """Run benchmark.py for one agent; return its summary dict (or None on failure)."""
    name = agent_name_of(import_path)
    cmd = [
        sys.executable, "benchmark.py",
        "--agent", import_path,
        "--run-dir", str(LOGS_DIR),
        "--dev-size", str(dev_size),
        "--concurrency", str(concurrency),
        "--source", SOURCE,
    ]
    try:
        result = subprocess.run(
            cmd, cwd=str(EVOLVE_DIR), timeout=timeout,
            stdout=None, stderr=subprocess.PIPE, text=True, env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        result = subprocess.CompletedProcess(cmd, 124, "", f"Timed out after {timeout}s")
    if result.returncode not in (0, 124):
        print(f"  {_red('benchmark failed')} exit={result.returncode} agent={name}")
        if result.stderr:
            print(f"  {_dim(result.stderr[:500])}")
    summary_file = LOGS_DIR / name / "summary.json"
    if summary_file.exists():
        return json.loads(summary_file.read_text())
    return None


def summary_to_result(summary):
    """(per_task dict, avg pass_rate, metrics dict) from a benchmark summary."""
    per_task = {k: float(v) for k, v in summary.get("per_task", {}).items()}
    avg = float(summary.get("pass_rate", 0.0))
    metrics = {
        "mean_cost_usd": summary.get("mean_cost_usd"),
        "mean_turns": summary.get("mean_turns"),
        "mean_tokens": summary.get("mean_tokens"),
        "n_pass": summary.get("n_pass"),
        "n_tasks": summary.get("n_tasks"),
    }
    return per_task, avg, metrics


# ── frontier + summary bookkeeping ───────────────────────────────────────────
def read_frontier():
    return json.loads(FRONTIER_VAL.read_text()) if FRONTIER_VAL.exists() else {}


def best_avg_agent():
    fr = read_frontier()
    best = fr.get("_best", {})
    return best.get("avg_pass_rate", best.get("pass_rate", 0)), best.get("agent", "none")


def count_iterations():
    if not EVOLUTION_SUMMARY.exists():
        return 0
    m = 0
    for line in EVOLUTION_SUMMARY.read_text().strip().split("\n"):
        if line.strip():
            try:
                m = max(m, json.loads(line).get("iteration", 0))
            except json.JSONDecodeError:
                pass
    return m


def update_evolution_summary(iteration, candidates, results, propose_time=None,
                             bench_time=None, metrics=None):
    best_avg, _ = best_avg_agent()
    metrics = metrics or {}
    with open(EVOLUTION_SUMMARY, "a") as f:
        for i, c in enumerate(candidates):
            name = c["name"]
            per_task, avg = results.get(name, ({}, 0))
            row = {
                "iteration": iteration,
                "agent": name,
                "import_path": c.get("import_path", ""),
                "pass_rate": round(avg, 3),
                "per_task": {k: round(v, 3) for k, v in per_task.items()},
                "hypothesis": c.get("hypothesis", ""),
                "changes": c.get("changes", ""),
                "delta": round(avg - best_avg, 3) if best_avg else None,
                "outcome": f"{avg:.1%} ({avg - best_avg:+.1%})" if avg > 0 else "failed",
            }
            if i == 0 and propose_time is not None:
                row["timing_s"] = {
                    "propose": round(propose_time, 1),
                    "bench": round(bench_time, 1) if bench_time else None,
                }
            if name in metrics:
                row["rollout_metrics"] = metrics[name]
            f.write(json.dumps(row) + "\n")


# ── proposer ─────────────────────────────────────────────────────────────────
def render_task_prompt(iteration):
    return (
        f"Run iteration {iteration} of the SpreadsheetBench scaffold evolution loop. "
        f"Solver model: {CONFIG.get('model', {}).get('name')} (cheap, via QNAIGC). "
        f"Start from agents/baseline_react.py as the parent.\n\n"
        f"## Eval: {DEV_SIZE} SpreadsheetBench instructions (dev subset), OJ-scored "
        f"(all test cases must pass).\n\n"
        f"Focus on scaffold changes that help the agent write a correct, general Python "
        f"solution program (exploration strategy, execution-feedback use, self-checking, "
        f"context management) — never task-specific hints.\n\n"
        f"## Run directories\n"
        f"- `{EVOLUTION_SUMMARY}` — past results\n"
        f"- `{FRONTIER_VAL}` — frontier\n"
        f"- Per-task trajectories + scores: `{LOGS_DIR}/<agent>/task_<id>.jsonl`\n"
        f"- Write pending_eval.json to: `{PENDING_EVAL}`"
    )


def propose_claude(task_prompt, iteration, timeout=2400):
    os.environ.pop("CLAUDECODE", None)
    saved_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    result = claude_wrapper.run(
        prompt=task_prompt,
        model="opus",
        allowed_tools=PROPOSER_ALLOWED_TOOLS,
        skills=[str(EVOLVE_DIR / ".claude/skills/meta-harness-spreadsheet-bench")],
        cwd=str(EVOLVE_DIR),
        log_dir=str(LOGS_DIR / "claude_sessions"),
        name=f"iter{iteration}",
        timeout_seconds=timeout,
        effort="max",
    )
    if saved_key:
        os.environ["ANTHROPIC_API_KEY"] = saved_key
    if result.exit_code != 0:
        print(f"  {_red('proposer failed')} exit={result.exit_code}")
        if result.stderr:
            print(f"  {_dim(result.stderr[:500])}")
        return False
    result.show()
    return PENDING_EVAL.exists()


# ── validation ───────────────────────────────────────────────────────────────
def validate_candidate(name, import_path):
    module_path = import_path.split(":")[0]
    result = run_cmd(
        ["uv", "run", "python", "-c", f"from {module_path} import *; print('OK')"],
        cwd=str(EVOLVE_DIR), timeout=60,
    )
    if result.returncode == 0 and "OK" in result.stdout:
        return True
    print(f"  {_red('import FAIL')}: {name}")
    if result.stderr:
        print(f"    {_dim(result.stderr[:400])}")
    return False


def smoke_test(name, import_path, timeout=900):
    """Run the candidate on ONE dev task; pass if inner_loop exits 0 without an error."""
    from data.loader import load_tasks, select_dev_subset

    seed = int(CONFIG.get("dataset", {}).get("seed", 42))
    tasks = select_dev_subset(load_tasks(SOURCE), DEV_SIZE, seed)
    if not tasks:
        print(f"  {_yellow('smoke skipped')}: no dev tasks")
        return True
    task_id = tasks[0].id
    out = LOGS_DIR / "_smoke" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    cmd = [
        sys.executable, "inner_loop.py", "--agent", import_path,
        "--task-id", task_id, "--source", SOURCE, "--out", str(out),
    ]
    result = run_cmd(cmd, cwd=str(EVOLVE_DIR), timeout=timeout)
    if result.returncode != 0 or not out.exists():
        print(f"  {_red('smoke FAIL')}: {name} exit={result.returncode}")
        if result.stderr:
            print(f"    {_dim(result.stderr[:400])}")
        return False
    data = json.loads(out.read_text())
    if data.get("error"):
        print(f"  {_red('smoke FAIL')}: {name} (runtime error)")
        return False
    print(f"  {_green('smoke OK')}: {name} (task {task_id})")
    return True


def fresh_start():
    if AGENTS_DIR.exists():
        for f in AGENTS_DIR.iterdir():
            if f.name in BASELINE_FILES or f.name == "__pycache__":
                continue
            if f.is_dir():
                run_cmd(["rm", "-rf", str(f)])
            elif f.suffix == ".py":
                f.unlink()
    for f in [EVOLUTION_SUMMARY, FRONTIER_VAL, PENDING_EVAL]:
        if f.exists():
            f.unlink()
    print("  Fresh start: cleared generated agents and log files")


# ── driver ───────────────────────────────────────────────────────────────────
def run_evolve(args):
    global LOGS_DIR, PENDING_EVAL, FRONTIER_VAL, EVOLUTION_SUMMARY

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    LOGS_DIR = EVOLVE_DIR / "logs" / run_name
    PENDING_EVAL = LOGS_DIR / "pending_eval.json"
    FRONTIER_VAL = LOGS_DIR / "frontier_val.json"
    EVOLUTION_SUMMARY = LOGS_DIR / "evolution_summary.jsonl"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.fresh:
        fresh_start()

    print(
        f"{_ts()} {_bold('SpreadsheetBench evolution')}  run={_cyan(run_name)}  "
        f"model={_cyan(CONFIG.get('model', {}).get('name'))}  iters={args.iterations}  "
        f"dev_tasks={DEV_SIZE}"
    )

    # ── Phase 0: baselines ─────────────────────────────────────
    if not args.skip_baseline:
        print(f"\n{_ts()} {_bold('Phase 0: Baselines')}  agents={len(BASELINES)}")
        for bl_name, bl_import in BASELINES:
            print(f"  {_ts()} running {_bold(bl_name)}...", flush=True)
            t0 = time.time()
            summary = benchmark_run(bl_import, DEV_SIZE, args.concurrent)
            if summary:
                _, avg, _ = summary_to_result(summary)
                print(f"    {bl_name}: avg={_rate_str(avg)} ({_elapsed(time.time() - t0)})")
            else:
                print(f"    {_red('FAIL')} {bl_name}")

    # ── Phase 1..N: evolution ──────────────────────────────────
    start_iteration = count_iterations() + 1
    for i in range(args.iterations):
        if _interrupted:
            print("Interrupted.")
            break
        iteration = start_iteration + i
        iter_start = time.time()
        best_avg, best_agent = best_avg_agent()
        print(
            f"\n{_ts()} {_bold(f'Iteration {iteration}')} ({i + 1}/{args.iterations})  "
            f"frontier={best_agent} @ {best_avg:.1%}"
        )
        print("─" * 60)

        if PENDING_EVAL.exists():
            PENDING_EVAL.unlink()

        propose_start = time.time()
        print(f"  {_ts()} {_cyan('proposing')} new candidate...", flush=True)
        ok = propose_claude(render_task_prompt(iteration), iteration, timeout=args.propose_timeout)
        propose_time = time.time() - propose_start
        if not ok:
            print(f"  {_red('FAIL')} proposer returned no candidates ({_elapsed(propose_time)})")
            continue

        candidates = json.loads(PENDING_EVAL.read_text()).get("candidates", [])
        for c in candidates:
            if "import_path" in c and ":" in c["import_path"]:
                module, _ = c["import_path"].rsplit(":", 1)
                c["import_path"] = f"{module}:AgentHarness"
            else:
                c["import_path"] = f"agents.{c['name']}:AgentHarness"
        print(f"  {_ts()} proposed {len(candidates)} candidate(s) in {_elapsed(propose_time)}")

        # Validate + smoke
        valid = []
        for ci, c in enumerate(candidates):
            name, import_path = c["name"], c["import_path"]
            prefix = f"    [{ci + 1}/{len(candidates)}] {name}:"
            if validate_candidate(name, import_path):
                if args.skip_smoke or smoke_test(name, import_path):
                    print(f"{prefix} {_green('valid')}")
                    valid.append(c)
                else:
                    print(f"{prefix} {_red('smoke FAIL')}")
            else:
                print(f"{prefix} {_red('import FAIL')}")
            if _interrupted:
                break

        if not valid:
            print(f"  {_red('0 valid')} candidates, skipping iteration")
            update_evolution_summary(iteration, candidates, {}, propose_time=propose_time)
            continue

        # Benchmark
        bench_start = time.time()
        results, all_metrics = {}, {}
        for ci, c in enumerate(valid):
            if _interrupted:
                break
            name, import_path = c["name"], c["import_path"]
            print(f"  {_ts()} {_cyan('benchmarking')} {_bold(name)}...", flush=True)
            t0 = time.time()
            summary = benchmark_run(import_path, DEV_SIZE, args.concurrent)
            if summary:
                per_task, avg, metrics = summary_to_result(summary)
                results[name] = (per_task, avg)
                all_metrics[name] = metrics
                delta = avg - best_avg
                dc = _green(f"{delta:+.1%}") if delta > 0 else (
                    _red(f"{delta:+.1%}") if delta < 0 else _dim(f"{delta:+.1%}"))
                cost = f"${metrics['mean_cost_usd']:.3f}" if metrics.get("mean_cost_usd") else "?"
                print(
                    f"    avg={_rate_str(avg)}  delta={dc}  "
                    f"mean_turns={metrics.get('mean_turns')}  mean_cost={cost}  "
                    f"({_elapsed(time.time() - t0)})"
                )
            else:
                results[name] = ({}, 0)
                print(f"    {_red('FAIL')} benchmark crashed ({_elapsed(time.time() - t0)})")
        bench_time = time.time() - bench_start

        update_evolution_summary(
            iteration, valid, results, propose_time=propose_time,
            bench_time=bench_time, metrics=all_metrics,
        )
        new_best_avg, new_best_agent = best_avg_agent()
        status = _green("NEW BEST") if new_best_avg > best_avg else _dim("no improvement")
        print(f"  {_ts()} {status}  frontier={new_best_agent} @ {new_best_avg:.1%}")
        print(f"  {_dim(f'timing: propose={_elapsed(propose_time)} bench={_elapsed(bench_time)} total={_elapsed(time.time() - iter_start)}')}")

    # ── Phase Final: winner on the full dataset ────────────────
    if _interrupted or not args.full_eval:
        return
    print(f"\n{_ts()} {_bold('Phase Final: full-dataset eval for the best agent')}")
    _, best_agent = best_avg_agent()
    if best_agent and best_agent != BASELINE_AGENT_NAME:
        import_path = f"agents.{best_agent}:AgentHarness"
        print(f"  {_ts()} running {_bold(best_agent)} on the full source...", flush=True)
        summary = benchmark_run(import_path, 0, args.concurrent)  # dev_size=0 -> all tasks
        if summary:
            _, avg, _ = summary_to_result(summary)
            print(f"  {_bold(best_agent)} (full): avg={_rate_str(avg)}")
    print(f"\n{_ts()} {_bold('Evolution complete.')}")


def main():
    p = argparse.ArgumentParser(description="SpreadsheetBench scaffold evolution loop")
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--propose-timeout", type=int, default=2400)
    p.add_argument("--run-name", default=None)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--skip-smoke", action="store_true")
    p.add_argument("--full-eval", action="store_true", help="final full-dataset eval of the winner")
    p.add_argument("--concurrent", type=int, default=DEFAULT_CONCURRENCY)
    args = p.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    run_evolve(args)


if __name__ == "__main__":
    main()
