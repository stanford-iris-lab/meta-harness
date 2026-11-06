"""Autonomous evolution loop for agent scaffolds on Terminal-Bench 2.0.

Starts from the shipped KIRA baseline and evolves improvements on the full
official TB2 dataset used in the paper runs.

    uv run python meta_harness.py --iterations 5
    uv run python meta_harness.py --iterations 10 --trials 2 --fresh
"""

import argparse
import importlib
import inspect
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import claude_wrapper
from dotenv import load_dotenv

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
    """Short timestamp prefix."""
    return _dim(datetime.now().strftime("[%H:%M:%S]"))


def _elapsed(seconds):
    """Format seconds as 1m23s or 45s."""
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def _rate_str(rate):
    """Color-coded pass rate."""
    s = f"{rate:.0%}"
    if rate >= 0.75:
        return _green(s)
    elif rate >= 0.25:
        return _yellow(s)
    return _red(s)


load_dotenv(override=True)

EVOLVE_DIR = Path(__file__).parent
LOGS_DIR = EVOLVE_DIR / "logs"
JOBS_DIR = EVOLVE_DIR / "jobs"
PENDING_EVAL = LOGS_DIR / "pending_eval.json"
FRONTIER_VAL = LOGS_DIR / "frontier_val.json"
EVOLUTION_SUMMARY = LOGS_DIR / "evolution_summary.jsonl"
AGENTS_DIR = EVOLVE_DIR / "agents"

BASELINES = [
    ("kira-baseline", "agents.baseline_kira:AgentHarness"),
    ("terminus2-baseline", "agents.baseline_terminus2:AgentHarness"),
]
BASELINE_AGENT_NAME = BASELINES[0][0]  # primary baseline for frontier comparison

EVAL_TASK_SET = "full"
N_EVAL_TASKS = 89  # full official TB2 dataset used in the paper runs

SMOKE_TEST_TASK = "extract-elf"  # simple task, reliably fast

MODEL = "anthropic/claude-opus-4-6"
DEFAULT_SEARCH_TRIALS = 2
DEFAULT_CONCURRENCY = 50
HARBOR_TIMEOUT_SECONDS = int(os.environ.get("HARBOR_TIMEOUT_SECONDS", 8 * 60 * 60))

PROPOSER_ALLOWED_TOOLS = [
    "Read",
    "Glob",
    "Grep",
    "Agent",
    "Write",
    "Edit",
    "Bash",
]

_interrupted = False


def _handle_signal(signum, frame):
    global _interrupted
    _interrupted = True
    print("\nInterrupted, finishing current step...", flush=True)


def run_cmd(cmd, timeout=HARBOR_TIMEOUT_SECONDS, cwd=None):
    """Run a subprocess; return CompletedProcess (returncode=124 on timeout)."""
    env = os.environ.copy()
    env["HARBOR_MODEL"] = MODEL
    # Ensure dotenv-loaded keys survive through uv run subprocess
    for key in ("RUNLOOP_API_KEY", "ANTHROPIC_API_KEY"):
        val = os.environ.get(key)
        if val:
            env[key] = val
    try:
        return subprocess.run(
            cmd, cwd=cwd, timeout=timeout, capture_output=True, text=True, env=env
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, "", f"Timed out after {timeout}s")


def validate_agent_class(import_path):
    """Load and return the exact Terminus2 subclass named by an import path."""
    module_name, separator, class_name = import_path.partition(":")
    if not separator or not module_name or not class_name:
        raise ValueError("agent import path must be MODULE:CLASS")

    module = importlib.import_module(module_name)
    agent_class = getattr(module, class_name)
    if not inspect.isclass(agent_class):
        raise TypeError(f"{import_path} is not a class")

    from harbor.agents.terminus_2.terminus_2 import Terminus2

    if not issubclass(agent_class, Terminus2):
        raise TypeError(f"{import_path} must subclass Terminus2")
    return agent_class


def harbor_run(import_path, job_name, n_trials=2, n_concurrent=10):
    """Run Harbor on the paper TB2 config using the selected environment.

    The result is None if Harbor fails or times out.
    Callers do not parse partial results from timed-out jobs.
    """
    cmd = [
        str(EVOLVE_DIR / "scripts" / "run_eval.sh"),
        import_path,
        EVAL_TASK_SET,
        str(n_trials),
        str(n_concurrent),
        "--job-name",
        job_name,
        "--jobs-dir",
        str(JOBS_DIR),
    ]

    env = os.environ.copy()
    env["HARBOR_MODEL"] = MODEL
    env["HARBOR_TIMEOUT_SECONDS"] = str(HARBOR_TIMEOUT_SECONDS)
    env["HARBOR_EXTERNAL_TIMEOUT"] = "1"

    try:
        result = subprocess.run(
            cmd,
            cwd=str(EVOLVE_DIR),
            timeout=HARBOR_TIMEOUT_SECONDS,
            stdout=None,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
    except subprocess.TimeoutExpired:
        result = subprocess.CompletedProcess(
            cmd, 124, "", f"Timed out after {HARBOR_TIMEOUT_SECONDS}s"
        )

    job_dir = JOBS_DIR / job_name
    if result.returncode not in (0, 124):
        print(f"  {_red('harbor failed')} exit={result.returncode} job={job_name}")
        if result.stderr:
            print(f"  {result.stderr[:500]}")
        return job_dir, None

    if result.returncode == 124:
        print(f"  {_yellow('harbor timed out')} job={job_name}")
        return job_dir, None

    return job_dir, True  # signal success; callers use parse_job_results directly


def parse_job_results(job_dir, expected_trials=None):
    """Parse per-task rewards from a harbor job directory.

    Always reads individual trial dirs (ground truth). Ignores top-level result.json
    which may be a stale snapshot written before all trials complete.

    Every trial dir counts: no result.json, no verifier_result, corrupt JSON,
    errored trials — all count as reward=0. This matches harbor's metric
    (total_passes / total_trials).

    Returns dict: {task_name: [reward1, reward2, ...]}
    """
    task_rewards = {}

    for trial_dir in sorted(job_dir.iterdir()):
        if not trial_dir.is_dir() or "__" not in trial_dir.name:
            continue
        task = trial_dir.name.rsplit("__", 1)[0]

        rf = trial_dir / "result.json"
        if not rf.exists():
            task_rewards.setdefault(task, []).append(0.0)
            continue
        try:
            r = json.loads(rf.read_text())
        except (json.JSONDecodeError, OSError):
            task_rewards.setdefault(task, []).append(0.0)
            continue

        vr = r.get("verifier_result") or {}
        reward = (vr.get("rewards") or {}).get("reward")
        task_rewards.setdefault(task, []).append(
            float(reward) if reward is not None else 0.0
        )

    # Validate trial counts
    if expected_trials:
        for task, rewards in task_rewards.items():
            if len(rewards) != expected_trials:
                print(
                    f"  {_yellow('warning')}: {task} has {len(rewards)}/{expected_trials} trials"
                )

    return task_rewards


def compute_pass_rates(task_rewards):
    """Compute pass rate per task and flat average. Returns (per_task, avg).

    avg is total_passes / total_trials (flat, matching harbor's metric),
    NOT the mean of per-task rates.
    """
    per_task = {}
    total_passes = 0
    total_trials = 0
    for task, rewards in task_rewards.items():
        per_task[task] = sum(r > 0 for r in rewards) / len(rewards) if rewards else 0.0
        total_passes += sum(r > 0 for r in rewards)
        total_trials += len(rewards)

    avg = total_passes / total_trials if total_trials else 0.0
    return per_task, avg


def parse_trial_metrics(job_dir):
    """Extract per-trial rollout metrics from a harbor job directory.

    Returns dict: {task_name: [trial_metrics, ...]}
    where trial_metrics = {n_input_tokens, n_output_tokens, n_cache_tokens,
                           cost_usd, n_turns, n_api_calls, reward}
    """
    per_task = {}
    for trial_dir in sorted(job_dir.iterdir()):
        if not trial_dir.is_dir() or "__" not in trial_dir.name:
            continue
        task = trial_dir.name.rsplit("__", 1)[0]
        rf = trial_dir / "result.json"
        if not rf.exists():
            continue
        try:
            r = json.loads(rf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        ar = r.get("agent_result") or {}
        md = ar.get("metadata") or {}
        vr = r.get("verifier_result") or {}
        reward = (vr.get("rewards") or {}).get("reward")
        metrics = {
            "n_input_tokens": ar.get("n_input_tokens"),
            "n_output_tokens": ar.get("n_output_tokens"),
            "n_cache_tokens": ar.get("n_cache_tokens"),
            "cost_usd": ar.get("cost_usd"),
            "n_turns": md.get("n_episodes"),
            "n_api_calls": len(md.get("api_request_times_msec", [])),
            "reward": reward,
        }
        per_task.setdefault(task, []).append(metrics)
    return per_task


def summarize_trial_metrics(trial_metrics):
    """Aggregate per-trial metrics into a summary dict."""
    all_costs = []
    all_input = []
    all_output = []
    all_cache = []
    all_turns = []
    per_task_summary = {}

    for task, trials in trial_metrics.items():
        task_costs = [t["cost_usd"] for t in trials if t["cost_usd"] is not None]
        task_turns = [t["n_turns"] for t in trials if t["n_turns"] is not None]
        per_task_summary[task] = {
            "mean_cost": round(sum(task_costs) / len(task_costs), 3)
            if task_costs
            else None,
            "mean_turns": round(sum(task_turns) / len(task_turns), 1)
            if task_turns
            else None,
            "n_trials": len(trials),
        }
        all_costs.extend(task_costs)
        all_turns.extend(task_turns)
        all_input.extend(
            t["n_input_tokens"] for t in trials if t["n_input_tokens"] is not None
        )
        all_output.extend(
            t["n_output_tokens"] for t in trials if t["n_output_tokens"] is not None
        )
        all_cache.extend(
            t["n_cache_tokens"] for t in trials if t["n_cache_tokens"] is not None
        )

    n_trials = sum(len(v) for v in trial_metrics.values())
    return {
        "n_trials": n_trials,
        "total_cost_usd": round(sum(all_costs), 2) if all_costs else None,
        "mean_cost_usd": round(sum(all_costs) / len(all_costs), 3)
        if all_costs
        else None,
        "total_input_tokens": sum(all_input) if all_input else None,
        "total_output_tokens": sum(all_output) if all_output else None,
        "total_cache_tokens": sum(all_cache) if all_cache else None,
        "mean_turns": round(sum(all_turns) / len(all_turns), 1) if all_turns else None,
        "per_task": per_task_summary,
    }


def count_iterations():
    """Highest iteration in evolution_summary.jsonl (for resume)."""
    if not EVOLUTION_SUMMARY.exists():
        return 0
    max_iter = 0
    for line in EVOLUTION_SUMMARY.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            max_iter = max(max_iter, json.loads(line).get("iteration", 0))
        except json.JSONDecodeError:
            continue
    return max_iter


def update_frontier(candidates_results):
    """Update frontier_val.json with best agent per task and overall best."""
    frontier = json.loads(FRONTIER_VAL.read_text()) if FRONTIER_VAL.exists() else {}

    for agent_name, (per_task, avg) in candidates_results.items():
        for task, rate in per_task.items():
            current_best = frontier.get(task, {}).get("pass_rate", -1)
            if rate > current_best:
                frontier[task] = {
                    "best_agent": agent_name,
                    "pass_rate": rate,
                }

        current_best_avg = frontier.get("_best", {}).get("avg_pass_rate", -1)
        if avg > current_best_avg:
            frontier["_best"] = {
                "agent": agent_name,
                "avg_pass_rate": avg,
            }

    FRONTIER_VAL.write_text(json.dumps(frontier, indent=2))


def update_evolution_summary(
    iteration,
    candidates,
    results,
    previous_best,
    propose_time=None,
    bench_time=None,
    metrics=None,
):
    """Record deltas against the frontier score at the start of the iteration."""
    metrics = metrics or {}

    with open(EVOLUTION_SUMMARY, "a") as f:
        for i, c in enumerate(candidates):
            name = c["name"]
            per_task, avg = results.get(name, ({}, 0))
            row = {
                "iteration": iteration,
                "agent": name,
                "import_path": c.get("import_path", ""),
                "avg_pass_rate": round(avg, 3),
                "per_task": {k: round(v, 3) for k, v in per_task.items()},
                "hypothesis": c.get("hypothesis", ""),
                "changes": c.get("changes", ""),
                "delta": round(avg - previous_best, 3) if previous_best else None,
                "outcome": f"{avg:.1%} ({avg - previous_best:+.1%})"
                if avg > 0
                else "failed",
            }
            if i == 0 and propose_time is not None:
                row["timing_s"] = {
                    "propose": round(propose_time, 1),
                    "bench": round(bench_time, 1) if bench_time else None,
                }
            if name in metrics:
                row["rollout_metrics"] = metrics[name]
            f.write(json.dumps(row) + "\n")


def propose_claude(task_prompt, iteration, timeout=2400):
    """Call Claude Code to propose new agent candidates. Returns True if pending_eval.json exists."""
    os.environ.pop("CLAUDECODE", None)
    # Strip API key so claude CLI uses subscription auth, not API (avoids rate limits)
    saved_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    result = claude_wrapper.run(
        prompt=task_prompt,
        model="opus",
        allowed_tools=PROPOSER_ALLOWED_TOOLS,
        skills=[str(EVOLVE_DIR / ".claude/skills/meta-harness-terminal-bench-2")],
        cwd=str(EVOLVE_DIR),
        log_dir=str(LOGS_DIR / "claude_sessions"),
        name=f"iter{iteration}",
        timeout_seconds=timeout,
        effort="max",
    )
    # Restore API key for harbor eval runs
    if saved_key:
        os.environ["ANTHROPIC_API_KEY"] = saved_key
    if result.exit_code != 0:
        print(f"  {_red('proposer failed')} exit={result.exit_code}")
        if result.stderr:
            print(f"  {_dim(result.stderr[:500])}")
        return False
    result.show()
    return PENDING_EVAL.exists()


def validate_candidate(name, import_path):
    """Import-check a candidate agent. Returns True if valid."""
    result = run_cmd(
        [
            "uv",
            "run",
            "python",
            str(Path(__file__)),
            "--validate-agent",
            import_path,
        ],
        cwd=str(EVOLVE_DIR),
        timeout=30,
    )
    if result.returncode == 0 and "OK" in result.stdout:
        return True
    print(f"  {_red('import FAIL')}: {name}")
    if result.stderr:
        print(f"    {_dim(result.stderr[:300])}")
    return False


def smoke_test(name, import_path, timeout=1800):
    """Run 1 trial on 1 task to check for runtime crashes. Returns True if passed."""
    job_name = f"smoke-{name}"
    job_dir = JOBS_DIR / job_name
    if job_dir.exists():
        run_cmd(["rm", "-rf", str(job_dir)])

    cmd = [
        str(EVOLVE_DIR / "scripts" / "run_eval.sh"),
        import_path,
        "full",
        "1",
        "1",
        "-i",
        SMOKE_TEST_TASK,
        "--job-name",
        job_name,
        "--jobs-dir",
        str(JOBS_DIR),
    ]
    t0 = time.time()
    result = run_cmd(cmd, timeout=timeout, cwd=str(EVOLVE_DIR))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(
            f"  {_red('smoke FAIL')}: {name} exit={result.returncode} ({_elapsed(elapsed)})"
        )
        if result.stderr:
            print(f"    {_dim(result.stderr[:300])}")
        return False

    result_file = job_dir / "result.json"
    if not result_file.exists():
        print(f"  {_red('smoke FAIL')}: {name} (no result.json, {_elapsed(elapsed)})")
        return False

    data = json.loads(result_file.read_text())
    n_errors = data.get("stats", {}).get("n_errors", 0)
    if n_errors > 0:
        print(
            f"  {_red('smoke FAIL')}: {name} ({n_errors} errors, {_elapsed(elapsed)})"
        )
        return False

    print(f"  {_green('smoke OK')}: {name} ({_elapsed(elapsed)})")
    return True


def render_task_prompt(iteration, n_trials):
    """Build the prompt for the proposer Claude session."""
    return (
        f"Run iteration {iteration} of the scaffold evolution loop (KIRA track). "
        f"Model: {MODEL} (Opus). "
        f"Start from agents/baseline_kira.py as the parent.\n\n"
        f"## Eval split: {N_EVAL_TASKS} official TB2 tasks x {n_trials} trials\n\n"
        f"This reference example uses the full TB2 dataset from the paper runs. "
        f"Focus on scaffold changes that help the agent solve complex, long-horizon tasks.\n\n"
        f"## Run directories\n"
        f"All logs and results for this run are under `{LOGS_DIR}/`.\n"
        f"- `{LOGS_DIR / 'evolution_summary.jsonl'}` — past results\n"
        f"- `{LOGS_DIR / 'frontier_val.json'}` — frontier\n"
        f"- `{LOGS_DIR / 'reports'}/` — post-eval reports\n"
        f"- Write pending_eval.json to: `{PENDING_EVAL}`"
    )


def fresh_start():
    """Clear proposed agents (keep kira_baseline) and logs for a fresh run."""
    if AGENTS_DIR.exists():
        for f in AGENTS_DIR.iterdir():
            if f.name in (
                "__pycache__",
                "__init__.py",
                "baseline_kira.py",
                "baseline_terminus2.py",
            ):
                continue
            if f.is_dir():
                run_cmd(["rm", "-rf", str(f)])
                print(f"  Cleared {f.name}/")
            elif f.suffix == ".py":
                f.unlink()
                print(f"  Cleared {f.name}")

    for f in [EVOLUTION_SUMMARY, FRONTIER_VAL, PENDING_EVAL]:
        if f.exists():
            f.unlink()

    print("  Fresh start: cleared generated agents/ files and log files")


def run_evolve(args):
    global JOBS_DIR, LOGS_DIR, PENDING_EVAL, FRONTIER_VAL, EVOLUTION_SUMMARY

    # Isolate run outputs under run-name subdirs
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    JOBS_DIR = EVOLVE_DIR / "jobs" / run_name
    LOGS_DIR = EVOLVE_DIR / "logs" / run_name
    PENDING_EVAL = LOGS_DIR / "pending_eval.json"
    FRONTIER_VAL = LOGS_DIR / "frontier_val.json"
    EVOLUTION_SUMMARY = LOGS_DIR / "evolution_summary.jsonl"

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.fresh:
        fresh_start()

    n_tasks = N_EVAL_TASKS
    print(
        f"{_ts()} {_bold('Evolution (KIRA track)')}  run={_cyan(run_name)}  model={_cyan(MODEL)}  iters={args.iterations}  trials={args.trials}  tasks={n_tasks}"
    )

    # ── Phase 0: Baselines ─────────────────────────────────────
    baseline_dirs = {}
    if not args.skip_baseline:
        print(
            f"\n{_ts()} {_bold('Phase 0: Baselines')}  agents={len(BASELINES)}  trials={args.trials}"
        )

    for bl_name, bl_import in BASELINES:
        bl_job = f"{bl_name}-t{args.trials}"
        bl_dir = JOBS_DIR / bl_job
        baseline_dirs[bl_name] = bl_dir

        if not args.skip_baseline:
            cached_ok = False
            if bl_dir.exists() and parse_job_results(bl_dir):
                # Validate cached baseline matches current config
                cfg_file = bl_dir / "config.json"
                if cfg_file.exists():
                    try:
                        cfg = json.loads(cfg_file.read_text())
                        cfg_model = cfg.get("model") or cfg.get("agent", {}).get(
                            "model", ""
                        )
                        cfg_attempts = cfg.get("n_attempts", 0)
                        if MODEL not in str(cfg_model):
                            print(
                                f"  {_yellow('stale')} {bl_name}: model mismatch (cached={cfg_model}, want={MODEL}), re-running"
                            )
                        elif cfg_attempts != args.trials:
                            print(
                                f"  {_yellow('stale')} {bl_name}: trials mismatch (cached={cfg_attempts}, want={args.trials}), re-running"
                            )
                        else:
                            cached_ok = True
                    except (json.JSONDecodeError, OSError):
                        pass
                else:
                    cached_ok = True  # no config to check, trust it
            if cached_ok:
                print(f"  {_dim('cached')} {bl_name}: {bl_dir}")
            else:
                print(
                    f"  {_ts()} running {_bold(bl_name)}: {n_tasks} tasks x {args.trials} trials...",
                    flush=True,
                )
                t0 = time.time()
                bl_dir, ok = harbor_run(
                    bl_import,
                    bl_job,
                    n_trials=args.trials,
                    n_concurrent=args.concurrent,
                )
                baseline_dirs[bl_name] = bl_dir
                elapsed = time.time() - t0
                if not ok:
                    print(
                        f"  {_red('FAIL')} {bl_name} crashed after {_elapsed(elapsed)}"
                    )
                else:
                    print(f"  {_ts()} {bl_name} completed in {_elapsed(elapsed)}")

    # Always seed frontier from baselines if results exist (even with --skip-baseline)
    for bl_name, bl_dir in baseline_dirs.items():
        if bl_dir.exists():
            task_rewards = parse_job_results(bl_dir, expected_trials=args.trials)
            if task_rewards:
                per_task, avg = compute_pass_rates(task_rewards)
                update_frontier({bl_name: (per_task, avg)})
                if not args.skip_baseline:
                    print(f"  {_bold(bl_name)}: avg={_rate_str(avg)}")
                    max_name = max(len(t) for t in per_task) if per_task else 0
                    for task, rate in sorted(per_task.items()):
                        print(f"    {task:<{max_name}}  {_rate_str(rate)}")

    # ── Phase 1..N: Evolution ──────────────────────────────────
    start_iteration = count_iterations() + 1
    for i in range(args.iterations):
        if _interrupted:
            print("Interrupted.")
            break

        iteration = start_iteration + i
        iter_start = time.time()
        frontier = json.loads(FRONTIER_VAL.read_text()) if FRONTIER_VAL.exists() else {}
        best_avg = frontier.get("_best", {}).get("avg_pass_rate", 0)
        best_agent = frontier.get("_best", {}).get("agent", "none")
        print(
            f"\n{_ts()} {_bold(f'Iteration {iteration}')} ({i + 1}/{args.iterations})  frontier={best_agent} @ {best_avg:.1%}"
        )
        print(f"{'─' * 60}")

        # Propose
        if PENDING_EVAL.exists():
            PENDING_EVAL.unlink()

        propose_start = time.time()
        task_prompt = render_task_prompt(iteration, args.trials)
        print(f"  {_ts()} {_cyan('proposing')} new candidates...", flush=True)
        ok = propose_claude(task_prompt, iteration, timeout=args.propose_timeout)
        propose_time = time.time() - propose_start

        if not ok:
            print(
                f"  {_red('FAIL')} proposer returned no candidates after {_elapsed(propose_time)}"
            )
            continue

        candidates = json.loads(PENDING_EVAL.read_text()).get("candidates", [])
        # Normalize class name: always AgentHarness regardless of what proposer wrote
        for c in candidates:
            if "import_path" in c and ":" in c["import_path"]:
                module, _ = c["import_path"].rsplit(":", 1)
                c["import_path"] = f"{module}:AgentHarness"
        print(
            f"  {_ts()} proposed {len(candidates)} candidate(s) in {_elapsed(propose_time)}"
        )
        for ci, c in enumerate(candidates):
            hyp = c.get("hypothesis", "")
            print(f"    {ci + 1}. {_bold(c['name'])}: {hyp[:80]}")

        # Validate
        valid = []
        print(f"  {_ts()} {_cyan('validating')} {len(candidates)} candidate(s)...")
        for ci, c in enumerate(candidates):
            name = c["name"]
            import_path = c["import_path"]
            prefix = f"    [{ci + 1}/{len(candidates)}] {name}:"
            if validate_candidate(name, import_path):
                if args.skip_smoke:
                    print(f"{prefix} {_green('import OK')} (smoke skipped)")
                    valid.append(c)
                elif smoke_test(name, import_path):
                    print(f"{prefix} {_green('import OK + smoke OK')}")
                    valid.append(c)
                else:
                    print(f"{prefix} {_red('smoke FAIL')}")
            else:
                print(f"{prefix} {_red('import FAIL')}")
            if _interrupted:
                break

        if not valid:
            print(
                f"  {_red('0 valid')} out of {len(candidates)} candidates, skipping iteration"
            )
            update_evolution_summary(
                iteration,
                candidates,
                {},
                previous_best=best_avg,
                propose_time=propose_time,
            )
            continue
        print(f"  {_green(f'{len(valid)} valid')} out of {len(candidates)} candidates")

        # Benchmark
        bench_start = time.time()
        results = {}
        all_metrics = {}
        n_evals = len(valid) * n_tasks * args.trials
        print(
            f"  {_ts()} {_cyan('benchmarking')} {len(valid)} agent(s) x {n_tasks} tasks x {args.trials} trials = {n_evals} evals"
        )
        for ci, c in enumerate(valid):
            if _interrupted:
                break
            name = c["name"]
            import_path = c["import_path"]
            job_name = f"evolve-{name}-t{args.trials}"

            print(f"    [{ci + 1}/{len(valid)}] {_bold(name)}...", flush=True)
            t0 = time.time()
            job_dir, job_result = harbor_run(
                import_path,
                job_name,
                n_trials=args.trials,
                n_concurrent=args.concurrent,
            )
            elapsed = time.time() - t0
            if job_result:
                task_rewards = parse_job_results(job_dir, expected_trials=args.trials)
                per_task, avg = compute_pass_rates(task_rewards)
                results[name] = (per_task, avg)
                delta = avg - best_avg
                delta_str = f"{delta:+.1%}"
                delta_colored = (
                    _green(delta_str)
                    if delta > 0
                    else (_red(delta_str) if delta < 0 else _dim(delta_str))
                )

                trial_metrics = parse_trial_metrics(job_dir)
                metrics_summary = summarize_trial_metrics(trial_metrics)
                all_metrics[name] = metrics_summary
                cost_str = (
                    f"${metrics_summary['total_cost_usd']:.2f}"
                    if metrics_summary["total_cost_usd"]
                    else "?"
                )
                print(
                    f"         avg={_rate_str(avg)}  delta={delta_colored}  cost={cost_str}  ({_elapsed(elapsed)})"
                )
                max_name_len = max(len(t) for t in per_task) if per_task else 0
                baseline_per_task = {}
                if FRONTIER_VAL.exists():
                    fr = json.loads(FRONTIER_VAL.read_text())
                    for tk in per_task:
                        baseline_per_task[tk] = fr.get(tk, {}).get("pass_rate", 0)
                for task, rate in sorted(per_task.items()):
                    bl = baseline_per_task.get(task, 0)
                    td = rate - bl
                    td_s = f"{td:+.0%}" if td != 0 else "  ="
                    td_c = (
                        _green(td_s)
                        if td > 0
                        else (_red(td_s) if td < 0 else _dim(td_s))
                    )
                    tm = metrics_summary.get("per_task", {}).get(task, {})
                    tc = (
                        f"${tm['mean_cost']:.2f}"
                        if tm.get("mean_cost") is not None
                        else ""
                    )
                    tt = (
                        f"{tm['mean_turns']:.0f}t"
                        if tm.get("mean_turns") is not None
                        else ""
                    )
                    suffix = f"  {_dim(tc)} {_dim(tt)}" if tc else ""
                    print(
                        f"         {task:<{max_name_len}}  {_rate_str(rate)}  {td_c}{suffix}"
                    )
            else:
                results[name] = ({}, 0)
                print(
                    f"         {_red('FAIL')} benchmark crashed ({_elapsed(elapsed)})"
                )

        bench_time = time.time() - bench_start

        update_frontier(results)
        update_evolution_summary(
            iteration,
            valid,
            results,
            previous_best=best_avg,
            propose_time=propose_time,
            bench_time=bench_time,
            metrics=all_metrics,
        )

        wall_time = time.time() - iter_start
        frontier_now = (
            json.loads(FRONTIER_VAL.read_text()) if FRONTIER_VAL.exists() else {}
        )
        new_best_avg = frontier_now.get("_best", {}).get("avg_pass_rate", 0)
        new_best_agent = frontier_now.get("_best", {}).get("agent", "none")
        improved = new_best_avg > best_avg
        status = _green("NEW BEST") if improved else _dim("no improvement")
        print(f"  {_ts()} {status}  frontier={new_best_agent} @ {new_best_avg:.1%}")
        print(
            f"  {_dim(f'timing: propose={_elapsed(propose_time)} bench={_elapsed(bench_time)} total={_elapsed(wall_time)}')}"
        )

    # ── Phase Final: Winners get 5-trial eval on the full dataset ────────────
    if _interrupted or not args.full_eval:
        return

    print(f"\n{_ts()} {_bold('Phase Final: 5-trial eval for frontier agents')}")
    frontier = json.loads(FRONTIER_VAL.read_text()) if FRONTIER_VAL.exists() else {}
    best_agent = frontier.get("_best", {}).get("agent")
    if best_agent and best_agent != BASELINE_AGENT_NAME:
        import_path = None
        for line in EVOLUTION_SUMMARY.read_text().strip().split("\n"):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("agent") == best_agent and row.get("import_path"):
                import_path = row["import_path"]
                break

        if import_path:
            job_name = f"final-{best_agent}-t5"
            print(f"  {_ts()} running {_bold(best_agent)} x 5 trials...", flush=True)
            t0 = time.time()
            job_dir, _ = harbor_run(
                import_path,
                job_name,
                n_trials=5,
                n_concurrent=args.concurrent,
            )
            if job_dir.exists():
                task_rewards = parse_job_results(job_dir, expected_trials=5)
                per_task, avg = compute_pass_rates(task_rewards)
                print(
                    f"  {_bold(best_agent)} (5-trial): avg={_rate_str(avg)}  ({_elapsed(time.time() - t0)})"
                )
                max_name_len = max(len(t) for t in per_task) if per_task else 0
                for task, rate in sorted(per_task.items()):
                    print(f"    {task:<{max_name_len}}  {_rate_str(rate)}")

    print(f"\n{_ts()} {_bold('Evolution complete.')}")


def main():
    parser = argparse.ArgumentParser(
        description="Scaffold evolution loop for Terminal-Bench (KIRA track)"
    )
    parser.add_argument(
        "--iterations", type=int, default=5, help="Number of evolution iterations"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_SEARCH_TRIALS,
        help="Trials per task during evolution (default: 2)",
    )
    parser.add_argument(
        "--propose-timeout",
        type=int,
        default=2400,
        help="Timeout for proposer (seconds)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name for isolated output dirs (jobs/<run>/*, logs/<run>/*). Auto-generated if not set.",
    )
    parser.add_argument(
        "--fresh", action="store_true", help="Clear proposed agents and reset logs"
    )
    parser.add_argument(
        "--skip-baseline", action="store_true", help="Skip Phase 0 baseline eval"
    )
    parser.add_argument("--skip-smoke", action="store_true", help="Skip smoke tests")
    parser.add_argument(
        "--full-eval",
        action="store_true",
        help="Run the optional 5-trial winner eval on the full dataset",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Max concurrent trials (default: 50)",
    )
    parser.add_argument(
        "--validate-agent",
        metavar="MODULE:CLASS",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.validate_agent:
        try:
            agent_class = validate_agent_class(args.validate_agent)
        except Exception as exc:
            print(f"Invalid agent {args.validate_agent}: {exc}", file=sys.stderr)
            return 1
        print(f"OK: {agent_class.__module__}:{agent_class.__name__}")
        return 0

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    run_evolve(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
