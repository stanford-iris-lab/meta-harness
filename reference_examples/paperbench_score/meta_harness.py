from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AGENTS_DIR = ROOT / "agents"
LOGS_DIR = ROOT / "logs"
PENDING_EVAL = LOGS_DIR / "pending_eval.json"
FRONTIER_VAL = LOGS_DIR / "frontier_val.json"
EVOLUTION_SUMMARY = LOGS_DIR / "evolution_summary.jsonl"
BASELINE_AGENT = "baseline_basic_agent"
PAPER_ID = "sequential-neural-score-estimation"
PAPERBENCH_ROOT = os.environ.get(
    "PAPERBENCH_ROOT",
    "/Users/emmett/Projects/research/frontier-evals/project/paperbench",
)

sys.path.insert(0, str(ROOT.parent / "terminal_bench_2"))
import claude_wrapper  # noqa: E402

PROPOSER_ALLOWED_TOOLS = ["Read", "Glob", "Grep", "Agent", "Write", "Edit"]
_interrupted = False


def _handle_signal(signum, frame) -> None:
    del signum, frame
    global _interrupted
    _interrupted = True
    print("\nInterrupted; finishing current step...", flush=True)


def _run(cmd: list[str], timeout: int = 120, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, cwd=str(cwd), timeout=timeout, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, "", f"Timed out after {timeout}s")


def _set_run_paths(run_name: str) -> None:
    global LOGS_DIR, PENDING_EVAL, FRONTIER_VAL, EVOLUTION_SUMMARY
    LOGS_DIR = ROOT / "logs" / run_name
    PENDING_EVAL = LOGS_DIR / "pending_eval.json"
    FRONTIER_VAL = LOGS_DIR / "frontier_val.json"
    EVOLUTION_SUMMARY = LOGS_DIR / "evolution_summary.jsonl"


def fresh_start() -> None:
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    for path in AGENTS_DIR.glob("*.py"):
        if path.name in {"__init__.py", f"{BASELINE_AGENT}.py"}:
            continue
        path.unlink()
    for path in [PENDING_EVAL, FRONTIER_VAL, EVOLUTION_SUMMARY]:
        if path.exists():
            path.unlink()


def validate_candidate(name: str) -> bool:
    agent_file = AGENTS_DIR / f"{name}.py"
    if not agent_file.exists():
        print(f"  invalid {name}: missing {agent_file}")
        return False
    cmd = [
        "uv",
        "run",
        "--project",
        PAPERBENCH_ROOT,
        "python",
        "-c",
        (
            "import sys; "
            f"sys.path.insert(0, {str(ROOT)!r}); "
            f"from agents.{name} import BasicAgentSolver; "
            "print(BasicAgentSolver().shortname())"
        ),
    ]
    result = _run(cmd)
    if result.returncode == 0:
        return True
    print(f"  invalid {name}: {(result.stderr or result.stdout)[:500]}")
    return False


def evaluate_candidate(
    name: str,
    run_name: str,
    agent_time_limit: int,
    reproduction_timeout: int,
    gpu: str,
) -> dict:
    cmd = [
        sys.executable,
        str(ROOT / "benchmark.py"),
        "--agent",
        name,
        "--run-name",
        run_name,
        "--paper-id",
        PAPER_ID,
        "--agent-time-limit",
        str(agent_time_limit),
        "--reproduction-timeout",
        str(reproduction_timeout),
        "--gpu",
        gpu,
    ]
    eval_log = LOGS_DIR / name / "benchmark_stdout.log"
    eval_log.parent.mkdir(parents=True, exist_ok=True)
    with eval_log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
    result_path = LOGS_DIR / name / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
    else:
        result = {"score": 0.0, "error": "missing benchmark result.json", "returncode": proc.returncode}
    result["benchmark_stdout_log"] = str(eval_log)
    return result


def update_frontier(result: dict) -> None:
    score = float(result.get("score") or 0.0)
    frontier = json.loads(FRONTIER_VAL.read_text()) if FRONTIER_VAL.exists() else {}
    best = frontier.get("_best", {})
    if score >= float(best.get("score") or -1.0):
        frontier["_best"] = {
            "agent": result.get("agent"),
            "score": score,
            "feedback": result.get("feedback", ""),
            "result_path": str(LOGS_DIR / result.get("agent", "") / "result.json"),
        }
        FRONTIER_VAL.write_text(json.dumps(frontier, indent=2, sort_keys=True))


def append_summary(iteration: int, candidate: dict, result: dict) -> None:
    row = {
        "iteration": iteration,
        "agent": result.get("agent") or candidate.get("name"),
        "score": float(result.get("score") or 0.0),
        "axis": candidate.get("axis", "?"),
        "hypothesis": candidate.get("hypothesis", ""),
        "feedback": result.get("feedback", ""),
        "log_path": result.get("log_path"),
        "error": result.get("error"),
    }
    with EVOLUTION_SUMMARY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def propose(iteration: int, timeout: int, batch_size: int) -> list[dict]:
    if PENDING_EVAL.exists():
        PENDING_EVAL.unlink()
    frontier = FRONTIER_VAL.read_text() if FRONTIER_VAL.exists() else "{}"
    summary = EVOLUTION_SUMMARY.read_text() if EVOLUTION_SUMMARY.exists() else ""
    prompt = f"""Run PaperBench-score Meta-Harness iteration {iteration}.

Goal: propose {batch_size} diverse candidate agent files that improve the real PaperBench overall score on `{PAPER_ID}`.

Candidate interface:
- Write a Python file under `agents/`.
- It must export `BasicAgentSolver`.
- Start from `agents/{BASELINE_AGENT}.py` or subclass it.
- Do not edit `benchmark.py`, `modal_eval.py`, the judge, or PaperBench scoring.
- Be conservative: preserve baseline behavior unless your change directly targets a proven failure.
- Do not propose a candidate that could make the agent omit `reproduce.sh`. A submitted repo without
  `reproduce.sh` gets zero even if it creates other files.
- The original baseline to beat is:
  Overall: 0.6634
  Code Development: 70/81
  Code Execution: 1/7
  Result Analysis: 0/20
- A prior meta candidate `exec_gate_basic_agent` failed badly: Overall 0.0000 because the submitted
  repository contained `src/`, `results/`, and `figures/` but no `/submission/reproduce.sh`.
  Treat this as negative feedback: do not add overbearing instructions that distract the agent from
  creating and committing the required reproduction script.

Feedback from prior runs:
<frontier>
{frontier}
</frontier>

<evolution_summary>
{summary[-8000:]}
</evolution_summary>

Write pending eval JSON to `{PENDING_EVAL}` with exactly {batch_size} entries:
{{"candidates": [{{"name": "<file_stem>", "hypothesis": "...", "axis": "..."}}]}}
"""
    result = claude_wrapper.run(
        prompt=prompt,
        model="opus",
        allowed_tools=PROPOSER_ALLOWED_TOOLS,
        skills=[str(ROOT / ".claude/skills/meta-harness-paperbench-score")],
        cwd=str(ROOT),
        log_dir=str(LOGS_DIR / "claude_sessions"),
        name=f"iter{iteration}",
        timeout_seconds=timeout,
        effort="max",
    )
    result.show()
    if result.exit_code != 0 or not PENDING_EVAL.exists():
        return []
    return json.loads(PENDING_EVAL.read_text()).get("candidates", [])[:batch_size]


def evaluate_batch(iteration: int, candidates: list[dict], args: argparse.Namespace, run_name: str) -> list[tuple[dict, dict]]:
    results: list[tuple[dict, dict]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.parallelism)) as executor:
        futures = {
            executor.submit(
                evaluate_candidate,
                candidate["name"],
                run_name,
                args.agent_time_limit,
                args.reproduction_timeout,
                args.gpu,
            ): candidate
            for candidate in candidates
        }
        for future in as_completed(futures):
            candidate = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"agent": candidate["name"], "score": 0.0, "error": repr(exc)}
            print(f"Finished {candidate['name']}: score={float(result.get('score') or 0.0):.4f}", flush=True)
            if result.get("feedback"):
                print(result["feedback"], flush=True)
            results.append((candidate, result))
    batch_path = LOGS_DIR / f"batch_results_iter{iteration}.json"
    batch_path.write_text(json.dumps([{"candidate": c, "result": r} for c, r in results], indent=2, sort_keys=True))
    return results


def run(args: argparse.Namespace) -> None:
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    _set_run_paths(run_name)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        fresh_start()

    print(f"PaperBench score Meta-Harness run={run_name} iterations={args.iterations}")
    if not args.skip_baseline:
        print("Evaluating baseline...")
        baseline = {"name": BASELINE_AGENT, "hypothesis": "baseline", "axis": "baseline"}
        result = evaluate_candidate(
            BASELINE_AGENT,
            run_name,
            args.agent_time_limit,
            args.reproduction_timeout,
            args.gpu,
        )
        print(result.get("feedback", result))
        update_frontier(result)
        append_summary(0, baseline, result)

    for iteration in range(1, args.iterations + 1):
        if _interrupted:
            break
        print(f"\nIteration {iteration}")
        candidates = propose(iteration, args.propose_timeout, args.batch_size)
        valid = [c for c in candidates if validate_candidate(c["name"])]
        if not valid:
            print("No valid candidates.")
            continue
        print(f"Evaluating batch of {len(valid)} candidates with parallelism={args.parallelism}")
        batch_results = evaluate_batch(iteration, valid, args, run_name)
        print(f"Committing {len(batch_results)} batch results to frontier storage")
        for candidate, result in batch_results:
            update_frontier(result)
            append_summary(iteration, candidate, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--propose-timeout", type=int, default=2400)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--parallelism", type=int, default=10)
    parser.add_argument("--agent-time-limit", type=int, default=600)
    parser.add_argument("--reproduction-timeout", type=int, default=600)
    parser.add_argument("--gpu", default="A10G")
    args = parser.parse_args()
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    run(args)


if __name__ == "__main__":
    main()
