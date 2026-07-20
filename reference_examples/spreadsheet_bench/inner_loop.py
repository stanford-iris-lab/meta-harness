"""Inner loop: run ONE agent scaffold on ONE SpreadsheetBench instruction and score it.

    uv run python inner_loop.py --agent agents/baseline_react.py --task-id <id>

Resolves the solver model/api like text_classification_local (QNAIGC key from env),
loads the agent's AgentHarness, runs the multi-round loop to produce a solution program,
then OJ-scores that program across all of the instruction's test cases.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import yaml

from data.loader import get_task
from evaluation import score_instruction
from llm import LLM

EVOLVE_DIR = Path(__file__).resolve().parent


def load_config(path: str | None) -> dict:
    cfg_path = Path(path) if path else EVOLVE_DIR / "config.yaml"
    return yaml.safe_load(cfg_path.read_text())


def build_llm(cfg: dict, model_override=None, api_base_override=None) -> LLM:
    mcfg = cfg.get("model", {})
    model = model_override or mcfg.get("name", "gpt-oss-120b")
    api_base = api_base_override if api_base_override is not None else mcfg.get("api_base")
    # Cloud-routed model strings carry their own provider; no custom api_base/key.
    if model.startswith(("openrouter/", "gemini/")):
        api_base = None
    api_key = None
    if api_base is not None:
        # litellm defaults api_key to "local" for custom api_base, which QNAIGC rejects.
        api_key = os.environ.get("QNAIGC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    temperature = mcfg.get("temperature")
    return LLM(model=model, api_base=api_base, api_key=api_key, temperature=temperature)


def load_agent_class(agent_arg: str):
    """Resolve `agents/foo.py`, `agents.foo`, or `agents.foo:Class` to a class."""
    from agent import SpreadsheetAgent

    spec = agent_arg
    cls_name = None
    if ":" in spec:
        spec, cls_name = spec.split(":", 1)
    if spec.endswith(".py"):
        spec = spec[:-3]
    module_name = spec.replace("/", ".").replace("\\", ".")
    module = importlib.import_module(module_name)

    if cls_name and hasattr(module, cls_name):
        return getattr(module, cls_name)
    if hasattr(module, "AgentHarness"):
        return module.AgentHarness
    for v in vars(module).values():
        if (
            isinstance(v, type)
            and issubclass(v, SpreadsheetAgent)
            and v is not SpreadsheetAgent
        ):
            return v
    raise ImportError(f"No AgentHarness/SpreadsheetAgent subclass in {module_name}")


def run_one(args) -> dict:
    cfg = load_config(args.config)
    source = args.source or cfg.get("dataset", {}).get("source", "sample_data_200")
    eval_cfg = cfg.get("eval", {})
    agent_cfg = cfg.get("agent", {})

    task = get_task(source, args.task_id)
    agent_cls = load_agent_class(args.agent)
    llm = build_llm(cfg, args.model, args.api_base)
    agent = agent_cls(llm, config=agent_cfg, python_exe=sys.executable)

    workdir = Path(args.workdir) if args.workdir else Path(
        tempfile.mkdtemp(prefix=f"sb_{args.task_id}_")
    )
    t0 = time.time()
    error = None
    solve_out = {"solution_code": "", "n_turns": 0, "produced": False, "trajectory": []}
    try:
        solve_out = agent.solve(task, workdir / "agent")
        score = score_instruction(
            solve_out["solution_code"],
            task,
            workdir / "score",
            timeout=int(agent_cfg.get("code_timeout_s", 60)),
            tol=float(eval_cfg.get("numeric_tol", 1e-6)),
            metric=eval_cfg.get("metric", "hard"),
            backend=eval_cfg.get("backend", "value"),
            python_exe=sys.executable,
        )
    except Exception:
        error = traceback.format_exc()
        score = {"passed": False, "score": 0.0, "n_cases": len(task.test_cases),
                 "n_pass": 0, "per_case": []}
    runtime = time.time() - t0
    usage = llm.get_usage()

    result = {
        "task_id": task.id,
        "agent": args.agent,
        "instruction_type": task.instruction_type,
        "passed": score["passed"],
        "score": score["score"],
        "n_cases": score["n_cases"],
        "n_pass": score["n_pass"],
        "per_case": score["per_case"],
        "n_turns": solve_out["n_turns"],
        "produced": solve_out["produced"],
        "llm_calls": usage["calls"],
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
        "cost_usd": usage["estimated_cost_usd"],
        "runtime_s": round(runtime, 1),
        "error": error,
        "solution_code": solve_out["solution_code"],
    }

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        with open(args.log, "w") as f:
            for m in solve_out["trajectory"]:
                f.write(json.dumps({"type": "message", **m}) + "\n")
            f.write(json.dumps({"type": "score", **score}) + "\n")
            if error:
                f.write(json.dumps({"type": "error", "traceback": error}) + "\n")

    if not args.keep_workdir and not args.workdir:
        shutil.rmtree(workdir, ignore_errors=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="SpreadsheetBench inner loop (one task)")
    ap.add_argument("--agent", required=True, help="agents/foo.py | agents.foo[:Class]")
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--source", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="write per-task result JSON here")
    ap.add_argument("--log", default=None, help="write trajectory JSONL here")
    ap.add_argument("--workdir", default=None, help="persist workdir here (else temp)")
    ap.add_argument("--keep-workdir", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--api-base", default=None)
    args = ap.parse_args()

    result = run_one(args)
    status = "PASS" if result["passed"] else "fail"
    print(
        f"[{status}] {result['task_id']}  {result['n_pass']}/{result['n_cases']} cases  "
        f"turns={result['n_turns']}  tokens={result['total_tokens']}  "
        f"{result['runtime_s']}s"
    )
    if result["error"]:
        print(result["error"][-800:], file=sys.stderr)


if __name__ == "__main__":
    main()
