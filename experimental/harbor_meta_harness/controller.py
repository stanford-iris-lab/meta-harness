"""Short-lived, trusted evaluator for generated Harbor harnesses.

Candidate code is never imported by the outer agent/controller.  This process
only stages it and launches Harbor child processes, where Harbor imports it.
"""

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).parent
DEFAULT_SUITE = ROOT / "suite.toml"
UNIVERSAL_FORBIDDEN = (
    "/tests",
    "test_outputs",
    "verifier",
    "/solution",
    "task.toml",
)

Aggregator = Callable[[Sequence[float]], float]


def aggregate_mean(rewards: Sequence[float]) -> float:
    return sum(rewards) / len(rewards)


def aggregate_min(rewards: Sequence[float]) -> float:
    return min(rewards)


def aggregate_fraction_solved(rewards: Sequence[float]) -> float:
    return sum(1.0 for reward in rewards if reward >= 1.0) / len(rewards)


AGGREGATORS: dict[str, Aggregator] = {
    "mean": aggregate_mean,
    "min": aggregate_min,
    "fraction_solved": aggregate_fraction_solved,
}


@dataclass(frozen=True)
class SuiteConfig:
    tasks: tuple[Path, ...]
    child_model: str
    environment: str
    n_attempts: int
    eval_budget: int
    seed_harness: Path
    aggregate: str
    child_timeout_sec: int

    @property
    def aggregator(self) -> Aggregator:
        try:
            return AGGREGATORS[self.aggregate]
        except KeyError as error:
            known = ", ".join(sorted(AGGREGATORS))
            raise ValueError(
                f"unknown aggregate {self.aggregate!r}; choose from {known}"
            ) from error


@dataclass(frozen=True)
class ChildResult:
    reward: float
    summary: str
    job_dir: str


@dataclass(frozen=True)
class TaskResult:
    task: str
    reward: float
    summary: str
    job_dir: str


@dataclass(frozen=True)
class EvaluationResult:
    source_sha256: str
    accepted: bool
    reason: str
    reward: float | None
    tasks: tuple[TaskResult, ...]


def load_suite(path: Path = DEFAULT_SUITE) -> SuiteConfig:
    path = path.resolve()
    raw = tomllib.loads(path.read_text())
    # Task paths in suite files are always relative to the experiment root.
    tasks = tuple((ROOT / task).resolve() for task in raw["tasks"])
    if not tasks:
        raise ValueError("suite.tasks must be a non-empty list")
    for task in tasks:
        if not (task / "task.toml").exists():
            raise FileNotFoundError(f"missing Harbor task.toml: {task}")
    aggregate = raw.get("aggregate", "mean")
    if aggregate not in AGGREGATORS:
        known = ", ".join(sorted(AGGREGATORS))
        raise ValueError(f"unknown aggregate {aggregate!r}; choose from {known}")
    return SuiteConfig(
        tasks=tasks,
        child_model=raw.get("child_model", "openai/gpt-5.4-nano"),
        environment=raw.get("environment", "modal"),
        n_attempts=int(raw.get("n_attempts", 1)),
        eval_budget=int(raw.get("eval_budget", 4)),
        seed_harness=(
            ROOT
            / raw.get("seed_harness", "tasks/select-harness/environment/harness.py")
        ).resolve(),
        aggregate=aggregate,
        child_timeout_sec=int(raw.get("child_timeout_sec", 600)),
    )


def task_forbidden_references(task: Path) -> tuple[str, ...]:
    meta = tomllib.loads((task / "task.toml").read_text()).get("meta_harness") or {}
    values = meta.get("forbidden_references") or []
    return tuple(str(value) for value in values)


def suite_forbidden_references(suite: SuiteConfig) -> tuple[str, ...]:
    forbidden: list[str] = list(UNIVERSAL_FORBIDDEN)
    seen = set(forbidden)
    for task in suite.tasks:
        for value in task_forbidden_references(task):
            if value not in seen:
                seen.add(value)
                forbidden.append(value)
    return tuple(forbidden)


def validate_source(source: str, forbidden: Sequence[str]) -> str | None:
    """Return a rejection reason, or None for the minimal safe interface."""
    lowered = source.lower()
    for item in forbidden:
        if item.lower() in lowered:
            return f"forbidden benchmark reference: {item}"
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return f"syntax error: {error.msg}"
    harness = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "AgentHarness"
        ),
        None,
    )
    if harness is None:
        return "missing AgentHarness class"
    if not any(
        isinstance(node, ast.AsyncFunctionDef) and node.name == "run"
        for node in harness.body
    ):
        return "AgentHarness must define async run"
    if not any(
        (isinstance(base, ast.Name) and base.id == "BaseAgent")
        or (isinstance(base, ast.Attribute) and base.attr == "BaseAgent")
        for base in harness.bases
    ):
        return "AgentHarness must inherit BaseAgent"
    return None


def modal_environment() -> dict[str, str]:
    """Copy Modal credentials into the child environment without logging them."""
    environment = os.environ.copy()
    if environment.get("MODAL_TOKEN_ID") and environment.get("MODAL_TOKEN_SECRET"):
        return environment
    config_path = Path.home() / ".modal.toml"
    if not config_path.exists():
        return environment
    config = tomllib.loads(config_path.read_text())
    profile = next(
        (
            value
            for value in config.values()
            if isinstance(value, dict) and value.get("active")
        ),
        next((value for value in config.values() if isinstance(value, dict)), {}),
    )
    if profile.get("token_id") and profile.get("token_secret"):
        environment["MODAL_TOKEN_ID"] = profile["token_id"]
        environment["MODAL_TOKEN_SECRET"] = profile["token_secret"]
    return environment


def command_for(
    suite: SuiteConfig, task: Path, candidate_dir: Path, jobs_dir: Path, name: str
) -> list[str]:
    return [
        str(Path(sys.executable).with_name("harbor")),
        "run",
        "-p",
        str(task),
        "--agent",
        "candidate:AgentHarness",
        "-e",
        suite.environment,
        "-m",
        suite.child_model,
        "-n",
        "1",
        "--n-attempts",
        str(suite.n_attempts),
        "--job-name",
        name,
        "--jobs-dir",
        str(jobs_dir),
    ]


def child_result(job_dir: Path, expected_attempts: int = 1) -> ChildResult:
    """Average all expected attempts, counting missing or malformed ones as zero."""
    if expected_attempts < 1:
        raise ValueError("expected_attempts must be positive")
    result_files = sorted(job_dir.glob("*/result.json"))
    if not result_files:
        return ChildResult(0.0, "missing Harbor result", str(job_dir))
    if len(result_files) > expected_attempts:
        return ChildResult(
            0.0,
            f"found {len(result_files)} Harbor results; expected {expected_attempts}",
            str(job_dir),
        )
    rewards: list[float] = []
    summaries: list[str] = []
    for result_file in result_files:
        try:
            result = json.loads(result_file.read_text())
            verifier = result["verifier_result"] or {}
            reward = float(verifier["rewards"]["reward"])
            summary = json.dumps(verifier, sort_keys=True)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            reward = 0.0
            summary = "malformed Harbor result"
        rewards.append(reward)
        summaries.append(summary)
    if expected_attempts == 1:
        summary = summaries[0]
    else:
        summary = json.dumps(
            {
                "attempts": summaries,
                "missing_attempts": expected_attempts - len(result_files),
            },
            sort_keys=True,
        )
    return ChildResult(sum(rewards) / expected_attempts, summary, str(job_dir))


def run_child(
    suite: SuiteConfig, task: Path, candidate_dir: Path, jobs_dir: Path, name: str
) -> ChildResult:
    job_dir = jobs_dir / name
    environment = modal_environment()
    environment["PYTHONPATH"] = (
        str(candidate_dir) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    try:
        completed = subprocess.run(
            command_for(suite, task, candidate_dir, jobs_dir, name),
            cwd=ROOT,
            env=environment,
            check=False,
            timeout=suite.child_timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return ChildResult(
            0.0, f"timed out after {suite.child_timeout_sec}s", str(job_dir)
        )
    if completed.returncode != 0:
        return ChildResult(0.0, f"Harbor exited {completed.returncode}", str(job_dir))
    return child_result(job_dir, expected_attempts=suite.n_attempts)


def evaluate_source(
    source_path: Path, jobs_dir: Path, suite: SuiteConfig | None = None
) -> EvaluationResult:
    suite = suite or load_suite()
    source_path = source_path.resolve()
    jobs_dir = jobs_dir.resolve()
    source = source_path.read_text()
    source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    reason = validate_source(source, suite_forbidden_references(suite))
    if reason:
        return EvaluationResult(source_sha256, False, reason, None, ())
    jobs_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="harbor-harness-") as temp_dir:
        candidate_dir = Path(temp_dir)
        (candidate_dir / "candidate.py").write_text(source)
        task_results: list[TaskResult] = []
        for task in suite.tasks:
            child = run_child(suite, task, candidate_dir, jobs_dir, task.name)
            task_results.append(
                TaskResult(task.name, child.reward, child.summary, child.job_dir)
            )
    rewards = [result.reward for result in task_results]
    return EvaluationResult(
        source_sha256,
        True,
        "scored",
        suite.aggregator(rewards),
        tuple(task_results),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--jobs-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    args = parser.parse_args()
    result = evaluate_source(args.source, args.jobs_dir, load_suite(args.suite))
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(asdict(result), indent=2) + "\n")


if __name__ == "__main__":
    main()
