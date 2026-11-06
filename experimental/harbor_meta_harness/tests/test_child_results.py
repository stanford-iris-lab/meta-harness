import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import controller
import pytest


def _result(job_dir: Path, trial: str, reward: float) -> Path:
    path = job_dir / trial / "result.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"verifier_result": {"rewards": {"reward": reward}}}))
    return path


def test_single_attempt_preserves_result(tmp_path: Path) -> None:
    _result(tmp_path, "trial", 0.75)

    result = controller.child_result(tmp_path)

    assert result == controller.ChildResult(
        0.75, '{"rewards": {"reward": 0.75}}', str(tmp_path)
    )


@pytest.mark.parametrize(
    ("rewards", "expected"),
    [([1.0, 1.0], 1.0), ([1.0, 0.0], 0.5), ([0.25, 0.75, 0.5], 0.5)],
)
def test_multiple_attempts_are_averaged(
    tmp_path: Path, rewards: list[float], expected: float
) -> None:
    for index, reward in enumerate(rewards):
        _result(tmp_path, f"trial-{index}", reward)

    result = controller.child_result(tmp_path, expected_attempts=len(rewards))

    assert result.reward == expected
    assert result.job_dir == str(tmp_path)


def test_missing_attempts_do_not_inflate_reward(tmp_path: Path) -> None:
    _result(tmp_path, "trial", 1.0)

    result = controller.child_result(tmp_path, expected_attempts=2)

    assert result.reward == 0.5
    assert "missing" in result.summary


@pytest.mark.parametrize("expected_attempts", [1, 3])
def test_no_results_score_zero(tmp_path: Path, expected_attempts: int) -> None:
    result = controller.child_result(tmp_path, expected_attempts=expected_attempts)

    assert result.reward == 0
    assert result.summary == "missing Harbor result"


@pytest.mark.parametrize(
    "contents",
    [
        "{",
        "[]",
        "null",
        "{}",
        '{"verifier_result": null}',
        '{"verifier_result": {"rewards": {"reward": "invalid"}}}',
    ],
)
def test_malformed_attempt_counts_as_zero(tmp_path: Path, contents: str) -> None:
    _result(tmp_path, "good", 1.0)
    _result(tmp_path, "bad", 0.0).write_text(contents)

    result = controller.child_result(tmp_path, expected_attempts=2)

    assert result.reward == 0.5
    assert "malformed Harbor result" in result.summary


def test_single_malformed_attempt_preserves_error(tmp_path: Path) -> None:
    _result(tmp_path, "trial", 0.0).write_text("{")

    assert controller.child_result(tmp_path) == controller.ChildResult(
        0.0, "malformed Harbor result", str(tmp_path)
    )


def test_extra_results_are_rejected(tmp_path: Path) -> None:
    for index in range(3):
        _result(tmp_path, f"trial-{index}", 1.0)

    result = controller.child_result(tmp_path, expected_attempts=2)

    assert result.reward == 0
    assert "expected 2" in result.summary


@pytest.mark.parametrize("expected_attempts", [0, -1])
def test_attempt_count_must_be_positive(tmp_path: Path, expected_attempts: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        controller.child_result(tmp_path, expected_attempts=expected_attempts)


@pytest.mark.skipif(os.name != "posix", reason="uses a POSIX executable stub")
def test_controller_passes_attempt_count_to_result_reader(tmp_path: Path) -> None:
    # Exercise the real CLI, subprocess launch, and result reader. Only the
    # external Harbor executable is replaced; no model or cloud calls are made.
    python = tmp_path / "python"
    python.symlink_to(sys.executable)
    harbor = tmp_path / "harbor"
    harbor.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent("""\
        import json
        import sys
        from pathlib import Path

        args = sys.argv
        attempts = int(args[args.index("--n-attempts") + 1])
        assert attempts == 2
        job = Path(args[args.index("--jobs-dir") + 1])
        job /= args[args.index("--job-name") + 1]
        for index in range(attempts):
            trial = job / f"trial-{index}"
            trial.mkdir(parents=True)
            result = {"verifier_result": {"rewards": {"reward": float(index)}}}
            (trial / "result.json").write_text(json.dumps(result))
    """)
    )
    harbor.chmod(0o755)
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text("")
    suite = tmp_path / "suite.toml"
    suite.write_text(f"tasks = [{json.dumps(str(task))}]\nn_attempts = 2\n")
    source = tmp_path / "candidate.py"
    source.write_text(
        "class AgentHarness(BaseAgent):\n"
        "    async def run(self, instruction, environment, context):\n"
        "        pass\n"
    )
    result_path = tmp_path / "evaluation.json"
    completed = subprocess.run(
        [
            str(python),
            str(controller.ROOT / "controller.py"),
            "--source",
            str(source),
            "--suite",
            str(suite),
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--result",
            str(result_path),
        ],
        env={**os.environ, "MODAL_TOKEN_ID": "test", "MODAL_TOKEN_SECRET": "test"},
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(result_path.read_text())
    assert result["accepted"]
    assert result["reward"] == 0.5
    assert result["tasks"][0]["reward"] == 0.5
