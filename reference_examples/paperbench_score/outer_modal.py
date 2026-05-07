from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import modal
from modal.config import config as modal_config

os.environ.setdefault("MODAL_IMAGE_BUILDER_VERSION", "2025.06")

APP = modal.App("paperbench-score-meta-harness-outer")
ROOT = Path(__file__).resolve().parent
PAPERS_ROOT = Path.home() / "Projects" / "papers"
FRONTIER_ROOT = Path.home() / "Projects" / "research" / "frontier-evals"
LOG_VOLUME = modal.Volume.from_name("paperbench-score-meta-harness-logs", create_if_missing=True)


def load_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k.endswith("_API_KEY") or k == "HF_TOKEN"}
    for dotenv in [
        PAPERS_ROOT / ".env",
        Path.home() / "Projects" / "aster" / ".env",
        FRONTIER_ROOT / "project" / "paperbench" / ".env",
    ]:
        if dotenv.exists():
            for raw in dotenv.read_text().splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = (part.strip() for part in line.split("=", 1))
                if key.endswith("_API_KEY") or key == "HF_TOKEN":
                    env[key] = value.strip("'\"")

    token_id = os.environ.get("MODAL_TOKEN_ID") or modal_config.get("token_id")
    token_secret = os.environ.get("MODAL_TOKEN_SECRET") or modal_config.get("token_secret")
    if token_id and token_secret:
        env["MODAL_TOKEN_ID"] = str(token_id)
        env["MODAL_TOKEN_SECRET"] = str(token_secret)
    env.setdefault("GRADER_OPENAI_API_KEY", env.get("OPENAI_API_KEY", ""))

    required = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"]
    missing = [key for key in required if not env.get(key)]
    if missing:
        raise RuntimeError(f"Missing required secret(s): {', '.join(missing)}")
    return env


def image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install(
            "build-essential",
            "curl",
            "ffmpeg",
            "git",
            "git-lfs",
            "libsm6",
            "libxext6",
            "nodejs",
            "npm",
            "p7zip-full",
            "python-is-python3",
            "python3-dev",
            "python3-pip",
            "python3-venv",
            "unzip",
            "zip",
        )
        .pip_install("modal")
        .run_commands(
            "useradd -m -s /bin/bash mh",
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            "ln -sf /root/.local/bin/uv /usr/local/bin/uv",
            "ln -sf /root/.local/bin/uvx /usr/local/bin/uvx",
            "npm install -g @anthropic-ai/claude-code",
            "git clone https://github.com/openai/frontier-evals.git --filter=blob:none /work/frontier-evals",
            "cd /work/frontier-evals && git checkout 51052cede8cc608f95bb00346635e03759013e5a",
            "mkdir -p /work/papers",
        )
        .add_local_dir(
            str(ROOT),
            "/work/paperbench_score",
            copy=True,
            ignore=["logs", "logs/**", "__pycache__", "__pycache__/**"],
        )
        .add_local_dir(str(ROOT.parent / "terminal_bench_2"), "/work/terminal_bench_2", copy=True)
        .add_local_file(str(PAPERS_ROOT / "bare_runtime.py"), "/work/papers/bare_runtime.py", copy=True)
    )


@APP.function(
    image=image(),
    secrets=[modal.Secret.from_dict(load_env())],
    volumes={"/mnt/paperbench_score_logs": LOG_VOLUME},
    gpu="A10G",
    timeout=24 * 60 * 60,
)
def run_outer(
    iterations: int = 1,
    run_name: str = "paperbench-score-a10-001",
    fresh: bool = False,
    skip_baseline: bool = True,
    agent_time_limit: int = 600,
    reproduction_timeout: int = 600,
    gpu: str = "A10G",
) -> int:
    seed_dir = Path("/work/paperbench_score/logs") / run_name
    seed_dir.mkdir(parents=True, exist_ok=True)
    frontier_path = seed_dir / "frontier_val.json"
    summary_path = seed_dir / "evolution_summary.jsonl"
    if not frontier_path.exists():
        frontier_path.write_text(
            """{
  "_best": {
    "agent": "baseline_basic_agent",
    "feedback": "Overall: 0.6634\\nCode Development: 70/81\\nCode Execution: 1/7\\nResult Analysis: 0/20\\nsubmission_exists: true\\nreproduction_metadata_present: true\\njudge_output_present: true\\nerror: reproduce.sh failed with AttributeError: Namespace has no attribute method; did you mean methods?",
    "result_path": "seeded-from-one-off-modal-run",
    "score": 0.6634253588727687
  }
}
""",
            encoding="utf-8",
        )
    if not summary_path.exists():
        summary_path.write_text(
            '{"agent": "baseline_basic_agent", "axis": "seed", "error": "reproduce.sh failed with AttributeError: Namespace has no attribute method; did you mean methods?", "feedback": "Overall: 0.6634\\nCode Development: 70/81\\nCode Execution: 1/7\\nResult Analysis: 0/20\\nsubmission_exists: true\\nreproduction_metadata_present: true\\njudge_output_present: true", "hypothesis": "Seeded from one-off stricter-provenance Modal run.", "iteration": 0, "log_path": "modal app ap-TWrU33qJxL5bBEEaej7jYL", "score": 0.6634253588727687}\n',
            encoding="utf-8",
        )
    subprocess.run(
        [
            "chown",
            "-R",
            "mh:mh",
            "/work/paperbench_score",
            "/work/terminal_bench_2",
            "/work/frontier-evals",
            "/work/papers",
        ],
        check=False,
    )
    subprocess.run(["chmod", "-R", "a+rwX", "/pkg"], check=False)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"/root/.local/bin:{env.get('PATH', '')}",
            "PAPERBENCH_ROOT": "/work/frontier-evals/project/paperbench",
            "PAPERS_ROOT": "/work/papers",
            "PAPERBENCH_INLINE_EVAL": "1",
        }
    )
    cmd = [
        "python",
        "meta_harness.py",
        "--iterations",
        str(iterations),
        "--run-name",
        run_name,
        "--agent-time-limit",
        str(agent_time_limit),
        "--reproduction-timeout",
        str(reproduction_timeout),
        "--gpu",
        gpu,
    ]
    if fresh:
        cmd.append("--fresh")
    if skip_baseline:
        cmd.append("--skip-baseline")

    shell_cmd = shlex.join(cmd)
    proc = subprocess.Popen(
        ["su", "mh", "-c", f"cd /work/paperbench_score && {shell_cmd}"],
        cwd="/work/paperbench_score",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    code = proc.wait()
    archive_dir = Path("/mnt/paperbench_score_logs") / run_name
    archive_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["cp", "-a", f"/work/paperbench_score/logs/{run_name}/.", str(archive_dir)],
        check=False,
    )
    LOG_VOLUME.commit()
    return code


@APP.local_entrypoint()
def main(
    iterations: int = 1,
    run_name: str = "paperbench-score-a10-001",
    fresh: bool = False,
    skip_baseline: bool = True,
    agent_time_limit: int = 600,
    reproduction_timeout: int = 600,
    gpu: str = "A10G",
):
    call = run_outer.spawn(
        iterations=iterations,
        run_name=run_name,
        fresh=fresh,
        skip_baseline=skip_baseline,
        agent_time_limit=agent_time_limit,
        reproduction_timeout=reproduction_timeout,
        gpu=gpu,
    )
    print(f"Spawned remote outer loop: {call.object_id}")
