from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from unittest.mock import AsyncMock

import meta_harness
from agents.baseline_kira import AgentHarness
from harbor.llms.base import LLMResponse
from harbor.models.agent.context import AgentContext


ROOT = Path(__file__).resolve().parents[1]
RUN_EVAL = ROOT / "scripts" / "run_eval.sh"


class TaskPromptTests(unittest.TestCase):
    def test_prompt_lists_only_paths_the_loop_uses(self) -> None:
        prompt = meta_harness.render_task_prompt(1, 2)
        listed = {
            path.rstrip("/") for path in re.findall(r"`([^`]+)`", prompt)
        }
        expected = {
            str(meta_harness.LOGS_DIR),
            str(meta_harness.EVOLUTION_SUMMARY),
            str(meta_harness.FRONTIER_VAL),
            str(meta_harness.PENDING_EVAL),
        }
        self.assertEqual(listed, expected)


class AgentClassValidationTests(unittest.TestCase):
    def test_accepts_terminus2_subclass(self) -> None:
        agent_class = meta_harness.validate_agent_class(
            "agents.baseline_kira:AgentHarness"
        )

        self.assertEqual(agent_class.__name__, "AgentHarness")

    def test_rejects_missing_exact_class(self) -> None:
        with self.assertRaisesRegex(AttributeError, "AgentHarness"):
            meta_harness.validate_agent_class("anthropic_caching:AgentHarness")

    def test_rejects_non_class(self) -> None:
        with self.assertRaisesRegex(TypeError, "is not a class"):
            meta_harness.validate_agent_class("anthropic_caching:add_anthropic_caching")

    def test_rejects_class_that_is_not_terminus2(self) -> None:
        with self.assertRaisesRegex(TypeError, "must subclass Terminus2"):
            meta_harness.validate_agent_class("pathlib:Path")

    def test_run_eval_rejects_invalid_class_before_harbor(self) -> None:
        result = subprocess.run(
            [str(RUN_EVAL), "pathlib:Path", "full", "1", "1"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must subclass Terminus2", result.stderr)
        self.assertNotIn("harbor run", result.stdout)


class TimeoutWiringTests(unittest.TestCase):
    def _run_shell(
        self,
        harbor_environment: str | None = None,
        external_timeout: bool = False,
    ) -> tuple[list[str], str]:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            capture = tmp_path / "timeout-args"
            capture_pythonpath = tmp_path / "pythonpath"
            fake_timeout = tmp_path / "timeout"
            fake_uv = tmp_path / "uv"
            fake_timeout.write_text(
                '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE"\n'
                'printf "%s" "$PYTHONPATH" > "$CAPTURE_PYTHONPATH"\n'
            )
            fake_timeout.chmod(0o755)
            fake_uv.write_text(
                "#!/usr/bin/env bash\n"
                'if [[ "$*" == *"harbor run"* ]]; then\n'
                '  printf "%s\\n" "$@" > "$CAPTURE"\n'
                '  printf "%s" "$PYTHONPATH" > "$CAPTURE_PYTHONPATH"\n'
                "fi\n"
            )
            fake_uv.chmod(0o755)
            env = os.environ.copy()
            env.pop("HARBOR_ENVIRONMENT", None)
            env.pop("HARBOR_EXTERNAL_TIMEOUT", None)
            env.pop("PYTHONPATH", None)
            env.update(
                {
                    "CAPTURE": str(capture),
                    "CAPTURE_PYTHONPATH": str(capture_pythonpath),
                    "HARBOR_TIMEOUT_SECONDS": "12345",
                    "PATH": f"{tmp_path}:{env['PATH']}",
                }
            )
            if harbor_environment:
                env["HARBOR_ENVIRONMENT"] = harbor_environment
            if external_timeout:
                env["HARBOR_EXTERNAL_TIMEOUT"] = "1"

            result = subprocess.run(
                [
                    str(RUN_EVAL),
                    "agents.baseline_kira:AgentHarness",
                    "full",
                    "1",
                    "1",
                ],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            return capture.read_text().splitlines(), capture_pythonpath.read_text()

    def test_python_runner_uses_harbor_timeout(self) -> None:
        completed = subprocess.CompletedProcess([], 0, "", "")
        with (
            mock.patch.object(meta_harness, "HARBOR_TIMEOUT_SECONDS", 12345),
            mock.patch.object(
                meta_harness.subprocess, "run", return_value=completed
            ) as run,
        ):
            _, result = meta_harness.harbor_run(
                "agents.baseline_kira:AgentHarness", "test-job"
            )

        self.assertTrue(result)
        self.assertEqual(run.call_args.kwargs["timeout"], 12345)
        self.assertEqual(run.call_args.kwargs["env"]["HARBOR_TIMEOUT_SECONDS"], "12345")
        self.assertEqual(run.call_args.kwargs["env"]["HARBOR_EXTERNAL_TIMEOUT"], "1")

    def test_python_runner_does_not_complete_on_timeout(self) -> None:
        timed_out = subprocess.CompletedProcess([], 124, "", "timed out")
        with mock.patch.object(meta_harness.subprocess, "run", return_value=timed_out):
            _, result = meta_harness.harbor_run(
                "agents.baseline_kira:AgentHarness", "test-job"
            )

        self.assertIsNone(result)

    def test_shell_runner_uses_harbor_timeout(self) -> None:
        timeout_args, pythonpath = self._run_shell()

        self.assertEqual(
            timeout_args[:3], ["--signal=TERM", "--kill-after=60", "12345"]
        )
        self.assertEqual(timeout_args[3:6], ["uv", "run", "harbor"])
        agent_flag = timeout_args.index("--agent")
        self.assertEqual(
            timeout_args[agent_flag + 1], "agents.baseline_kira:AgentHarness"
        )
        environment_flag = timeout_args.index("-e")
        self.assertEqual(timeout_args[environment_flag + 1], "runloop")
        self.assertIn("temperature=0.7", timeout_args)
        self.assertEqual(Path(pythonpath.split(os.pathsep)[0]), ROOT)

    def test_shell_runner_supports_modal(self) -> None:
        timeout_args, _ = self._run_shell("modal")

        environment_flag = timeout_args.index("-e")
        self.assertEqual(timeout_args[environment_flag + 1], "modal")

    def test_shell_runner_defers_to_external_timeout(self) -> None:
        command_args, _ = self._run_shell(external_timeout=True)

        self.assertEqual(command_args[:3], ["run", "harbor", "run"])


class KiraLoopSignatureTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_agent_loop_accepts_current_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = AgentHarness(
                logs_dir=Path(tmp),
                model_name="openai/gpt-4o-mini",
                max_turns=1,
                enable_summarize=False,
                suppress_max_turns_warning=True,
            )
            agent._context = AgentContext()
            agent._session = SimpleNamespace(
                is_session_alive=AsyncMock(return_value=True)
            )
            chat = SimpleNamespace(
                total_input_tokens=0,
                total_output_tokens=0,
                total_cache_tokens=0,
                total_cost=0,
            )
            agent._handle_llm_interaction = AsyncMock(
                return_value=(
                    [],
                    False,
                    "",
                    "",
                    "",
                    LLMResponse(content=""),
                    None,
                )
            )
            agent._execute_commands = AsyncMock(return_value=(False, ""))
            agent._dump_trajectory = mock.Mock()

            await agent._run_agent_loop("task", chat, original_instruction="task")

            self.assertEqual(agent._n_episodes, 1)
            self.assertEqual(len(agent._trajectory_steps), 1)


if __name__ == "__main__":
    unittest.main()
