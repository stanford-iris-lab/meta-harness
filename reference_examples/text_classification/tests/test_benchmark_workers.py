from __future__ import annotations

import asyncio
import os
import signal
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

if os.name == "posix":
    import fcntl

from text_classification.benchmark import _run_with_retries


class BenchmarkWorkerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.log = self.root / "worker.log"

    async def test_success_does_not_retry(self) -> None:
        ok = await _run_with_retries(
            [sys.executable, "-c", "print('finished')"], self.log, timeout=5
        )
        self.assertTrue(ok)
        self.assertIn("finished", self.log.read_text())
        self.assertEqual(self.log.read_text().count("exit=0"), 1)
        self.assertNotIn("retry", self.log.read_text())

    async def test_nonzero_exit_retries(self) -> None:
        marker = self.root / "attempt"
        script = (
            "from pathlib import Path; import sys; "
            "p = Path(sys.argv[1]); first = not p.exists(); "
            "p.touch(); sys.exit(1 if first else 0)"
        )
        ok = await _run_with_retries(
            [sys.executable, "-c", script, str(marker)],
            self.log,
            max_retries=1,
            timeout=5,
        )
        self.assertTrue(ok)
        self.assertIn("exit=1", self.log.read_text())
        self.assertIn("retry 1", self.log.read_text())
        self.assertIn("exit=0", self.log.read_text())

    def tree_command(self) -> list[str]:
        # Both processes hold locks, so liveness checks also work when an
        # orphan has exited but has not yet been reaped by the host's init.
        script = self.root / "tree.py"
        script.write_text(
            textwrap.dedent("""\
            import fcntl
            import os
            import subprocess
            import sys
            import time
            from pathlib import Path

            root = Path(sys.argv[1])
            role = sys.argv[2]
            with (root / "pids").open("a") as pids:
                pids.write(f"{os.getpid()}\\n")
            lock = (root / f"{role}.lock").open("w")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                (root / "overlap").touch()
            if role == "parent":
                child = subprocess.Popen(
                    [sys.executable, __file__, str(root), "child"]
                )
                child.wait()
            else:
                with (root / "ready").open("a") as ready:
                    ready.write("ready\\n")
                time.sleep(60)
        """)
        )
        self.addCleanup(self.kill_test_processes)
        return [sys.executable, str(script), str(self.root), "parent"]

    def kill_test_processes(self) -> None:
        pids = self.root / "pids"
        if pids.exists():
            for pid in pids.read_text().splitlines():
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def tree_has_exited(self) -> bool:
        for role in ("parent", "child"):
            with (self.root / f"{role}.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return False
        return True

    async def assert_tree_exited(self) -> None:
        deadline = asyncio.get_running_loop().time() + 2
        while not self.tree_has_exited():
            if asyncio.get_running_loop().time() >= deadline:
                self.fail("worker processes are still running")
            await asyncio.sleep(0.01)

    @unittest.skipUnless(os.name == "posix", "process groups require POSIX")
    async def test_timeout_stops_tree_before_retry(self) -> None:
        ok = await _run_with_retries(
            self.tree_command(), self.log, max_retries=1, timeout=1
        )
        self.assertFalse(ok)
        self.assertEqual(self.log.read_text().count("exit=124"), 2)
        self.assertEqual((self.root / "ready").read_text().splitlines(), ["ready"] * 2)
        self.assertFalse((self.root / "overlap").exists())
        await self.assert_tree_exited()

    @unittest.skipUnless(os.name == "posix", "process groups require POSIX")
    async def test_cancellation_stops_tree_and_propagates(self) -> None:
        task = asyncio.create_task(
            _run_with_retries(self.tree_command(), self.log, timeout=30)
        )
        try:
            deadline = asyncio.get_running_loop().time() + 5
            while not (self.root / "ready").exists():
                if asyncio.get_running_loop().time() >= deadline:
                    self.fail("worker did not start")
                await asyncio.sleep(0.01)
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        await self.assert_tree_exited()
        self.assertNotIn("retry", self.log.read_text())


if __name__ == "__main__":
    unittest.main()
