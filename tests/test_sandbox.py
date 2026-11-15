"""Sandbox lifecycle guarantees that racing relies on."""

from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from antihydra.sandbox import OutputFlood, Sandbox, _run_streaming

MENU = (
    "=== Thermite Charge Inventory ===\n"
    "1) plant charge\n"
    "2) defuse charge\n"
    "3) rewire charge\n"
    "4) inspect charge\n"
    "5) leave\n"
    "> no such option\n"
)


class SandboxStopTests(unittest.TestCase):
    def test_concurrent_stop_calls_docker_once(self) -> None:
        sandbox = Sandbox(image="fake:test", docker="docker")
        sandbox.container_id = "abc123"
        commands: list[list[str]] = []

        def fake_run(command, **kwargs):
            commands.append(command)
            return mock.Mock(returncode=0, stdout=b"", stderr=b"")

        threads = [threading.Thread(target=sandbox.stop) for _ in range(6)]
        with mock.patch("antihydra.sandbox.subprocess.run", side_effect=fake_run):
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.assertIsNone(sandbox.container_id)
        self.assertEqual([command[1] for command in commands], ["stop", "rm"])
        self.assertTrue(all("abc123" in command for command in commands))

    def test_stop_after_stop_is_a_noop(self) -> None:
        sandbox = Sandbox(image="fake:test", docker="docker")
        sandbox.container_id = "abc123"

        with mock.patch("antihydra.sandbox.subprocess.run") as run:
            run.return_value = mock.Mock(returncode=0, stdout=b"", stderr=b"")
            sandbox.stop()
            sandbox.stop()

        self.assertEqual(run.call_count, 2)


class OutputFloodTests(unittest.TestCase):
    def test_repeating_line_trips(self) -> None:
        guard = OutputFlood()
        self.assertFalse(guard.feed("no such option\n" * 10))
        self.assertTrue(guard.feed("no such option\n" * 500))
        self.assertIn("no such option", guard.describe())

    def test_repeated_menu_block_trips(self) -> None:
        guard = OutputFlood()
        self.assertTrue(guard.feed(MENU * 200))
        self.assertIn("repeated", guard.describe())

    def test_distinct_output_does_not_trip(self) -> None:
        guard = OutputFlood()
        self.assertFalse(guard.feed("".join(f"line {i}\n" for i in range(20000))))

    def test_blank_lines_are_ignored(self) -> None:
        guard = OutputFlood()
        self.assertFalse(guard.feed("\n" * 100000))

    def test_byte_backstop_trips(self) -> None:
        guard = OutputFlood()
        self.assertFalse(guard.feed("x" * 1000))
        self.assertTrue(guard.feed("y" * 2_000_001))
        self.assertIn("bytes", guard.describe())


class StreamingExecTests(unittest.TestCase):
    """The runner is exercised with a real local process, not docker."""

    def run_code(self, code: str, timeout: float = 30.0):
        return _run_streaming([sys.executable, "-c", code], timeout=timeout)

    def test_normal_command_returns_output_and_status(self) -> None:
        result = self.run_code("print('hello')")
        self.assertTrue(result.ok)
        self.assertEqual(result.stdout, "hello\n")
        self.assertIsNone(result.killed)

    def test_stderr_stays_separate(self) -> None:
        result = self.run_code("import sys; print('bad', file=sys.stderr)")
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "bad\n")

    def test_nonzero_exit_is_reported(self) -> None:
        result = self.run_code("raise SystemExit(3)")
        self.assertEqual(result.exit_code, 3)
        self.assertFalse(result.ok)

    def test_looping_output_is_killed_early(self) -> None:
        started = time.time()
        result = self.run_code(
            "print(('no such option' + chr(10)) * 500000, end='')"
        )
        elapsed = time.time() - started
        self.assertEqual(result.killed, "output_flood")
        self.assertIn("no such option", result.note)
        self.assertLess(elapsed, 10)
        # the pipe buffer is drained after the kill, so a little more arrives
        self.assertLess(len(result.stdout), 300000)

    def test_menu_loop_without_stdin_is_killed(self) -> None:
        """Hells Bells: a menu binary with closed stdin spins on 'no such option'."""
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "menu.py"
            script.write_text(
                "menu = '=== Thermite Charge Inventory ==='\n"
                "option = '> no such option'\n"
                "while True:\n"
                "    print(menu)\n"
                "    print(option)\n",
                encoding="utf-8",
            )
            started = time.time()
            result = _run_streaming([sys.executable, str(script)], timeout=30)
            elapsed = time.time() - started
        self.assertEqual(result.killed, "output_flood")
        self.assertIn("repeated", result.note)
        self.assertIn("no such option", result.stdout)
        self.assertLess(elapsed, 10)

    def test_silent_command_still_times_out(self) -> None:
        started = time.time()
        result = self.run_code("import time; time.sleep(30)", timeout=1.0)
        self.assertTrue(result.timed_out)
        self.assertEqual(result.killed, "timeout")
        self.assertLess(time.time() - started, 10)

    def test_flood_reason_reaches_the_agent_payload(self) -> None:
        payload = self.run_code(
            "print(('spamspam' + chr(10)) * 500000, end='')"
        ).to_dict()
        self.assertEqual(payload["killed"], "output_flood")
        self.assertIsNone(payload["exit_code"])
        self.assertIn("spamspam", payload["killed_note"])


if __name__ == "__main__":
    unittest.main()
