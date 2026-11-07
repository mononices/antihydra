"""Sandbox lifecycle guarantees that racing relies on."""

from __future__ import annotations

import threading
import unittest
from unittest import mock

from antihydra.sandbox import Sandbox


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


if __name__ == "__main__":
    unittest.main()
