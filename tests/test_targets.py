"""Reachable-target resolution: host-published ports are useless inside ctfnet."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from antihydra.challenge import load
from antihydra.prompts import initial_prompt, system_prompt


class TargetTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def make_challenge(self, metadata: dict, *, compose: bool = False):
        directory = self.root / "web" / "demo"
        directory.mkdir(parents=True)
        (directory / "challenge.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        if compose:
            (directory / "docker-compose.yml").write_text(
                "services:\n  game-server:\n    image: demo\n", encoding="utf-8"
            )
        return load(directory, overrides={})

    def test_remote_only_when_no_compose_file(self) -> None:
        challenge = self.make_challenge(
            {
                "category": "web",
                "name": "gradebook",
                "box": "web_client.chal.csaw.io",
                "internal_port": 4747,
                "files": [],
            }
        )

        self.assertTrue(challenge.is_remote_only())
        self.assertIsNone(challenge.reachable_target())
        self.assertIn("not reachable", initial_prompt(challenge))
        self.assertNotIn(
            "reachable from inside the sandbox", system_prompt(challenge, 300)
        )

    def test_local_target_wins_when_discovered(self) -> None:
        challenge = self.make_challenge(
            {
                "category": "web",
                "name": "orion",
                "box": "web.chal.csaw.io",
                "internal_port": 3555,
                "files": [],
            },
            compose=True,
        )
        challenge.local_target = "web.chal.csaw.io:3000"

        self.assertEqual(challenge.reachable_target(), "web.chal.csaw.io:3000")
        self.assertIn("web.chal.csaw.io:3000", initial_prompt(challenge))

    def test_discover_local_target_reads_container_port_and_alias(self) -> None:
        challenge = self.make_challenge(
            {
                "category": "web",
                "name": "orion",
                "box": "web.chal.csaw.io",
                "internal_port": 3555,
                "files": [],
            },
            compose=True,
        )
        inspect = [
            {
                "Name": "/demo-game-server-1",
                "Config": {
                    "ExposedPorts": {"3000/tcp": {}},
                    "Labels": {"com.docker.compose.service": "game-server"},
                },
                "NetworkSettings": {
                    "Networks": {
                        "ctfnet": {
                            "Aliases": [
                                "game-server",
                                "web.chal.csaw.io",
                                "demo-game-server-1",
                            ]
                        }
                    }
                },
            }
        ]

        def fake_run(command, **kwargs):
            if "ps" in command:
                return mock.Mock(returncode=0, stdout=b"abc123\n")
            return mock.Mock(returncode=0, stdout=json.dumps(inspect).encode())

        with mock.patch("antihydra.challenge.subprocess.run", side_effect=fake_run):
            target = challenge.discover_local_target("docker")

        self.assertEqual(target, "web.chal.csaw.io:3000")
        self.assertEqual(challenge.reachable_target(), "web.chal.csaw.io:3000")

    def test_discover_local_target_without_a_stack(self) -> None:
        challenge = self.make_challenge(
            {"category": "crypto", "name": "offline", "files": []}
        )

        with mock.patch("antihydra.challenge.subprocess.run") as run:
            self.assertIsNone(challenge.discover_local_target("docker"))

        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
