"""Image prebuild: build errors must surface before a race starts spending."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from antihydra.challenge import COMPOSE_BUILD_PATTERN, ChallengeError, load

BUILD_STACK = "services:\n  app:\n    build: .\n"
BUILD_STACK_JSON = '{"services": {"app": {"build": {"context": "."}}}}'
PULL_STACK = "services:\n  app:\n    image: nginx:alpine\n"


def fake_docker(*, fail: bool = False):
    """Compose probe succeeds; build/up returns a scripted exit status."""
    calls: list[list[str]] = []

    def run(command, **kwargs):
        calls.append(list(command))
        if command[-1] == "version":
            return mock.Mock(returncode=0, stdout=b"", stderr=b"")
        if fail:
            return mock.Mock(
                returncode=1,
                stdout=b"#5 [1/3] RUN pip install -r requirements.txt\n",
                stderr=b"pip: connection reset\n",
            )
        return mock.Mock(returncode=0, stdout=b"built ok\n", stderr=b"")

    return run, calls


class BuildImagesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def make_challenge(self, compose: str, name: str = "demo"):
        directory = self.root / "web" / name
        directory.mkdir(parents=True)
        (directory / "challenge.json").write_text(
            json.dumps(
                {"name": name, "category": "web", "flag": "csawctf{x}"}
            ),
            encoding="utf-8",
        )
        (directory / "docker-compose.yml").write_text(compose, encoding="utf-8")
        return load(directory, overrides={})

    def test_build_images_runs_compose_build_and_keeps_the_log(self) -> None:
        challenge = self.make_challenge(BUILD_STACK)
        log = self.root / "run" / "build.log"
        run, calls = fake_docker()

        with mock.patch("antihydra.challenge.subprocess.run", side_effect=run):
            challenge.build_images("docker", log_path=log)

        self.assertEqual(calls[-1][-1], "build")
        self.assertIn("built ok", log.read_text(encoding="utf-8"))

    def test_build_images_raises_with_the_failing_step(self) -> None:
        challenge = self.make_challenge(BUILD_STACK_JSON)
        log = self.root / "run" / "build.log"
        run, _ = fake_docker(fail=True)

        with mock.patch("antihydra.challenge.subprocess.run", side_effect=run):
            with self.assertRaises(ChallengeError) as caught:
                challenge.build_images("docker", log_path=log)

        self.assertIn("pip: connection reset", str(caught.exception))
        written = log.read_text(encoding="utf-8")
        self.assertIn("pip install -r requirements.txt", written)
        self.assertIn("pip: connection reset", written)

    def test_build_images_leaves_pull_only_stacks_to_compose(self) -> None:
        challenge = self.make_challenge(PULL_STACK)
        log = self.root / "run" / "build.log"
        run, calls = fake_docker()

        with mock.patch("antihydra.challenge.subprocess.run", side_effect=run):
            challenge.build_images("docker", log_path=log)

        self.assertEqual(calls, [])
        self.assertIn("no build section", log.read_text(encoding="utf-8"))

    def test_start_service_can_skip_building(self) -> None:
        challenge = self.make_challenge(BUILD_STACK)
        run, calls = fake_docker()

        with mock.patch("antihydra.challenge.subprocess.run", side_effect=run):
            challenge.start_service("docker", build=False)

        self.assertIn("--no-build", calls[-1])

    def test_start_service_builds_by_default(self) -> None:
        challenge = self.make_challenge(BUILD_STACK)
        run, calls = fake_docker()

        with mock.patch("antihydra.challenge.subprocess.run", side_effect=run):
            challenge.start_service("docker")

        self.assertNotIn("--no-build", calls[-1])

    def test_build_section_detection_covers_both_compose_flavours(self) -> None:
        self.assertTrue(COMPOSE_BUILD_PATTERN.search(BUILD_STACK))
        self.assertTrue(COMPOSE_BUILD_PATTERN.search(BUILD_STACK_JSON))
        self.assertTrue(
            COMPOSE_BUILD_PATTERN.search('{\n  "build": {\n    "context": "."\n  }\n}')
        )
        self.assertFalse(COMPOSE_BUILD_PATTERN.search(PULL_STACK))
        self.assertFalse(
            COMPOSE_BUILD_PATTERN.search(
                "services:\n  app:\n    dockerfile: build/Dockerfile\n"
            )
        )

    def make_patched_challenge(self, *, source: Path | None = None):
        """Challenge whose build context is corrected by a build_files override."""
        directory = self.root / "pwn" / "vespers"
        (directory / "build").mkdir(parents=True)
        (directory / "src").mkdir(parents=True)
        (directory / "challenge.json").write_text(
            json.dumps({"name": "vespers", "category": "pwn", "flag": "csaw{x}"}),
            encoding="utf-8",
        )
        (directory / "docker-compose.yml").write_text(
            '{"services": {"app": {"build": {"dockerfile": "build/Dockerfile"}}}}',
            encoding="utf-8",
        )
        (directory / "build" / "Dockerfile").write_text(
            "FROM ubuntu:22.04\nRUN apt-get install -y --no-install-recommends gcc\n",
            encoding="utf-8",
        )
        (directory / "src" / "vespers.c").write_text("int main(void){}\n", encoding="utf-8")
        fix = source or (self.root / "fixes" / "Dockerfile")
        fix.parent.mkdir(parents=True, exist_ok=True)
        fix.write_text(
            "FROM ubuntu:22.04\nRUN apt-get install -y gcc libc6-dev\n",
            encoding="utf-8",
        )
        overrides = {
            "pwn/vespers": {"build_files": {"build/Dockerfile": str(fix)}}
        }
        return load(directory, overrides=overrides)

    def test_build_files_override_builds_from_a_staged_context(self) -> None:
        challenge = self.make_patched_challenge()
        log = self.root / "run" / "build.log"
        run, calls = fake_docker()

        with mock.patch("antihydra.challenge.subprocess.run", side_effect=run):
            challenge.build_images("docker", log_path=log)

        staged = challenge.build_context()
        self.assertNotEqual(staged, challenge.directory)
        self.assertIn("libc6-dev", (staged / "build" / "Dockerfile").read_text())
        self.assertNotIn(
            "libc6-dev",
            (challenge.directory / "build" / "Dockerfile").read_text(encoding="utf-8"),
        )
        self.assertTrue((staged / "src" / "vespers.c").is_file())
        self.assertEqual(calls[-1][-1], "build")
        # compose runs from the staged copy, so relative build paths resolve there
        self.assertEqual(challenge.compose_file().parent, staged)
        self.assertIn(str(challenge.compose_file()), calls[-1])

    def test_build_files_override_also_applies_to_start_service(self) -> None:
        challenge = self.make_patched_challenge()
        run, calls = fake_docker()

        with mock.patch("antihydra.challenge.subprocess.run", side_effect=run):
            service = challenge.start_service("docker", build=False)

        self.assertEqual(service.compose_file.parent, challenge.build_context())
        self.assertIn(str(challenge.compose_file()), calls[-1])
        self.assertIn("--no-build", calls[-1])

    def test_build_files_source_must_exist(self) -> None:
        challenge = self.make_patched_challenge()
        (self.root / "fixes" / "Dockerfile").unlink()

        with self.assertRaises(ChallengeError) as caught:
            challenge.build_context()

        self.assertIn("does not exist", str(caught.exception))

    def test_build_files_cannot_escape_the_context(self) -> None:
        directory = self.root / "pwn" / "escape"
        directory.mkdir(parents=True)
        (directory / "challenge.json").write_text(
            json.dumps({"name": "escape", "category": "pwn", "flag": "csaw{x}"}),
            encoding="utf-8",
        )
        fix = self.root / "fixes" / "Dockerfile"
        fix.parent.mkdir(parents=True, exist_ok=True)
        fix.write_text("FROM ubuntu:22.04\n", encoding="utf-8")
        challenge = load(
            directory,
            overrides={
                "pwn/escape": {"build_files": {"../outside": str(fix)}}
            },
        )

        with self.assertRaises(ChallengeError) as caught:
            challenge.build_context()

        self.assertIn("escapes the build context", str(caught.exception))

    def test_start_service_error_keeps_stdout(self) -> None:
        challenge = self.make_challenge(BUILD_STACK)
        run, _ = fake_docker(fail=True)

        with mock.patch("antihydra.challenge.subprocess.run", side_effect=run):
            with self.assertRaises(ChallengeError) as caught:
                challenge.start_service("docker")

        message = str(caught.exception)
        self.assertIn("pip install -r requirements.txt", message)
        self.assertIn("pip: connection reset", message)


if __name__ == "__main__":
    unittest.main()
