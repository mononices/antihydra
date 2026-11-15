"""Evaluation harness tests with fake coordinators (no docker, no network)."""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from antihydra.challenge import Challenge, ChallengeError, safe_dir_name
from antihydra.evaluation import (
    EvalConfig,
    exclusive_resources,
    prebuild_images,
    run_dir_name,
    run_eval,
)


def make_challenge(directory: Path, name: str, category: str) -> Challenge:
    return Challenge(
        directory=directory,
        metadata={
            "name": name,
            "category": category,
            "flag": "csawctf{x}",
            "description": "demo",
        },
    )


def report(solved: bool = False, flag: str | None = None) -> dict:
    return {
        "solved": solved,
        "flag": flag,
        "solved_by_model": "m1" if solved else None,
        "total_cost": 0.1,
        "elapsed_seconds": 1.0,
        "k": 2,
        "lanes": [
            {
                "id": 1,
                "model": "m1",
                "status": "solved" if solved else "unsolved",
                "cost": 0.1,
                "rounds": 3,
                "flag": flag,
                "transcript": "lane-1.json",
            },
            {
                "id": 2,
                "model": "m2",
                "status": "aborted",
                "cost": 0.0,
                "rounds": 1,
                "flag": None,
                "transcript": "lane-2.json",
            },
        ],
    }


class Tracker:
    """Records peak concurrent coordinators."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.now = 0
        self.max = 0

    def enter(self) -> None:
        with self.lock:
            self.now += 1
            self.max = max(self.max, self.now)

    def exit(self) -> None:
        with self.lock:
            self.now -= 1


class FakeCoordinator:
    """Writes a scripted race summary, optionally sleeping to expose concurrency."""

    def __init__(
        self,
        result: dict,
        race_dir: Path,
        tracker: Tracker | None = None,
        delay: float = 0.0,
    ) -> None:
        self.result = result
        self.race_dir = race_dir
        self.tracker = tracker
        self.delay = delay

    def run(self) -> dict:
        if self.tracker is not None:
            self.tracker.enter()
        try:
            if self.delay:
                time.sleep(self.delay)
            self.race_dir.mkdir(parents=True, exist_ok=True)
            (self.race_dir / "race-summary.json").write_text(
                json.dumps(self.result), encoding="utf-8"
            )
            return self.result
        finally:
            if self.tracker is not None:
                self.tracker.exit()


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_eval_writes_artifacts(self) -> None:
        challenges = [
            make_challenge(self.root, "easy", "crypto"),
            make_challenge(self.root, "hard", "pwn"),
        ]
        eval_dir = self.root / "eval"

        def factory(challenge, run_dir):
            if "easy" in run_dir.name:
                return FakeCoordinator(report(True, "csawctf{found}"), run_dir)
            return FakeCoordinator(report(False), run_dir)

        summary = run_eval(
            challenges,
            eval_dir=eval_dir,
            coordinator_factory=factory,
            config=EvalConfig(lane_models=["m1", "m2"]),
            quiet=True,
        )

        self.assertEqual(summary["challenges"], 2)
        self.assertEqual(summary["solved"], 1)
        self.assertEqual(summary["k"], 2)
        self.assertEqual(summary["by_category"]["pwn"]["solved"], 0)
        self.assertAlmostEqual(summary["total_cost"], 0.2)
        self.assertAlmostEqual(summary["lane_solve_rate"], 0.25)

        flag_path = eval_dir / "01__easy" / "flag.txt"
        self.assertEqual(
            flag_path.read_text(encoding="utf-8").strip(), "csawctf{found}"
        )
        self.assertFalse((eval_dir / "02__hard" / "flag.txt").exists())

        saved = json.loads((eval_dir / "eval-summary.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["solved"], 1)
        self.assertEqual(saved["config"]["concurrency"], 3)
        leaderboard = json.loads(
            (eval_dir / "summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(leaderboard["results"], {"easy": True, "hard": False})
        self.assertEqual(leaderboard["metadata"]["comment"], "pass@2")
        traces = (
            (eval_dir / "traces.jsonl").read_text(encoding="utf-8").strip().splitlines()
        )
        self.assertEqual(len(traces), 4)

    def test_skip_existing_reuses_summaries(self) -> None:
        challenges = [
            make_challenge(self.root, "a", "misc"),
            make_challenge(self.root, "b", "misc"),
        ]
        eval_dir = self.root / "eval"
        calls = {"count": 0}

        def factory(challenge, run_dir):
            calls["count"] += 1
            return FakeCoordinator(report(False), run_dir)

        run_eval(challenges, eval_dir=eval_dir, coordinator_factory=factory, quiet=True)
        self.assertEqual(calls["count"], 2)

        run_eval(
            challenges,
            eval_dir=eval_dir,
            coordinator_factory=factory,
            config=EvalConfig(skip_existing=True),
            quiet=True,
        )
        self.assertEqual(calls["count"], 2)

    def test_one_failing_challenge_does_not_stop_the_eval(self) -> None:
        challenges = [
            make_challenge(self.root, "good", "misc"),
            make_challenge(self.root, "bad", "misc"),
        ]

        class Boom:
            def run(self):
                raise RuntimeError("boom")

        def factory(challenge, run_dir):
            if "bad" in run_dir.name:
                return Boom()
            return FakeCoordinator(report(False), run_dir)

        summary = run_eval(
            challenges,
            eval_dir=self.root / "eval",
            coordinator_factory=factory,
            quiet=True,
        )

        self.assertEqual(summary["challenges"], 2)
        self.assertIn("boom", summary["results"][1]["error"])

    def test_concurrency_limit_is_respected(self) -> None:
        challenges = [
            make_challenge(self.root, f"c{index}", "misc") for index in range(4)
        ]
        tracker = Tracker()

        def factory(challenge, run_dir):
            return FakeCoordinator(report(False), run_dir, tracker=tracker, delay=0.15)

        started = time.time()
        run_eval(
            challenges,
            eval_dir=self.root / "eval",
            coordinator_factory=factory,
            config=EvalConfig(concurrency=2),
            quiet=True,
        )
        elapsed = time.time() - started

        self.assertEqual(tracker.max, 2)
        self.assertLess(elapsed, 0.55)

    def test_shared_host_ports_are_serialized(self) -> None:
        directories = [self.root / "alpha", self.root / "beta"]
        for directory in directories:
            directory.mkdir()
            (directory / "docker-compose.yml").write_text(
                "services:\n  app:\n    image: x\n    ports:\n      - 22333:21004\n",
                encoding="utf-8",
            )
        challenges = [
            make_challenge(directories[0], "alpha", "crypto"),
            make_challenge(directories[1], "beta", "rev"),
        ]
        tracker = Tracker()

        def factory(challenge, run_dir):
            return FakeCoordinator(report(False), run_dir, tracker=tracker, delay=0.15)

        run_eval(
            challenges,
            eval_dir=self.root / "eval",
            coordinator_factory=factory,
            config=EvalConfig(concurrency=2),
            quiet=True,
        )

        self.assertEqual(tracker.max, 1)

    def make_compose_challenge(self, name: str, slug: str) -> Challenge:
        directory = self.root / slug
        directory.mkdir()
        (directory / "docker-compose.yml").write_text(
            "services:\n  app:\n    build: .\n", encoding="utf-8"
        )
        return make_challenge(directory, name, "web")

    def test_prebuild_images_builds_every_stack_before_the_race(self) -> None:
        challenge = self.make_compose_challenge("alpha", "alpha")
        recorder = mock.Mock()

        with mock.patch.object(challenge, "build_images", recorder):
            failures = prebuild_images(
                "docker", [challenge], self.root / "eval", say=lambda _: None
            )

        self.assertEqual(failures, {})
        recorder.assert_called_once()
        args, kwargs = recorder.call_args
        self.assertEqual(args, ("docker",))
        self.assertEqual(
            kwargs["log_path"], self.root / "eval" / "01__alpha" / "build.log"
        )

    def test_prebuild_images_reports_failures_by_directory(self) -> None:
        challenge = self.make_compose_challenge("alpha", "alpha")
        broken = ChallengeError("failed to build challenge images: pip install failed")

        with mock.patch.object(challenge, "build_images", side_effect=broken):
            failures = prebuild_images(
                "docker", [challenge], self.root / "eval", say=lambda _: None
            )

        self.assertEqual(list(failures), [str(challenge.directory)])
        self.assertIn("pip install failed", failures[str(challenge.directory)])

    def test_prebuild_images_skips_existing_runs(self) -> None:
        challenge = self.make_compose_challenge("alpha", "alpha")
        run_dir = self.root / "eval" / "01__alpha"
        run_dir.mkdir(parents=True)
        (run_dir / "race-summary.json").write_text("{}", encoding="utf-8")

        with mock.patch.object(challenge, "build_images") as build:
            failures = prebuild_images(
                "docker",
                [challenge],
                self.root / "eval",
                skip_existing=True,
                say=lambda _: None,
            )

        build.assert_not_called()
        self.assertEqual(failures, {})

    def test_prebuild_images_ignores_remote_only_challenges(self) -> None:
        challenge = make_challenge(self.root / "beta", "beta", "web")

        with mock.patch.object(challenge, "build_images") as build:
            failures = prebuild_images(
                "docker", [challenge], self.root / "eval", say=lambda _: None
            )

        build.assert_not_called()
        self.assertEqual(failures, {})

    def test_challenge_names_with_colons_still_run(self) -> None:
        """A name that is illegal as a Windows path must not drop a challenge."""
        challenge = make_challenge(self.root, "Breach: Zero Day", "forensics")
        eval_dir = self.root / "eval"
        seen: list[str] = []

        def factory(challenge, run_dir):
            seen.append(run_dir.name)
            return FakeCoordinator(report(False), run_dir)

        summary = run_eval(
            [challenge],
            eval_dir=eval_dir,
            coordinator_factory=factory,
            config=EvalConfig(lane_models=["m1"], concurrency=1),
            quiet=True,
        )

        self.assertEqual(summary["planned"], 1)
        self.assertEqual(summary["challenges"], 1)
        self.assertEqual(len(summary["results"]), 1)
        self.assertEqual(summary["results"][0]["challenge"], "Breach: Zero Day")
        self.assertNotIn(":", seen[0])
        self.assertEqual(seen[0], run_dir_name(1, challenge))
        self.assertTrue((eval_dir / seen[0] / "race-summary.json").is_file())

    def test_unexpected_harness_error_is_recorded_not_dropped(self) -> None:
        """Even a failure outside the race must show up as an error result."""
        challenge = make_challenge(self.root, "corrupt", "misc")
        eval_dir = self.root / "eval"
        run_dir = eval_dir / run_dir_name(1, challenge)
        run_dir.mkdir(parents=True)
        (run_dir / "race-summary.json").write_text("{not json", encoding="utf-8")

        summary = run_eval(
            [challenge],
            eval_dir=eval_dir,
            coordinator_factory=lambda challenge, run_dir: FakeCoordinator(
                report(False), run_dir
            ),
            config=EvalConfig(
                lane_models=["m1"], concurrency=1, skip_existing=True
            ),
            quiet=True,
        )

        self.assertEqual(summary["planned"], 1)
        self.assertEqual(summary["challenges"], 1)
        self.assertIn("JSONDecodeError", summary["results"][0]["error"])

    def test_safe_dir_name_handles_windows_hostile_names(self) -> None:
        self.assertEqual(safe_dir_name("Breach: Zero Day"), "Breach- Zero Day")
        self.assertEqual(safe_dir_name("a/b\\c"), "a-b-c")
        self.assertEqual(safe_dir_name("CON"), "CON_")
        self.assertEqual(safe_dir_name("..."), "challenge")

    def test_exclusive_resources_reads_compose(self) -> None:
        directory = self.root / "chal"
        directory.mkdir()
        (directory / "docker-compose.yml").write_text(
            "services:\n  app:\n    container_name: fixed_name\n    ports:\n"
            '      - "21009:21009"\n      - 22333:21004\n',
            encoding="utf-8",
        )

        resources = exclusive_resources(make_challenge(directory, "chal", "misc"))

        self.assertEqual(
            resources, {"port:21009", "port:22333", "container:fixed_name"}
        )


if __name__ == "__main__":
    unittest.main()
