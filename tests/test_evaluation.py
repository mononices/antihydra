"""Evaluation harness tests with fake coordinators (no docker, no network)."""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from antihydra.challenge import Challenge
from antihydra.evaluation import EvalConfig, exclusive_resources, run_eval


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
