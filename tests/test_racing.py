"""Racing tests with fake lane runners (no docker, no network)."""

from __future__ import annotations

import _thread
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from csaw26.challenge import Challenge
from csaw26.overseer import Advice, OverseerConfig
from csaw26.racing import (
    LaneResult,
    LaneSpec,
    LaneState,
    RaceCoordinator,
    load_lane_specs,
)


class FakeLane:
    """Stand-in for a Lane with scripted behavior."""

    def __init__(self, spec: LaneSpec, behavior: dict, emit, hints=None) -> None:
        self.spec = spec
        self.behavior = behavior
        self.emit = emit
        self.hints = hints
        self.aborted = threading.Event()
        self.force_stopped = False
        self.delivered: list[str] = []

    def run(self) -> LaneResult:
        self.emit(f"{self.spec.label} working")
        if self.behavior.get("publish") and self.hints is not None:
            self.hints.publish(source=self.spec.label, text=self.behavior["publish"])
        if self.behavior.get("interrupt"):
            time.sleep(self.behavior.get("delay", 0.05))
            _thread.interrupt_main()
        if self.behavior.get("wait_for_abort"):
            self.aborted.wait(timeout=self.behavior.get("timeout", 30))
            return LaneResult(spec=self.spec, status="aborted")
        time.sleep(self.behavior.get("delay", 0))
        return LaneResult(
            spec=self.spec,
            status=self.behavior["status"],
            cost=self.behavior.get("cost", 0.0),
            rounds=self.behavior.get("rounds", 0),
        )

    def abort(self) -> None:
        self.aborted.set()

    def force_stop(self) -> None:
        self.force_stopped = True
        self.aborted.set()

    def deliver(self, text: str) -> None:
        self.delivered.append(text)

    def state(self) -> LaneState:
        return LaneState(
            label=self.spec.label,
            model=self.spec.model,
            status="running",
            rounds=self.behavior.get("rounds", 0),
            cost=self.behavior.get("cost", 0.0),
        )

    def notes(self) -> list[str]:
        return []

    def recent_slice(self, limit: int = 30) -> str:
        return "run_command; -> output"


class RacingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.challenge = Challenge(
            directory=self.root,
            metadata={
                "name": "demo",
                "category": "pwn",
                "flag": "csawctf{x}",
                "description": "demo",
            },
        )
        self.fakes: dict[int, FakeLane] = {}

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def make_coordinator(
        self, specs: list[LaneSpec], behaviors: dict[int, dict], **kwargs
    ) -> RaceCoordinator:
        def lane_factory(spec, lane_dir, emit, hints):
            lane = FakeLane(spec, behaviors[spec.id], emit, hints)
            self.fakes[spec.id] = lane
            return lane

        return RaceCoordinator(
            self.challenge,
            specs,
            race_dir=self.root / "race",
            lane_factory=lane_factory,
            start_service=False,
            follow=False,
            quiet=True,
            poll_seconds=0.02,
            **kwargs,
        )

    def test_first_solved_lane_wins_and_others_abort(self) -> None:
        specs = [LaneSpec(id=1, model="fake/fast"), LaneSpec(id=2, model="fake/slow")]
        report = self.make_coordinator(
            specs,
            {
                1: {"delay": 0.05, "status": "solved", "cost": 0.1, "rounds": 5},
                2: {"wait_for_abort": True},
            },
        ).run()

        self.assertTrue(report["solved"])
        self.assertEqual(report["solved_by"], 1)
        self.assertEqual(report["pass_at_k"], 1)
        self.assertTrue(self.fakes[2].aborted.is_set())
        lane2 = next(lane for lane in report["lanes"] if lane["id"] == 2)
        self.assertEqual(lane2["status"], "aborted")

    def test_all_lanes_failing_reports_unsolved(self) -> None:
        specs = [LaneSpec(id=1, model="fake/a"), LaneSpec(id=2, model="fake/b")]
        report = self.make_coordinator(
            specs,
            {
                1: {"delay": 0.02, "status": "unsolved"},
                2: {"delay": 0.05, "status": "error", "cost": 0.3},
            },
        ).run()

        self.assertFalse(report["solved"])
        self.assertEqual(report["pass_at_k"], 0)
        statuses = sorted(lane["status"] for lane in report["lanes"])
        self.assertEqual(statuses, ["error", "unsolved"])
        self.assertAlmostEqual(report["total_cost"], 0.3)

        summary = json.loads(
            (self.root / "race" / "race-summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["k"], 2)
        self.assertFalse(summary["solved"])

    def test_lane_output_lands_in_run_log(self) -> None:
        specs = [LaneSpec(id=1, model="fake/a")]
        self.make_coordinator(specs, {1: {"delay": 0.02, "status": "unsolved"}}).run()

        lane_dir = self.root / "race" / "lane-1"
        self.assertIn(
            "lane-1 working", (lane_dir / "run.log").read_text(encoding="utf-8")
        )
        meta = json.loads((lane_dir / "lane.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["reasoning_effort"], "max")

    def test_lane_specs_from_json_file(self) -> None:
        lanes_file = self.root / "lanes.json"
        lanes_file.write_text(
            json.dumps(
                [
                    {"model": "a/b", "temperature": 1.0, "max_cost": 2.0},
                    {"model": "c/d", "no_subtasks": True},
                ]
            ),
            encoding="utf-8",
        )
        specs = load_lane_specs([], lanes_file)

        self.assertEqual([spec.model for spec in specs], ["a/b", "c/d"])
        self.assertEqual(specs[0].temperature, 1.0)
        self.assertEqual(specs[0].reasoning_effort, "max")
        self.assertEqual(specs[0].max_cost, 2.0)
        self.assertTrue(specs[1].no_subtasks)

    def test_lane_specs_from_toml_with_defaults(self) -> None:
        lanes_file = self.root / "lanes.toml"
        lanes_file.write_text(
            "[defaults]\n"
            "temperature = 0.9\n"
            'reasoning_effort = "low"\n'
            "max_cost = 3.0\n"
            "\n"
            "[[lanes]]\n"
            'model = "a/b"\n'
            "\n"
            "[[lanes]]\n"
            'model = "c/d"\n'
            "temperature = 0.1\n"
            'reasoning_effort = "high"\n',
            encoding="utf-8",
        )
        specs = load_lane_specs([], lanes_file)

        self.assertEqual([spec.model for spec in specs], ["a/b", "c/d"])
        self.assertEqual(specs[0].temperature, 0.9)
        self.assertEqual(specs[0].reasoning_effort, "low")
        self.assertEqual(specs[0].max_cost, 3.0)
        self.assertEqual(specs[0].max_rounds, 200)
        self.assertEqual(specs[1].temperature, 0.1)
        self.assertEqual(specs[1].reasoning_effort, "high")

    def test_cli_values_override_file_defaults(self) -> None:
        lanes_file = self.root / "lanes.toml"
        lanes_file.write_text(
            '[defaults]\ntemperature = 0.9\nreasoning_effort = "high"\n\n'
            '[[lanes]]\nmodel = "a/b"\n',
            encoding="utf-8",
        )
        specs = load_lane_specs([], lanes_file, temperature=0.2, reasoning_effort="low")

        self.assertEqual(specs[0].temperature, 0.2)
        self.assertEqual(specs[0].reasoning_effort, "low")

    def test_invalid_reasoning_effort_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_lane_specs(["a/b"], None, reasoning_effort="turbo")

    def test_interrupt_aborts_lanes_and_reports(self) -> None:
        specs = [LaneSpec(id=1, model="fake/a")]
        report = self.make_coordinator(
            specs,
            {1: {"interrupt": True, "wait_for_abort": True}},
        ).run()

        self.assertTrue(report["interrupted"])
        self.assertFalse(report["solved"])
        self.assertEqual(report["lanes"][0]["status"], "aborted")
        self.assertTrue((self.root / "race" / "race-summary.json").is_file())

    def test_lanes_publish_to_the_hint_bus(self) -> None:
        specs = [LaneSpec(id=1, model="fake/a"), LaneSpec(id=2, model="fake/b")]
        report = self.make_coordinator(
            specs,
            {
                1: {"delay": 0.02, "status": "unsolved", "publish": "offset is 40"},
                2: {"delay": 0.05, "status": "unsolved"},
            },
        ).run()

        self.assertEqual(report["hints"]["count"], 1)
        self.assertEqual(report["hints"]["sources"], {"lane-1": 1})

    def test_overseer_hint_is_pushed_to_the_target_lane(self) -> None:
        class StubOverseer:
            def __init__(self) -> None:
                self.config = OverseerConfig(
                    interval_seconds=600.0, min_gap_seconds=600.0
                )
                self.spent = 0.0
                self.calls = 0

            def review(self, digest, *, inspect_target=None, inspect_slice=""):
                self.calls += 1
                return Advice(
                    action="hint", target="lane-1", text="use %27$p for the leak"
                )

        specs = [LaneSpec(id=1, model="fake/a"), LaneSpec(id=2, model="fake/b")]
        report = self.make_coordinator(
            specs,
            {
                1: {"delay": 0.4, "status": "unsolved"},
                2: {"delay": 0.4, "status": "unsolved"},
            },
            overseer=StubOverseer(),
        ).run()

        self.assertEqual(len(self.fakes[1].delivered), 1)
        self.assertIn("use %27$p", self.fakes[1].delivered[0])
        self.assertEqual(self.fakes[2].delivered, [])
        self.assertEqual(report["overseer"]["hints"], 1)
        self.assertEqual(report["hints"]["sources"], {"overseer": 1})

    def test_lane_specs_require_a_source(self) -> None:
        with self.assertRaises(ValueError):
            load_lane_specs([], None)


if __name__ == "__main__":
    unittest.main()
