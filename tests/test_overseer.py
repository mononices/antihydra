"""Overseer tests with a scripted LLM (no network)."""

from __future__ import annotations

import unittest
from pathlib import Path

from antihydra.challenge import Challenge
from antihydra.llm import Reply
from antihydra.overseer import (
    Overseer,
    OverseerConfig,
    build_digest,
    evaluate_triggers,
    parse_advice,
)
from antihydra.racing import LaneState


class ScriptedLLM:
    model = "fake/overseer"

    def __init__(self, replies: list[Reply]) -> None:
        self.replies = list(replies)
        self.calls = 0

    def chat(self, messages, tools=None, **kwargs) -> Reply:
        self.calls += 1
        return self.replies.pop(0)


def reply(content: str, cost: float = 0.01) -> Reply:
    return Reply(content=content, tool_calls=[], cost=cost)


class OverseerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.challenge = Challenge(
            directory=Path("."),
            metadata={
                "name": "demo",
                "category": "pwn",
                "flag": "csawctf{x}",
                "description": "demo challenge",
                "box": "pwn.chal.csaw.io",
                "internal_port": 21003,
            },
        )

    def test_parse_advice_variants(self) -> None:
        self.assertEqual(parse_advice('{"action": "none"}').action, "none")
        hint = parse_advice(
            '```json\n{"action": "hint", "target": "lane-1", "text": "try X"}\n```'
        )
        self.assertEqual(hint.text, "try X")
        inspect = parse_advice(
            '{"action": "inspect", "target": "lane-2", "query": "why?"}'
        )
        self.assertEqual(inspect.action, "inspect")
        self.assertIsNone(parse_advice("not json"))
        self.assertIsNone(parse_advice(None))

    def test_evaluate_triggers(self) -> None:
        config = OverseerConfig()
        states = [
            LaneState(
                label="lane-1", model="m", rounds=50, remote_attempts=0, started_at=1.0
            ),
            LaneState(
                label="lane-2",
                model="m",
                rounds=10,
                remote_attempts=1,
                last_tool_at=1000.0,
                failed_submits=2,
            ),
        ]
        triggers = " | ".join(
            evaluate_triggers(config, states, {"count": 0, "reads": 3}, now=2000.0)
        )

        self.assertIn("no remote contact after 50 rounds", triggers)
        self.assertIn("failed submits: 2", triggers)
        self.assertIn("lane-2 stalled", triggers)
        self.assertIn("bus starved", triggers)

    def test_build_digest_contains_lane_details(self) -> None:
        digest = build_digest(
            challenge=self.challenge,
            states=[
                LaneState(
                    label="lane-1",
                    model="glm",
                    rounds=12,
                    cost=0.05,
                    recent=["run_command: objdump"],
                    notes=["offset 144"],
                )
            ],
            bus_stats={"count": 0, "reads": 2},
            triggers=["lane-1 stalled"],
            elapsed_seconds=300,
            lane_spend=0.05,
            overseer_spend=0.0,
        )

        self.assertIn("demo (pwn) target=pwn.chal.csaw.io:21003", digest)
        self.assertIn("offset 144", digest)
        self.assertIn("lane-1 stalled", digest)

    def test_review_returns_hint_and_tracks_cost(self) -> None:
        llm = ScriptedLLM(
            [reply('{"action": "hint", "target": "lane-1", "text": "use %27$p"}')]
        )
        overseer = Overseer(llm, OverseerConfig())

        advice = overseer.review("digest")

        self.assertEqual(advice.action, "hint")
        self.assertEqual(advice.target, "lane-1")
        self.assertEqual(overseer.calls, 1)
        self.assertAlmostEqual(overseer.spent, 0.01)

    def test_review_respects_call_cap(self) -> None:
        llm = ScriptedLLM([reply('{"action": "none"}')])
        overseer = Overseer(llm, OverseerConfig(max_calls=0))

        self.assertIsNone(overseer.review("digest"))
        self.assertEqual(llm.calls, 0)

    def test_review_survives_broken_output(self) -> None:
        llm = ScriptedLLM([reply("sorry, no json"), reply('{"action": "none"}')])
        overseer = Overseer(llm, OverseerConfig())

        self.assertIsNone(overseer.review("digest"))
        self.assertEqual(overseer.calls, 1)
        self.assertEqual(overseer.review("digest").action, "none")


if __name__ == "__main__":
    unittest.main()
