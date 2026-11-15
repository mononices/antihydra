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
    check_hint_grounding,
    evidence_grounded,
    evaluate_triggers,
    parse_advice,
    similar_hint,
    unverified_addresses,
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

    def test_advice_carries_evidence_and_strips_own_tag(self) -> None:
        advice = parse_advice(
            '{"action": "hint", "target": "lane-1", "text": "[overseer] try X", '
            '"evidence": "lane-1 stalled: no tool call for 120s"}'
        )

        self.assertEqual(advice.text, "try X")
        self.assertEqual(advice.evidence, "lane-1 stalled: no tool call for 120s")

    def test_grounding_rejects_invented_connectivity(self) -> None:
        digest = "lane-1 (glm) status=running rounds=40 cost=$0.10 remote=0 failed_submits=0"
        # The exact hint that derailed the Music Security Department race.
        report = check_hint_grounding(
            text="Peers confirmed the target is up: TCP 4562 accepts connections; "
            "connect to challenge:4562",
            evidence="lane-1 (glm) status=running rounds=40",
            grounds=digest,
            verified=[],
        )

        self.assertFalse(report.ok)
        self.assertIn("connectivity claim", report.reason)

    def test_grounding_rejects_unverified_address_in_a_claim(self) -> None:
        report = check_hint_grounding(
            text="the admin relay at admin-relay:5000 listens for requests",
            evidence="lane-1 (glm) status=running rounds=40",
            grounds="lane-1 (glm) status=running rounds=40",
            verified=["low-tide.chal.csaw.io:80"],
        )

        self.assertFalse(report.ok)
        self.assertIn("admin-relay:5000", report.reason)

    def test_grounding_accepts_verified_target_claim(self) -> None:
        grounds = (
            "verified targets: low-tide-lb.ctf.csaw.io:8080\n"
            "lane-3 (ds) status=running rounds=22 cost=$0.05 remote=1"
        )
        report = check_hint_grounding(
            text="low-tide-lb.ctf.csaw.io:8080 is reachable and answers /station/checkin",
            evidence="lane-3 (ds) status=running rounds=22 cost=$0.05 remote=1",
            grounds=grounds,
            verified=["low-tide-lb.ctf.csaw.io:8080"],
        )

        self.assertTrue(report.ok, report.reason)

    def test_grounding_requires_a_real_quote(self) -> None:
        digest = "lane-1 (glm) status=running rounds=40 remote=0"
        invented_evidence = check_hint_grounding(
            text="try the tcache poisoning approach next",
            evidence="lane-2 found a double free in the retire path",
            grounds=digest,
            verified=[],
        )
        unsupported = check_hint_grounding(
            text="try the tcache poisoning approach next",
            evidence=None,
            grounds=digest,
            verified=[],
        )
        # The fallback only passes hints that reuse the digest's own vocabulary;
        # real hints should carry a quote, which is why the prompt demands one.
        echoed = check_hint_grounding(
            text="lane-1 status running rounds 40 remote 0",
            evidence=None,
            grounds=digest,
            verified=[],
        )

        self.assertFalse(invented_evidence.ok)
        self.assertIn("evidence", invented_evidence.reason)
        self.assertFalse(unsupported.ok)
        self.assertIn("no basis", unsupported.reason)
        self.assertTrue(echoed.ok, echoed.reason)
        self.assertTrue(
            evidence_grounded("lane-1 (glm) status=running rounds=40", digest)
        )

    def test_grounding_rejects_repeats(self) -> None:
        quote = "lane-1 (glm) status=running rounds=40 remote=0"
        first = "free a charge then rewire the tcache fd to __free_hook and defuse /bin/sh"
        second = "free a charge then rewire the tcache fd to __free_hook then defuse /bin/sh"
        report = check_hint_grounding(
            text=second,
            evidence=quote,
            grounds=quote,
            verified=[],
            previous=[first],
        )

        self.assertTrue(similar_hint(first, second))
        self.assertFalse(report.ok)
        self.assertIn("repeats", report.reason)

    def test_unverified_addresses_skips_known_targets(self) -> None:
        found = unverified_addresses(
            "connect to low-tide.chal.csaw.io:80 and then scan 172.18.0.9:5000",
            {"low-tide.chal.csaw.io:80"},
        )

        self.assertEqual(found, ["172.18.0.9:5000"])


if __name__ == "__main__":
    unittest.main()
