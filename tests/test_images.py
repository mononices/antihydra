"""Multimodal image input, command-timeout ceilings and peer-hint pushes."""

from __future__ import annotations

import base64
import json
import tempfile
import time
import unittest
from pathlib import Path

from antihydra.agent import Agent
from antihydra.blackboard import Blackboard
from antihydra.challenge import Challenge
from antihydra.compaction import estimate_tokens
from antihydra.hints import HintBus
from antihydra.images import (
    AttachedImage,
    parse_crop,
    redact_image_parts,
    sniff_media_type,
    text_only_parts,
)
from antihydra.llm import Reply, ToolCallRequest
from antihydra.sandbox import ExecResult, SandboxError
from antihydra.tools import AgentContext, CostLedger, RunState, build_main_tools

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


class FakeSandbox:
    """In-memory sandbox with blobs, byte reads and recorded command timeouts."""

    image = "fake:test"
    container_id = "fake"
    shell = "sh"

    def __init__(self, blobs: dict[str, bytes] | None = None) -> None:
        self.files: dict[str, str] = {}
        self.blobs = dict(blobs or {})
        self.commands: list[tuple[str, float]] = []

    def file_size(self, path: str) -> int:
        if path in self.blobs:
            return len(self.blobs[path])
        if path in self.files:
            return len(self.files[path].encode())
        return -1

    def read_bytes(self, path: str, limit: int = 4_000_000) -> bytes:
        if path not in self.blobs:
            raise SandboxError(f"no such file: {path}")
        data = self.blobs[path]
        if len(data) > limit:
            raise SandboxError("too large")
        return data

    def read_text(self, path: str, limit: int | None = None) -> str:
        if path not in self.files:
            raise SandboxError(f"no such file: {path}")
        text = self.files[path]
        return text if limit is None else text[:limit]

    def write_text(self, path: str, content: str) -> None:
        self.files[path] = content

    def exec(self, command: str, timeout: float = 300) -> ExecResult:
        self.commands.append((command, timeout))
        return ExecResult(stdout="", stderr="", exit_code=0, timed_out=False)


class CapturingLLM:
    model = "fake/model"

    def __init__(self, replies: list[Reply]) -> None:
        self.replies = list(replies)
        self.seen: list[list[dict]] = []

    def supports_images(self) -> bool | None:
        return True

    def chat(self, messages, tools=None, **kwargs) -> Reply:
        self.seen.append([dict(message) for message in messages])
        if not self.replies:
            raise AssertionError("CapturingLLM ran out of scripted replies")
        return self.replies.pop(0)


def reply(*calls, content=None, cost=0.01) -> Reply:
    return Reply(
        content=content,
        tool_calls=[
            ToolCallRequest(
                id=f"call_{index}",
                name=name,
                arguments=arguments or {},
                raw_arguments=json.dumps(arguments or {}),
            )
            for index, (name, arguments) in enumerate(calls)
        ],
        cost=cost,
    )


class ImageHelperTests(unittest.TestCase):
    def test_sniff_media_type_ignores_extensions(self) -> None:
        self.assertEqual(sniff_media_type(PNG_BYTES), "image/png")
        self.assertEqual(sniff_media_type(b"\xff\xd8\xff\xe0rest"), "image/jpeg")
        self.assertEqual(sniff_media_type(b"RIFF0000WEBPVP8 "), "image/webp")
        self.assertIsNone(sniff_media_type(b"PK\x03\x04zip"))

    def test_crop_accepts_common_spellings(self) -> None:
        self.assertEqual(parse_crop("10,20,30,40"), "30x40+10+20")
        self.assertEqual(parse_crop(" 10x20x30x40 "), "30x40+10+20")
        self.assertIsNone(parse_crop(""))
        with self.assertRaises(ValueError):
            parse_crop("1,2,3")

    def test_redact_and_text_only_parts(self) -> None:
        image = AttachedImage(
            path="/tmp/a.png", media_type="image/png", payload="A" * 400
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}, image.content_part()]}]
        redacted = redact_image_parts(messages)
        url = redacted[0]["content"][1]["image_url"]["url"]
        self.assertIn("omitted", url)
        self.assertLess(len(url), 100)
        stripped = text_only_parts(messages[0]["content"])
        self.assertFalse(any(part.get("type") == "image_url" for part in stripped))

    def test_estimate_tokens_counts_image_payloads(self) -> None:
        image = AttachedImage(
            path="/tmp/a.png", media_type="image/png", payload="A" * 4000
        )
        with_image = [{"role": "user", "content": [image.content_part()]}]
        without = [{"role": "user", "content": "tiny"}]
        self.assertGreater(estimate_tokens(with_image), estimate_tokens(without) + 500)


class ViewImageFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.challenge = Challenge(
            directory=self.root,
            metadata={
                "name": "stego demo",
                "category": "crypto",
                "flag": "csawctf{unit_test_flag_123}",
                "description": "demo challenge",
                "files": [],
            },
        )
        self.sandbox = FakeSandbox({"/tmp/frame.png": PNG_BYTES})
        self.blackboard = Blackboard(self.sandbox, path="/notes.md")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_view_image_reaches_the_next_request(self) -> None:
        llm = CapturingLLM(
            [
                reply(("view_image", {"path": "/tmp/frame.png", "question": "any text?"})),
                reply(("submit_flag", {"flag": self.challenge.flag})),
            ]
        )
        agent = Agent(
            self.challenge,
            self.sandbox,
            llm,
            build_main_tools(self.blackboard),
            ledger=CostLedger(),
            blackboard=self.blackboard,
            log_dir=self.root / "logs",
            quiet=True,
        )
        result = agent.run()

        self.assertTrue(result.solved)
        self.assertEqual(result.images, 1)
        second_request = llm.seen[1]
        image_messages = [
            message
            for message in second_request
            if isinstance(message.get("content"), list)
            and any(
                part.get("type") == "image_url"
                for part in message["content"]
                if isinstance(part, dict)
            )
        ]
        self.assertEqual(len(image_messages), 1)
        parts = image_messages[0]["content"]
        self.assertEqual(parts[0]["type"], "text")
        self.assertIn("any text?", parts[0]["text"])
        self.assertTrue(parts[1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(parts[1]["image_url"]["detail"], "auto")

        transcript = json.loads(Path(result.transcript_path).read_text(encoding="utf-8"))
        self.assertTrue(any(event["kind"] == "image" for event in transcript["events"]))
        saved = json.dumps(transcript["messages"])
        self.assertNotIn("iVBORw0KGgo", saved)

    def test_view_image_refuses_unsupported_files(self) -> None:
        self.sandbox.blobs["/tmp/notes.txt"] = b"not an image at all"
        llm = CapturingLLM(
            [
                reply(("view_image", {"path": "/tmp/notes.txt"})),
                reply(("submit_flag", {"flag": self.challenge.flag})),
            ]
        )
        agent = Agent(
            self.challenge,
            self.sandbox,
            llm,
            build_main_tools(self.blackboard),
            blackboard=self.blackboard,
            log_dir=self.root / "logs",
            quiet=True,
        )
        result = agent.run()
        self.assertEqual(result.images, 0)
        tool_messages = [
            message
            for message in json.loads(
                Path(result.transcript_path).read_text(encoding="utf-8")
            )["messages"]
            if message.get("role") == "tool"
        ]
        self.assertTrue(any("error" in str(m.get("content")) for m in tool_messages))


class TimeoutCeilingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.challenge = Challenge(
            directory=self.root,
            metadata={
                "name": "demo",
                "category": "crypto",
                "flag": "csawctf{unit_test_flag_123}",
                "description": "demo",
                "files": [],
            },
        )
        self.sandbox = FakeSandbox()
        self.registry = build_main_tools(None)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self, *, deadline: float | None, ceiling: float | None, timeout: float):
        context = AgentContext(
            sandbox=self.sandbox,
            challenge=self.challenge,
            state=RunState(),
            default_timeout=300.0,
            deadline=deadline,
            max_command_timeout=ceiling,
        )
        observation = self.registry.call(
            "run_command", {"command": "sleep 999", "timeout": timeout}, context
        )
        return json.loads(observation)

    def test_ceiling_wins_over_a_long_request(self) -> None:
        payload = self._run(deadline=None, ceiling=30.0, timeout=1700)
        self.assertEqual(payload["timeout_seconds"], 30.0)
        self.assertIn("clamped", payload["timeout_note"])

    def test_remaining_budget_clamps_the_command(self) -> None:
        payload = self._run(deadline=time.monotonic() + 60, ceiling=None, timeout=900)
        self.assertLessEqual(payload["timeout_seconds"], 60)
        self.assertGreater(payload["timeout_seconds"], 5)

    def test_expired_budget_refuses_new_commands(self) -> None:
        payload = self._run(deadline=time.monotonic(), ceiling=None, timeout=300)
        self.assertIn("error", payload)
        self.assertEqual(self.sandbox.commands, [])


class PeerHintPushTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.challenge = Challenge(
            directory=self.root,
            metadata={
                "name": "demo",
                "category": "crypto",
                "flag": "csawctf{unit_test_flag_123}",
                "description": "demo",
                "files": [],
            },
        )
        self.sandbox = FakeSandbox()
        self.blackboard = Blackboard(self.sandbox, path="/notes.md")
        self.bus = HintBus()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_fresh_peer_hints_are_injected_without_a_tool_call(self) -> None:
        self.bus.publish(source="lane-1", text="the payload sits in the uuid box")
        llm = CapturingLLM([reply(("submit_flag", {"flag": self.challenge.flag}))])
        agent = Agent(
            self.challenge,
            self.sandbox,
            llm,
            build_main_tools(self.blackboard, hints=self.bus),
            blackboard=self.blackboard,
            hints=self.bus,
            bus_source="lane-2",
            log_dir=self.root / "logs",
            quiet=True,
        )
        result = agent.run()

        first_request = llm.seen[0]
        injected = [
            message
            for message in first_request
            if isinstance(message.get("content"), str)
            and "uuid box" in message["content"]
        ]
        self.assertEqual(len(injected), 1)
        self.assertIn("lane-1", injected[0]["content"])
        self.assertGreaterEqual(self.bus.stats()["reads"], 1)
        transcript = json.loads(Path(result.transcript_path).read_text(encoding="utf-8"))
        self.assertTrue(any(event["kind"] == "peer_hints" for event in transcript["events"]))

    def test_check_hints_still_pulls_a_new_hint(self) -> None:
        registry = build_main_tools(self.blackboard, hints=self.bus)
        context = AgentContext(
            sandbox=self.sandbox,
            challenge=self.challenge,
            state=RunState(),
            hints=self.bus,
            bus_source="lane-2",
        )
        self.bus.publish(source="lane-1", text="first hint")
        first = json.loads(registry.call("check_hints", {}, context))
        self.assertEqual([hint["text"] for hint in first["hints"]], ["first hint"])
        self.bus.publish(source="lane-1", text="second hint")
        second = json.loads(registry.call("check_hints", {}, context))
        self.assertEqual([hint["text"] for hint in second["hints"]], ["second hint"])

    def test_own_hints_are_not_echoed_back(self) -> None:
        self.bus.publish(source="lane-2", text="my own finding")
        llm = CapturingLLM([reply(("submit_flag", {"flag": self.challenge.flag}))])
        agent = Agent(
            self.challenge,
            self.sandbox,
            llm,
            build_main_tools(self.blackboard, hints=self.bus),
            blackboard=self.blackboard,
            hints=self.bus,
            bus_source="lane-2",
            log_dir=self.root / "logs",
            quiet=True,
        )
        agent.run()
        self.assertFalse(
            any("my own finding" in str(m.get("content")) for m in llm.seen[0])
        )


class ChatPayloadTests(unittest.TestCase):
    def test_image_parts_survive_the_request_builder(self) -> None:
        from antihydra.llm import OpenRouterLLM

        llm = OpenRouterLLM("vendor/model", "test-key", catalog={}, retries=0)
        captured: dict = {}

        def fake_post(payload: dict) -> dict:
            captured.update(payload)
            return {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
            }

        llm._post = fake_post  # type: ignore[method-assign]
        image = AttachedImage(
            path="/tmp/a.png", media_type="image/png", payload="QUJD", detail="high"
        )
        llm.chat(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "read this"},
                        image.content_part(),
                    ],
                }
            ]
        )

        parts = captured["messages"][0]["content"]
        self.assertEqual(parts[0], {"type": "text", "text": "read this"})
        self.assertEqual(parts[1]["type"], "image_url")
        self.assertEqual(parts[1]["image_url"]["url"], "data:image/png;base64,QUJD")
        self.assertEqual(parts[1]["image_url"]["detail"], "high")


if __name__ == "__main__":
    unittest.main()
