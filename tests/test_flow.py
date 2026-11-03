"""End-to-end flow tests with a fake sandbox and scripted LLM.

These cover the main-agent/tool loop, the markdown findings board, subtask
spawning, and cost accounting without docker or network access.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from csaw26.agent import Agent
from csaw26.blackboard import Blackboard
from csaw26.challenge import Challenge
from csaw26.compaction import CompactionConfig, safe_cut_index
from csaw26.hints import HintBus
from csaw26.llm import Reply, ToolCallRequest
from csaw26.sandbox import ExecResult, SandboxError
from csaw26.stalls import StallConfig, fingerprint
from csaw26.tools import CostLedger, build_main_tools


class FakeSandbox:
    """In-memory stand-in for the docker sandbox."""

    image = "fake:test"
    container_id = "fake"
    shell = "sh"

    def __init__(self) -> None:
        self.files: dict[str, str] = {}

    def read_text(self, path: str, limit: int | None = None) -> str:
        if path not in self.files:
            raise SandboxError(f"no such file: {path}")
        text = self.files[path]
        return text if limit is None else text[:limit]

    def write_text(self, path: str, content: str) -> None:
        self.files[path] = content

    def exec(self, command: str, timeout: float = 300) -> ExecResult:
        return ExecResult(stdout="", stderr="", exit_code=0, timed_out=False)


class FakeLLM:
    """Pops scripted replies in order; shared between main agent and workers."""

    model = "fake/model"

    def __init__(self, replies: list) -> None:
        self.replies = list(replies)
        self.calls = 0

    def chat(self, messages, tools=None, **kwargs) -> Reply:
        self.calls += 1
        if not self.replies:
            raise AssertionError("FakeLLM ran out of scripted replies")
        return self.replies.pop(0)


def reply(*calls, content=None, cost=0.01) -> Reply:
    tool_calls = [
        ToolCallRequest(
            id=f"call_{index}",
            name=name,
            arguments=arguments or {},
            raw_arguments=json.dumps(arguments or {}),
        )
        for index, (name, arguments) in enumerate(calls)
    ]
    return Reply(content=content, tool_calls=tool_calls, cost=cost)


class FlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.challenge = Challenge(
            directory=self.root,
            metadata={
                "name": "demo",
                "category": "crypto",
                "flag": "csawctf{unit_test_flag_123}",
                "description": "demo challenge",
                "files": [],
            },
        )
        self.sandbox = FakeSandbox()
        self.blackboard = Blackboard(self.sandbox, path="/notes.md")
        self.ledger = CostLedger()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def make_agent(self, script: list, **kwargs) -> Agent:
        return Agent(
            self.challenge,
            self.sandbox,
            FakeLLM(script),
            build_main_tools(self.blackboard),
            ledger=self.ledger,
            blackboard=self.blackboard,
            log_dir=self.root / "logs",
            quiet=True,
            **kwargs,
        )

    def test_subtask_records_findings_then_main_submits(self) -> None:
        script = [
            reply(("spawn_subtask", {"task": "find the offset"})),
            reply(("record_finding", {"text": "offset is 40"})),
            reply(("finish_task", {"summary": "offset is 40; next: write exploit"})),
            reply(("submit_flag", {"flag": self.challenge.flag})),
        ]
        result = self.make_agent(script).run()

        self.assertTrue(result.solved)
        self.assertEqual(result.reason, "solved")
        board = self.blackboard.read()
        self.assertTrue(board.startswith("# Findings"))
        self.assertIn("- [fact] (subtask-1) offset is 40", board)
        self.assertIn(
            "- [subtask] (subtask-1) offset is 40; next: write exploit", board
        )
        self.assertAlmostEqual(self.ledger.total, 0.04)

        logs = sorted((self.root / "logs").glob("*.json"))
        self.assertTrue(any("_main_" in log.name for log in logs))
        self.assertTrue(any("_subtask-1_" in log.name for log in logs))

        main_log = next(log for log in logs if "_main_" in log.name)
        data = json.loads(main_log.read_text(encoding="utf-8"))
        self.assertTrue(any(event["kind"] == "subtask" for event in data["events"]))
        self.assertIn("notes", data)

    def test_worker_can_solve_and_propagate(self) -> None:
        script = [
            reply(("spawn_subtask", {"task": "grab the flag"})),
            reply(("submit_flag", {"flag": self.challenge.flag})),
        ]
        result = self.make_agent(script).run()

        self.assertTrue(result.solved)
        self.assertEqual(result.reason, "solved")
        self.assertEqual(result.flag, self.challenge.flag)
        self.assertAlmostEqual(self.ledger.total, 0.02)

    def test_blackboard_is_simple_markdown(self) -> None:
        self.blackboard.append("alpha offset", kind="fact", source="main")
        self.blackboard.append("beta path failed", kind="deadend", source="subtask-1")
        self.blackboard.append("gamma key recovered", kind="fact", source="main")

        self.assertTrue(self.blackboard.read().startswith("# Findings"))
        self.assertEqual(len(self.blackboard.lines()), 3)
        self.assertEqual(len(self.blackboard.search("gamma")), 1)
        self.assertEqual(len(self.blackboard.search("BETA")), 1)
        self.assertEqual(self.blackboard.search("missing"), [])

    def test_safe_cut_stays_on_turn_boundaries(self) -> None:
        body = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]},
            {"role": "tool", "tool_call_id": "a", "content": "x"},
            {"role": "assistant", "content": "thinking"},
            {"role": "user", "content": "keep going"},
        ]
        self.assertEqual(safe_cut_index(body, keep_recent=2), 2)
        self.assertIsNone(safe_cut_index(body, keep_recent=4))
        self.assertEqual(safe_cut_index(body[:1], keep_recent=0), 0)

    def test_fingerprint_normalizes_whitespace(self) -> None:
        self.assertEqual(
            fingerprint("run_command", {"command": "echo   a  b"}),
            fingerprint("run_command", {"command": "echo a b"}),
        )
        self.assertNotEqual(
            fingerprint("run_command", {"command": "echo a"}),
            fingerprint("read_file", {"path": "/x"}),
        )

    def test_compaction_replaces_old_history(self) -> None:
        script = [
            reply(("run_command", {"command": "echo one"})),
            reply(("run_command", {"command": "echo two"})),
            reply(content="SUMMARY: tried echo one and two", cost=0.005),
            reply(("submit_flag", {"flag": self.challenge.flag})),
        ]
        agent = Agent(
            self.challenge,
            self.sandbox,
            FakeLLM(script),
            build_main_tools(self.blackboard),
            ledger=self.ledger,
            blackboard=self.blackboard,
            log_dir=self.root / "logs",
            quiet=True,
            compaction=CompactionConfig(
                enabled=True,
                at_fraction=0.0,
                tokens_fallback=1,
                keep_recent=2,
                min_dropped=2,
                min_interval=0,
                input_clip=500,
            ),
        )
        result = agent.run()
        self.assertTrue(result.solved)
        self.assertEqual(result.compactions, 1)
        self.assertAlmostEqual(self.ledger.total, 0.035)
        main_log = next((self.root / "logs").glob("*_main_*.json"))
        data = json.loads(main_log.read_text(encoding="utf-8"))
        self.assertTrue(any(event["kind"] == "compaction" for event in data["events"]))
        self.assertEqual(len(data["summaries"]), 1)
        self.assertTrue(data["compacted_messages"])
        self.assertTrue(
            any(
                isinstance(message.get("content"), str)
                and message["content"].startswith("[Earlier context summary")
                for message in data["messages"]
            )
        )

    def test_stall_detector_aborts_run(self) -> None:
        same = ("run_command", {"command": "echo hello"})
        script = [reply(same), reply(same), reply(same)]
        agent = Agent(
            self.challenge,
            self.sandbox,
            FakeLLM(script),
            build_main_tools(self.blackboard),
            ledger=self.ledger,
            blackboard=self.blackboard,
            log_dir=self.root / "logs",
            quiet=True,
            compaction=CompactionConfig(enabled=False),
            stall=StallConfig(enabled=True, repeat=2, abort_after=2),
        )
        result = agent.run()
        self.assertFalse(result.solved)
        self.assertEqual(result.reason, "stalled")
        self.assertEqual(result.stall_events, 2)
        self.assertTrue(self.blackboard.search("repeated identical call"))
        main_log = next((self.root / "logs").glob("*_main_*.json"))
        data = json.loads(main_log.read_text(encoding="utf-8"))
        self.assertTrue(any(event["kind"] == "stall" for event in data["events"]))

    def test_hint_bus_shared_between_agents(self) -> None:
        bus = HintBus()
        publisher = Agent(
            self.challenge,
            self.sandbox,
            FakeLLM(
                [
                    reply(("publish_insight", {"text": "offset is 40"})),
                    reply(("giveup", {"confirm": True})),
                ]
            ),
            build_main_tools(self.blackboard, hints=bus),
            blackboard=self.blackboard,
            hints=bus,
            bus_source="lane-1",
            log_dir=self.root / "logs",
            quiet=True,
        )
        publisher.run()

        entries = bus.since(0)
        self.assertEqual([entry.text for entry in entries], ["offset is 40"])
        self.assertEqual(entries[0].source, "lane-1")

        reader = Agent(
            self.challenge,
            self.sandbox,
            FakeLLM(
                [
                    reply(("check_hints", {})),
                    reply(("submit_flag", {"flag": self.challenge.flag})),
                ]
            ),
            build_main_tools(self.blackboard, hints=bus),
            blackboard=self.blackboard,
            hints=bus,
            bus_source="lane-2",
            log_dir=self.root / "logs",
            quiet=True,
        )
        result = reader.run()

        self.assertTrue(result.solved)
        data = json.loads(Path(result.transcript_path).read_text(encoding="utf-8"))
        tool_messages = [
            message for message in data["messages"] if message.get("role") == "tool"
        ]
        self.assertTrue(
            any(
                "offset is 40" in str(message.get("content"))
                for message in tool_messages
            )
        )


if __name__ == "__main__":
    unittest.main()
