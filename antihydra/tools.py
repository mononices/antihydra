"""Tool registry, cost ledger, and the main/worker tool profiles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from .blackboard import Blackboard
from .challenge import Challenge
from .hints import HintBus
from .sandbox import Sandbox

TOOL_OUTPUT_LIMIT = 16000


def clip(text: str, limit: int = TOOL_OUTPUT_LIMIT, head_ratio: float = 0.6) -> str:
    """Trim long output, keeping the head and tail around an elision marker."""
    if not text:
        return ""
    if len(text) <= limit:
        return text
    head_len = int(limit * head_ratio)
    tail_len = limit - head_len
    omitted = len(text) - limit
    return (
        text[:head_len]
        + f"\n... [{omitted} characters omitted] ...\n"
        + text[-tail_len:]
    )


@dataclass
class CostLedger:
    """Shared cost accumulator across the main agent and its workers."""

    total: float = 0.0

    def add(self, cost: float) -> None:
        self.total += max(float(cost), 0.0)

    def remaining(self, limit: float) -> float:
        return max(float(limit) - self.total, 0.0)


@dataclass
class RunState:
    """Mutable state shared by an agent loop and its tools."""

    solved: bool = False
    gave_up: bool = False
    stalled: bool = False
    finished: bool = False
    flag: str | None = None
    summary: str | None = None


@dataclass
class AgentContext:
    """Everything a tool needs to act."""

    sandbox: Sandbox
    challenge: Challenge
    state: RunState
    default_timeout: float = 300.0
    blackboard: Blackboard | None = None
    ledger: CostLedger | None = None
    hints: HintBus | None = None
    bus_source: str = "main"
    hint_cursor: int = 0
    actor: str = "main"
    spawn_subtask: Callable[..., dict] | None = None


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]


def _error(message: str) -> str:
    return json.dumps({"error": message})


class ToolRegistry:
    """Holds tool specs, exposes OpenAI tool schemas, and dispatches calls."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            }
            for spec in self._tools.values()
        ]

    def call(
        self, name: str, arguments: dict[str, Any] | None, context: AgentContext
    ) -> str:
        """Execute a tool call and return the observation text."""
        spec = self._tools.get(name)
        if spec is None:
            return _error(
                f"unknown tool {name!r}; available: {', '.join(self.names())}"
            )
        if arguments is None:
            return _error(
                f"arguments for {name} were not valid JSON; retry with valid JSON"
            )
        try:
            result = spec.handler(context=context, **arguments)
        except TypeError as exc:
            return _error(f"invalid arguments for {name}: {exc}")
        except Exception as exc:  # noqa: BLE001 - surfaced to the model on purpose
            return _error(f"{type(exc).__name__}: {exc}")
        if isinstance(result, str):
            return result
        return json.dumps(result)


def _sandbox_tools(registry: ToolRegistry) -> None:
    def run_command(
        context: AgentContext, command: str, timeout: float | None = None
    ) -> dict:
        effective = float(timeout) if timeout else context.default_timeout
        payload = context.sandbox.exec(command, timeout=effective).to_dict()
        payload["stdout"] = clip(payload["stdout"])
        payload["stderr"] = clip(payload["stderr"])
        return payload

    def create_file(context: AgentContext, path: str, content: str) -> dict:
        context.sandbox.write_text(path, content)
        return {"status": "written", "path": path, "bytes": len(content)}

    def read_file(context: AgentContext, path: str) -> dict:
        return {
            "path": path,
            "content": clip(context.sandbox.read_text(path, limit=TOOL_OUTPUT_LIMIT)),
        }

    registry.register(
        ToolSpec(
            name="run_command",
            description=(
                "Run a shell command inside the sandbox and return stdout, stderr, "
                "exit_code and timed_out. Challenge files are in /home/ctfplayer/ctf_files."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "shell command to run",
                    },
                    "timeout": {
                        "type": "number",
                        "description": "optional timeout in seconds",
                    },
                },
                "required": ["command"],
            },
            handler=run_command,
        )
    )
    registry.register(
        ToolSpec(
            name="create_file",
            description="Write a text file inside the sandbox (e.g. an exploit or script).",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "absolute path inside the sandbox",
                    },
                    "content": {"type": "string", "description": "file content"},
                },
                "required": ["path", "content"],
            },
            handler=create_file,
        )
    )
    registry.register(
        ToolSpec(
            name="read_file",
            description="Read a text file from the sandbox and return its contents.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "absolute path inside the sandbox",
                    }
                },
                "required": ["path"],
            },
            handler=read_file,
        )
    )


def _finding_tools(registry: ToolRegistry) -> None:
    def record_finding(context: AgentContext, text: str, kind: str = "fact") -> dict:
        if context.blackboard is None:
            return {"error": "the findings board is not available"}
        line = context.blackboard.append(text, kind=kind, source=context.actor)
        return {"status": "recorded", "line": line}

    def get_findings(context: AgentContext, query: str | None = None) -> dict:
        if context.blackboard is None:
            return {"error": "the findings board is not available"}
        lines = context.blackboard.search(query)
        return {"path": context.blackboard.path, "count": len(lines), "findings": lines}

    registry.register(
        ToolSpec(
            name="record_finding",
            description=(
                "Append a durable finding to the shared notes board so it survives "
                "context limits and other agents can use it. One concise idea per call; "
                "mention artifact paths inline."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "concise finding, one idea",
                    },
                    "kind": {
                        "type": "string",
                        "description": "short label, e.g. fact, artifact, deadend, plan",
                    },
                },
                "required": ["text"],
            },
            handler=record_finding,
        )
    )
    registry.register(
        ToolSpec(
            name="get_findings",
            description=(
                "Read the shared notes board. Optionally filter to bullets containing "
                "a substring."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "optional substring filter",
                    }
                },
            },
            handler=get_findings,
        )
    )


def _hint_tools(registry: ToolRegistry) -> None:
    def publish_insight(context: AgentContext, text: str, kind: str = "fact") -> dict:
        if context.hints is None:
            return {"error": "the hint bus is not available"}
        hint = context.hints.publish(source=context.bus_source, text=text, kind=kind)
        return {
            "status": "published",
            "id": hint.id,
            "total": context.hints.stats()["count"],
        }

    def check_hints(context: AgentContext) -> dict:
        if context.hints is None:
            return {"error": "the hint bus is not available"}
        context.hints.note_read()
        fresh = context.hints.since(
            context.hint_cursor, exclude_source=context.bus_source
        )
        context.hint_cursor = context.hints.last_id()
        shown = fresh[-10:]
        return {
            "count": len(fresh),
            "shown": len(shown),
            "hints": [hint.as_dict() for hint in shown],
        }

    registry.register(
        ToolSpec(
            name="publish_insight",
            description=(
                "Share one durable finding with the other attempts running on this "
                "challenge: a fact, an artifact path, or a dead end worth avoiding. "
                "Keep it short and specific; never publish flags or speculation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "the finding, one idea"},
                    "kind": {
                        "type": "string",
                        "description": "short label, e.g. fact, artifact, deadend",
                    },
                },
                "required": ["text"],
            },
            handler=publish_insight,
        )
    )
    registry.register(
        ToolSpec(
            name="check_hints",
            description=(
                "Read findings published by the other attempts since your last check. "
                "Returns nothing when there is no news."
            ),
            parameters={"type": "object", "properties": {}},
            handler=check_hints,
        )
    )


def _submit_tool(registry: ToolRegistry) -> None:
    def submit_flag(context: AgentContext, flag: str) -> dict:
        expected = context.challenge.flag
        if expected is None:
            return {
                "error": "this challenge has no flag metadata; cannot verify a submission"
            }
        if context.challenge.matches_flag(flag):
            context.state.solved = True
            context.state.flag = expected
            return {"status": "correct", "flag": expected}
        return {"status": "incorrect", "message": "flag rejected; keep investigating"}

    registry.register(
        ToolSpec(
            name="submit_flag",
            description="Submit the flag for this challenge. Ends the run if correct.",
            parameters={
                "type": "object",
                "properties": {
                    "flag": {"type": "string", "description": "the flag to submit"}
                },
                "required": ["flag"],
            },
            handler=submit_flag,
        )
    )


def _giveup_tool(registry: ToolRegistry) -> None:
    def giveup(context: AgentContext, confirm: bool = True) -> dict:
        if not confirm:
            return {"status": "aborted"}
        context.state.gave_up = True
        return {"status": "given_up"}

    registry.register(
        ToolSpec(
            name="giveup",
            description="Give up on this challenge and end the run immediately.",
            parameters={
                "type": "object",
                "properties": {
                    "confirm": {
                        "type": "boolean",
                        "description": "must be true to give up",
                    }
                },
                "required": ["confirm"],
            },
            handler=giveup,
        )
    )


def _finish_tool(registry: ToolRegistry) -> None:
    def finish_task(context: AgentContext, summary: str) -> dict:
        context.state.finished = True
        context.state.summary = str(summary).strip()
        return {"status": "finished"}

    registry.register(
        ToolSpec(
            name="finish_task",
            description=(
                "Finish the assigned subtask and return a compact summary to the caller: "
                "what you found, artifacts created, what is still unknown, next step."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "compact summary"}
                },
                "required": ["summary"],
            },
            handler=finish_task,
        )
    )


def _spawn_tool(registry: ToolRegistry) -> None:
    def spawn_subtask(
        context: AgentContext,
        task: str,
        max_rounds: int | None = None,
        max_cost: float | None = None,
    ) -> dict:
        if context.spawn_subtask is None:
            return {"error": "subtasks are not enabled for this run"}
        return context.spawn_subtask(task, max_rounds=max_rounds, max_cost=max_cost)

    registry.register(
        ToolSpec(
            name="spawn_subtask",
            description=(
                "Run a fresh-context helper agent on one focused subtask (deep analysis, "
                "exploit development, multi-step investigation). The helper shares the "
                "sandbox and notes board but not your conversation, and returns a "
                "compact summary. It is synchronous and costs budget; do not use it for "
                "single quick commands."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "precise, self-contained task description",
                    },
                    "max_rounds": {
                        "type": "integer",
                        "description": "optional round cap for the helper",
                    },
                    "max_cost": {
                        "type": "number",
                        "description": "optional USD cap for the helper",
                    },
                },
                "required": ["task"],
            },
            handler=spawn_subtask,
        )
    )


def build_main_tools(
    blackboard: Blackboard | None = None,
    *,
    hints: HintBus | None = None,
    enable_subtasks: bool = True,
) -> ToolRegistry:
    """Tool set for the main agent."""
    registry = ToolRegistry()
    _sandbox_tools(registry)
    _finding_tools(registry)
    if hints is not None:
        _hint_tools(registry)
    _submit_tool(registry)
    _giveup_tool(registry)
    if enable_subtasks:
        _spawn_tool(registry)
    return registry


def build_worker_tools(
    blackboard: Blackboard | None = None, *, hints: HintBus | None = None
) -> ToolRegistry:
    """Tool set for spawned subtask workers (no nested spawning)."""
    registry = ToolRegistry()
    _sandbox_tools(registry)
    _finding_tools(registry)
    if hints is not None:
        _hint_tools(registry)
    _submit_tool(registry)
    _finish_tool(registry)
    return registry
