"""Tool registry, cost ledger, and the main/worker tool profiles."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .blackboard import Blackboard
from .challenge import Challenge
from .hints import HintBus
from .images import (
    ATTACHMENT_DIR,
    DEFAULT_MAX_DIMENSION,
    MAX_IMAGE_BYTES,
    MAX_IMAGES_PER_AGENT,
    MAX_READ_BYTES,
    SUPPORTED_MEDIA_TYPES,
    AttachedImage,
    encode,
    extension_for,
    parse_crop,
    prepare_script,
    slug,
    sniff_media_type,
)
from .sandbox import Sandbox, SandboxError

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
    deadline: float | None = None  # time.monotonic() wall-clock budget for the run
    max_command_timeout: float | None = None  # hard ceiling for one run_command
    pending_images: list[AttachedImage] = field(default_factory=list)
    images_used: int = 0
    max_images: int = MAX_IMAGES_PER_AGENT
    vision_supported: bool | None = None


def clamp_command_timeout(
    requested: float, context: AgentContext, *, minimum: float = 5.0
) -> tuple[float, str | None]:
    """Bound one command by the run deadline and the per-command ceiling."""
    limits = [float(requested)]
    notes: list[str] = []
    if context.max_command_timeout is not None:
        limits.append(float(context.max_command_timeout))
    if context.deadline is not None:
        remaining = context.deadline - time.monotonic()
        if remaining <= minimum:
            return 0.0, (
                "the run is out of time; stop starting long commands and finish or "
                "submit what you have"
            )
        limits.append(remaining - 1.0)
    effective = max(min(limits), 0.0)
    if effective + 0.5 < float(requested):
        notes.append(
            f"timeout clamped from {float(requested):.0f}s to {effective:.0f}s "
            "(run budget / per-command ceiling); split the work into smaller steps "
            "or hand the long job to a spawn_subtask worker"
        )
    return effective, "; ".join(notes) or None


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
        requested = float(timeout) if timeout else context.default_timeout
        effective, note = clamp_command_timeout(requested, context)
        if effective <= 0:
            return _error(note or "no time left in the run for more commands")
        payload = context.sandbox.exec(command, timeout=effective).to_dict()
        payload["stdout"] = clip(payload["stdout"])
        payload["stderr"] = clip(payload["stderr"])
        payload["timeout_seconds"] = round(effective, 1)
        if note:
            payload["timeout_note"] = note
        if payload.get("timed_out"):
            payload["hint"] = (
                "command timed out; do not rerun it unchanged. Split it into smaller "
                "steps that write intermediate results to files, or hand the long job "
                "to a spawn_subtask worker that can report a summary."
            )
        elif payload.get("killed") == "output_flood":
            payload["hint"] = (
                "killed early: the command kept printing the same output "
                f"({payload.get('killed_note')}). That usually means a program is "
                "waiting for input that never came - feed it stdin (a here-doc, "
                "`yes 5 |`, or redirect a file), or drive it from a script with "
                "short read timeouts (pwntools: context.timeout, recvuntil timeout=)."
            )
        return payload

    def create_file(context: AgentContext, path: str, content: str) -> dict:
        context.sandbox.write_text(path, content)
        return {"status": "written", "path": path, "bytes": len(content)}

    def read_file(context: AgentContext, path: str) -> dict:
        return {
            "path": path,
            "content": clip(context.sandbox.read_text(path, limit=TOOL_OUTPUT_LIMIT)),
        }

    def view_image(
        context: AgentContext,
        path: str,
        question: str | None = None,
        detail: str = "auto",
        crop: str | None = None,
        max_dimension: float | None = None,
    ) -> dict:
        if context.vision_supported is False:
            return _error(
                "this model does not accept image input; switch to a multimodal model "
                "or fall back to exiftool/strings/OCR extraction"
            )
        if context.images_used >= context.max_images:
            return _error(
                f"image limit reached ({context.max_images} per run); describe the "
                "remaining evidence in text instead"
            )
        detail_value = str(detail or "auto").strip().lower()
        if detail_value not in ("auto", "low", "high"):
            return _error("detail must be one of: auto, low, high")
        try:
            crop_spec = parse_crop(crop or "")
        except ValueError as exc:
            return _error(str(exc))

        original = path
        size = context.sandbox.file_size(original)
        if size < 0:
            return _error(f"no such file inside the sandbox: {original}")
        dimension = int(max_dimension) if max_dimension else DEFAULT_MAX_DIMENSION
        dimension = max(min(dimension, 2048), 64)

        raw: bytes | None = None
        media_type: str | None = None
        rendered: str | None = None
        if crop_spec is None and size <= MAX_IMAGE_BYTES:
            candidate = context.sandbox.read_bytes(original, limit=MAX_IMAGE_BYTES)
            media_type = sniff_media_type(candidate)
            if media_type in SUPPORTED_MEDIA_TYPES:
                raw = candidate
        if raw is None:
            if size > MAX_READ_BYTES:
                return _error(
                    f"{original} is {size} bytes, over the {MAX_READ_BYTES} byte read "
                    "limit; extract or crop a smaller file first"
                )
            for fmt in ("png", "jpeg"):
                rendered = (
                    f"{ATTACHMENT_DIR}/{slug(original)}"
                    f"{'-' + slug(crop_spec) if crop_spec else ''}{extension_for('image/' + fmt)}"
                )
                command = prepare_script(
                    original,
                    rendered,
                    fmt=fmt,
                    crop=crop_spec,
                    max_dimension=dimension,
                    upscale=crop_spec is not None or size < 64_000,
                )
                result = context.sandbox.exec(command, timeout=120)
                if result.exit_code != 0:
                    return _error(
                        f"failed to render {original}: {result.stderr.strip()[:300]}"
                    )
                try:
                    candidate = context.sandbox.read_bytes(
                        rendered, limit=MAX_IMAGE_BYTES
                    )
                except SandboxError:
                    continue  # still too large at this quality; try the next format
                media_type = sniff_media_type(candidate)
                if media_type in SUPPORTED_MEDIA_TYPES:
                    raw = candidate
                    break
        if raw is None or media_type is None:
            return _error(
                f"could not produce a supported image under {MAX_IMAGE_BYTES} bytes "
                "from " + original + "; try a smaller crop or max_dimension"
            )

        image = AttachedImage(
            path=original,
            media_type=media_type,
            payload=encode(raw),
            detail=detail_value,
            question=str(question or "").strip(),
        )
        context.pending_images.append(image)
        context.images_used += 1
        return {
            "status": "attached",
            "path": original,
            "rendered": rendered or original,
            "media_type": media_type,
            "bytes": image.bytes,
            "detail": detail_value,
            "pending": len(context.pending_images),
            "message": (
                "the image is attached to your next message; look at it before "
                "running more pixel-level commands"
            ),
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
                        "description": (
                            "optional timeout in seconds; clamped to the remaining run "
                            "budget and the per-command ceiling"
                        ),
                    },
                },
                "required": ["command"],
            },
            handler=run_command,
        )
    )
    registry.register(
        ToolSpec(
            name="view_image",
            description=(
                "Render an image from the sandbox into your context so you can look at "
                "it (screenshots, video frames, stego output, diagrams, PDF pages). "
                "Prefer this over pixel statistics, hex-dumping or OCR guessing. "
                "Use crop=\"x,y,w,h\" to zoom into a region; large images are "
                "downscaled, small crops are enlarged."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "absolute path of the image inside the sandbox",
                    },
                    "question": {
                        "type": "string",
                        "description": "what to look for, e.g. 'read the text in this frame'",
                    },
                    "detail": {
                        "type": "string",
                        "description": "vision detail level: auto, low or high",
                    },
                    "crop": {
                        "type": "string",
                        "description": "optional region x,y,w,h in pixels",
                    },
                    "max_dimension": {
                        "type": "number",
                        "description": "optional longest-side cap before sending (default 1200)",
                    },
                },
                "required": ["path"],
            },
            handler=view_image,
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
