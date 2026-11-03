"""The agent loop: a main agent plus optional fresh-context subtask workers."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .blackboard import Blackboard
from .challenge import Challenge
from .compaction import (
    CompactionConfig,
    archive_messages,
    build_prompt as build_compaction_prompt,
    estimate_tokens,
    safe_cut_index,
)
from .hints import HintBus
from .llm import LLMError, OpenRouterLLM, Reply
from .prompts import (
    KEEP_GOING,
    WORKER_KEEP_GOING,
    initial_prompt,
    system_prompt,
    worker_initial_prompt,
    worker_system_prompt,
)
from .sandbox import Sandbox
from .stalls import StallConfig, StallTracker, fingerprint, fingerprint_id
from .stalls import preview as stall_preview
from .tools import (
    AgentContext,
    CostLedger,
    RunState,
    ToolRegistry,
    build_worker_tools,
    clip,
)
from .transcript import Transcript


@dataclass
class RunResult:
    solved: bool
    flag: str | None
    reason: str
    rounds: int
    cost: float
    summary: str | None = None
    transcript_path: Path | None = None
    error: str | None = None
    stall_events: int = 0
    compactions: int = 0


class Agent:
    """Runs one conversation loop against one challenge in one sandbox."""

    def __init__(
        self,
        challenge: Challenge,
        sandbox: Sandbox,
        llm: OpenRouterLLM,
        tools: ToolRegistry,
        *,
        max_rounds: int = 200,
        max_cost: float = 3.0,
        timeout: float = 300.0,
        full_observations: int = 6,
        old_observation_limit: int = 2000,
        log_dir: str | Path = "logs",
        quiet: bool = False,
        emit: Callable[[str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        on_event: Callable[[str, dict], None] | None = None,
        poll_messages: Callable[[], list[str]] | None = None,
        state: RunState | None = None,
        ledger: CostLedger | None = None,
        blackboard: Blackboard | None = None,
        hints: HintBus | None = None,
        label: str = "main",
        bus_source: str | None = None,
        system_prompt: str | None = None,
        task: str | None = None,
        allow_subtasks: bool = True,
        max_subtasks: int = 4,
        subtask_rounds: int = 30,
        subtask_cost: float = 0.75,
        compaction: CompactionConfig | None = None,
        compaction_llm: OpenRouterLLM | None = None,
        stall: StallConfig | None = None,
    ) -> None:
        self.challenge = challenge
        self.sandbox = sandbox
        self.llm = llm
        self.tools = tools
        self.max_rounds = max_rounds
        self.max_cost = max_cost
        self.timeout = timeout
        self.full_observations = full_observations
        self.old_observation_limit = old_observation_limit
        self.log_dir = Path(log_dir)
        self.quiet = quiet
        self.emit = emit
        self.should_stop = should_stop
        self.on_event = on_event
        self.poll_messages = poll_messages
        self.state = state or RunState()
        self.ledger = ledger or CostLedger()
        self.blackboard = blackboard
        self.hints = hints
        self.label = label
        self.bus_source = bus_source or label
        self.custom_system_prompt = system_prompt
        self.custom_task = task
        self.allow_subtasks = allow_subtasks
        self.max_subtasks = max_subtasks
        self.subtask_rounds = subtask_rounds
        self.subtask_cost = subtask_cost
        self.compaction = compaction or CompactionConfig()
        self.compaction_llm = compaction_llm
        self.stall = stall or StallConfig()
        self._subtask_count = 0
        self._transcript: Transcript | None = None
        self._last_prompt_tokens = 0
        self._summary_present = False
        self._last_compaction_round = -10_000
        self._compactions = 0
        self._summaries: list[dict] = []
        self._compacted_archive: list[dict] = []
        self._tracker = StallTracker(self.stall.window)
        self._stall_events = 0
        self._stall_handled_round = -1

    def run(self) -> RunResult:
        state = self.state
        context = AgentContext(
            sandbox=self.sandbox,
            challenge=self.challenge,
            state=state,
            default_timeout=self.timeout,
            blackboard=self.blackboard,
            ledger=self.ledger,
            hints=self.hints,
            bus_source=self.bus_source,
            actor=self.label,
            spawn_subtask=self._spawn if self.allow_subtasks else None,
        )
        transcript = Transcript(
            challenge=self.challenge.name,
            category=self.challenge.category,
            model=self.llm.model,
            sandbox_image=self.sandbox.image,
        )
        transcript.data["agent"] = self.label
        self._transcript = transcript

        first_message = self.custom_system_prompt or system_prompt(
            self.challenge, self.timeout, coordinated=self.hints is not None
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": first_message},
            {
                "role": "user",
                "content": self.custom_task or initial_prompt(self.challenge),
            },
        ]
        if self.custom_task:
            transcript.event("task", task=self.custom_task)

        rounds = 0
        own_cost = 0.0
        reason = "unknown"
        error: str | None = None
        keep_going = WORKER_KEEP_GOING if self.custom_task else KEEP_GOING

        try:
            while not (
                state.solved or state.gave_up or state.stalled or state.finished
            ):
                if self.should_stop is not None and self.should_stop():
                    reason = "aborted"
                    break
                if self.poll_messages is not None:
                    for incoming in self.poll_messages():
                        messages.append(
                            {"role": "user", "content": f"[overseer] {incoming}"}
                        )
                        transcript.event("overseer", message=incoming)
                if rounds >= self.max_rounds:
                    reason = "max_rounds"
                    break
                if self.ledger.total >= self.max_cost:
                    reason = "budget"
                    break

                messages = self._maybe_compact(messages, rounds, transcript)
                reply = self.llm.chat(
                    self._prepare_messages(messages), self.tools.schemas()
                )
                rounds += 1
                own_cost += reply.cost
                self.ledger.add(reply.cost)
                self._last_prompt_tokens = int(reply.prompt_tokens or 0)
                transcript.event(
                    "assistant",
                    round=rounds,
                    content=reply.content,
                    tool_calls=[call.name for call in reply.tool_calls],
                    cost=round(reply.cost, 6),
                )
                self._emit_event(
                    "assistant",
                    {
                        "round": rounds,
                        "calls": [call.name for call in reply.tool_calls],
                    },
                )

                if reply.tool_calls:
                    messages.append(self._assistant_message(reply))
                    stall_message: str | None = None
                    for call in reply.tool_calls:
                        observation = self.tools.call(
                            call.name, call.arguments, context
                        )
                        self._say(
                            f"{self._prefix()}[{rounds:02d}] {call.name} {self._preview(call)}"
                        )
                        transcript.event(
                            "tool_call",
                            round=rounds,
                            name=call.name,
                            arguments=call.arguments,
                            errored='"error"' in observation,
                        )
                        self._emit_event(
                            "tool_call",
                            {"name": call.name, "arguments": call.arguments},
                        )
                        self._emit_event(
                            "tool_result",
                            {"name": call.name, "preview": observation[:400]},
                        )
                        if (
                            call.name == "submit_flag"
                            and '"status": "incorrect"' in observation
                        ):
                            self._emit_event("submit_failed", {"round": rounds})
                        self._check_flag(observation)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "content": observation,
                            }
                        )
                        if self.stall.enabled and stall_message is None:
                            call_key = fingerprint(call.name, call.arguments)
                            occurrences = self._tracker.add(call_key)
                            if (
                                occurrences >= self.stall.repeat
                                and self._stall_handled_round != rounds
                            ):
                                stall_message = self._handle_stall(
                                    call, call_key, occurrences, rounds, transcript
                                )
                    if stall_message is not None and not (
                        state.solved or state.stalled
                    ):
                        messages.append({"role": "user", "content": stall_message})
                else:
                    messages.append(
                        {"role": "assistant", "content": reply.content or ""}
                    )
                    if reply.content:
                        self._check_flag(reply.content)
                    if not (state.solved or state.finished):
                        messages.append({"role": "user", "content": keep_going})
            else:
                if state.solved:
                    reason = "solved"
                elif state.gave_up:
                    reason = "gave_up"
                elif state.stalled:
                    reason = "stalled"
                elif state.finished:
                    reason = "finished"
        except LLMError as exc:
            reason = "error"
            error = str(exc)
            transcript.event("error", message=str(exc))

        transcript.event(
            "finished",
            reason=reason,
            rounds=rounds,
            cost=round(own_cost, 6),
            solved=state.solved,
        )
        self._emit_event(
            "finished", {"reason": reason, "rounds": rounds, "cost": round(own_cost, 6)}
        )
        safe_name = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in (
                f"{self.challenge.category}_{self.challenge.name}_{self.label}_{self.llm.model}"
            )
        )
        transcript_path = transcript.save(
            self.log_dir,
            f"{time.strftime('%Y%m%d-%H%M%S')}__{safe_name[:90]}.json",
            messages=messages,
            notes=self.blackboard.read() if self.blackboard else "",
            summaries=self._summaries,
            compacted_messages=self._compacted_archive,
            total_cost=round(self.ledger.total, 6),
        )
        self._transcript = None
        return RunResult(
            solved=state.solved,
            flag=state.flag,
            reason=reason,
            rounds=rounds,
            cost=own_cost,
            summary=state.summary,
            transcript_path=transcript_path,
            error=error,
            stall_events=self._stall_events,
            compactions=self._compactions,
        )

    def _maybe_compact(self, messages, round_no, transcript) -> list[dict[str, Any]]:
        config = self.compaction
        if not config.enabled:
            return messages
        if round_no - self._last_compaction_round < config.min_interval:
            return messages
        limit = getattr(self.llm, "context_length", None)
        tokens = self._last_prompt_tokens or estimate_tokens(messages)
        triggered = (
            tokens >= config.at_fraction * limit
            if limit
            else tokens >= config.tokens_fallback
        )
        if not triggered:
            return messages
        return self._compact(messages, round_no, tokens, transcript)

    def _compact(
        self,
        messages: list[dict[str, Any]],
        round_no: int,
        tokens: int,
        transcript: Transcript,
    ) -> list[dict[str, Any]]:
        head = messages[:2]
        body = messages[2:]
        previous = None
        if self._summary_present and body:
            previous = str(body[0].get("content") or "")
            body = body[1:]
        cut = safe_cut_index(body, self.compaction.keep_recent)
        if cut is None or cut < self.compaction.min_dropped:
            return messages
        dropped, kept = body[:cut], body[cut:]
        llm = self.compaction_llm or self.llm
        self._last_compaction_round = round_no
        try:
            reply = llm.chat(
                [
                    {
                        "role": "user",
                        "content": build_compaction_prompt(
                            previous, dropped, self.compaction
                        ),
                    }
                ],
                temperature=0.0,
                max_tokens=self.compaction.max_output_tokens,
            )
        except LLMError as exc:
            transcript.event("compaction_error", round=round_no, message=str(exc))
            return messages
        self.ledger.add(reply.cost)
        summary = (reply.content or "").strip() or "(summary unavailable)"
        summary_message = {
            "role": "user",
            "content": "[Earlier context summary] Older turns were compressed; treat this as "
            "ground truth for facts, artifacts and dead ends.\n" + summary,
        }
        self._summary_present = True
        self._compactions += 1
        self._summaries.append(
            {
                "round": round_no,
                "dropped": len(dropped),
                "kept": len(kept),
                "tokens_before": tokens,
                "cost": round(reply.cost, 6),
                "summary": summary,
            }
        )
        self._compacted_archive.extend(archive_messages(dropped))
        transcript.event(
            "compaction",
            round=round_no,
            dropped=len(dropped),
            kept=len(kept),
            tokens_before=tokens,
            cost=round(reply.cost, 6),
            summary=summary,
        )
        self._say(
            f"{self._prefix()}compacted {len(dropped)} messages into a summary (${reply.cost:.4f})"
        )
        return head + [summary_message] + kept

    def _handle_stall(
        self,
        call: Any,
        call_key: str,
        occurrences: int,
        round_no: int,
        transcript: Transcript,
    ) -> str | None:
        self._stall_handled_round = round_no
        self._stall_events += 1
        call_preview = stall_preview(call.name, call.arguments)
        aborted = self._stall_events >= self.stall.abort_after
        transcript.event(
            "stall",
            round=round_no,
            call=call.name,
            fingerprint=fingerprint_id(call_key),
            occurrences=occurrences,
            action="abort" if aborted else "warn",
        )
        self._emit_event(
            "stall",
            {
                "round": round_no,
                "call": call.name,
                "occurrences": occurrences,
                "aborted": aborted,
            },
        )
        if self.blackboard is not None:
            self.blackboard.append(
                f"repeated identical call {occurrences}x: {call_preview}",
                kind="deadend",
                source=self.label,
            )
        if aborted:
            self.state.stalled = True
            self._say(
                f"{self._prefix()}stall detector: aborting after {self._stall_events} event(s)"
            )
            return None
        if self._stall_events >= 2:
            return (
                f"[stall detector] You are still repeating the same call ({occurrences}x: "
                f"{call_preview}). Change technique or tool, re-read the findings board, "
                "or call giveup."
            )
        return (
            f"[stall detector] The same call has now been made {occurrences} times with "
            f"identical arguments ({call_preview}). It will not produce a different result. "
            "Review the findings board and change approach."
        )

    def _spawn(
        self,
        task: str,
        *,
        max_rounds: int | None = None,
        max_cost: float | None = None,
    ) -> dict[str, Any]:
        """Run one fresh-context worker on a single subtask (synchronous)."""
        if not task or not str(task).strip():
            return {"error": "subtask task must not be empty"}
        if self._subtask_count >= self.max_subtasks:
            return {"error": f"subtask limit reached ({self.max_subtasks})"}
        remaining = self.ledger.remaining(self.max_cost)
        if remaining <= 0:
            return {"error": "budget exhausted; no subtasks left"}
        budget = min(float(max_cost) if max_cost else self.subtask_cost, remaining)

        self._subtask_count += 1
        label = f"subtask-{self._subtask_count}"
        rounds_cap = int(max_rounds) if max_rounds else self.subtask_rounds
        worker_state = RunState()
        worker = Agent(
            self.challenge,
            self.sandbox,
            self.llm,
            build_worker_tools(self.blackboard, hints=self.hints),
            max_rounds=rounds_cap,
            max_cost=budget,
            timeout=self.timeout,
            full_observations=self.full_observations,
            old_observation_limit=self.old_observation_limit,
            log_dir=self.log_dir,
            quiet=self.quiet,
            state=worker_state,
            ledger=self.ledger,
            blackboard=self.blackboard,
            hints=self.hints,
            bus_source=self.bus_source,
            label=label,
            system_prompt=worker_system_prompt(
                self.challenge, self.timeout, coordinated=self.hints is not None
            ),
            task=worker_initial_prompt(self.challenge, task),
            allow_subtasks=False,
            compaction=self.compaction,
            compaction_llm=self.compaction_llm,
            stall=self.stall,
            emit=self.emit,
            should_stop=self.should_stop,
        )
        self._say(f"{self._prefix()}-> {label} starts: {task.strip()[:100]}")
        result = worker.run()

        summary = worker_state.summary or f"(worker ended: {result.reason}, no summary)"
        if worker_state.solved:
            self.state.solved = True
            self.state.flag = worker_state.flag or self.challenge.flag
        if self.blackboard is not None:
            self.blackboard.append(summary, kind="subtask", source=label)
        if self.hints is not None and worker_state.summary:
            try:
                self.hints.publish(
                    source=self.bus_source,
                    kind="subtask",
                    text=f"{label}: {worker_state.summary}",
                )
            except ValueError:
                pass
        if self._transcript is not None:
            self._transcript.event(
                "subtask",
                label=label,
                task=str(task).strip(),
                status=result.reason,
                rounds=result.rounds,
                cost=round(result.cost, 6),
                summary=summary,
                transcript=(
                    str(result.transcript_path) if result.transcript_path else None
                ),
            )
        self._say(
            f"{self._prefix()}<- {label} done ({result.reason}), ${result.cost:.3f}"
        )
        return {
            "status": result.reason,
            "summary": clip(summary, 4000),
            "rounds": result.rounds,
            "cost": round(result.cost, 6),
            "transcript": (
                str(result.transcript_path) if result.transcript_path else None
            ),
            "solved": worker_state.solved,
        }

    def _check_flag(self, text: str | None) -> None:
        if not self.state.solved and self.challenge.matches_flag(text):
            self.state.solved = True
            self.state.flag = self.challenge.flag

    def _assistant_message(self, reply: Reply) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": reply.content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.raw_arguments
                        or json.dumps(call.arguments or {}),
                    },
                }
                for call in reply.tool_calls
            ],
        }

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Shorten older tool outputs; tool-call/result pairs are never dropped."""
        prepared: list[dict[str, Any]] = []
        tool_count = 0
        for message in reversed(messages):
            if message.get("role") == "tool":
                tool_count += 1
                if tool_count > self.full_observations:
                    content = message.get("content") or ""
                    if len(content) > self.old_observation_limit:
                        message = {
                            **message,
                            "content": content[: self.old_observation_limit]
                            + "\n... [earlier output trimmed]",
                        }
            prepared.append(message)
        prepared.reverse()
        return prepared

    def _preview(self, call: Any) -> str:
        if call.name == "run_command" and call.arguments:
            command = str(call.arguments.get("command", ""))
            return command.replace("\n", " ")[:110]
        if call.name in {"spawn_subtask", "finish_task"} and call.arguments:
            text = str(
                call.arguments.get("task") or call.arguments.get("summary") or ""
            )
            return text.replace("\n", " ")[:110]
        if call.arguments is not None:
            return json.dumps(call.arguments)[:110]
        return ""

    def _prefix(self) -> str:
        return "" if self.label == "main" else f"({self.label}) "

    def _emit_event(self, kind: str, payload: dict[str, Any] | None = None) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(kind, payload or {})
        except Exception:  # noqa: BLE001 - observers must not break the loop
            pass

    def _say(self, message: str) -> None:
        if self.quiet:
            return
        if self.emit is not None:
            self.emit(message)
        else:
            print(message)
