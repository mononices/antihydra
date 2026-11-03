"""Parallel racing: lane objects driven by one coordinator."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 has no tomllib
    tomllib = None

from .agent import Agent, RunResult
from .blackboard import Blackboard
from .challenge import Challenge
from .compaction import CompactionConfig
from .hints import HintBus
from .llm import REASONING_EFFORTS, OpenRouterLLM
from .overseer import Advice, Overseer, build_digest, evaluate_triggers
from .sandbox import (
    Sandbox,
    cleanup_orphans,
    daemon_version,
    ensure_network,
    find_docker,
)
from .stalls import StallConfig
from .tools import CostLedger, build_main_tools

POLL_SECONDS = 0.5
HEARTBEAT_SECONDS = 30.0
ABORT_GRACE_SECONDS = 15.0
MEMORY_UNITS = {"k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}
FALLBACK_TEMPERATURE = 0.7
FALLBACK_REASONING_EFFORT = "max"
FALLBACK_MAX_COST = 4.0
FALLBACK_MAX_ROUNDS = 200
MIN_REVIEW_GAP_SECONDS = 5.0


@dataclass
class LaneSpec:
    id: int
    model: str
    temperature: float | None = None
    reasoning_effort: str = "max"
    max_cost: float | None = None
    max_rounds: int | None = None
    no_subtasks: bool = False

    @property
    def label(self) -> str:
        return f"lane-{self.id}"

    @property
    def short_model(self) -> str:
        return self.model.rsplit("/", 1)[-1]


@dataclass
class LaneResult:
    spec: LaneSpec
    status: str
    cost: float = 0.0
    rounds: int = 0
    transcript: str | None = None
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "id": self.spec.id,
            "model": self.spec.model,
            "temperature": self.spec.temperature,
            "status": self.status,
            "cost": round(self.cost, 6),
            "rounds": self.rounds,
            "transcript": self.transcript,
            "error": self.error,
        }


@dataclass
class LaneState:
    label: str
    model: str
    status: str = "running"
    rounds: int = 0
    cost: float = 0.0
    started_at: float = 0.0
    last_tool_at: float = 0.0
    remote_attempts: int = 0
    failed_submits: int = 0
    recent: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class LaneRunner(Protocol):
    """What the coordinator needs from a lane (real or fake in tests)."""

    spec: LaneSpec

    def run(self) -> LaneResult: ...

    def abort(self) -> None: ...

    def force_stop(self) -> None: ...

    def deliver(self, text: str) -> None: ...

    def state(self) -> LaneState: ...

    def notes(self) -> list[str]: ...

    def recent_slice(self, limit: int = 30) -> str: ...


def lane_label(race_id: str, spec: LaneSpec) -> str:
    return f"csaw26.lane.{race_id}.{spec.id}"


def load_lane_specs(
    models: Sequence[str] = (),
    lanes_file: str | Path | None = None,
    *,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    max_cost: float | None = None,
    max_rounds: int | None = None,
    no_subtasks: bool | None = None,
) -> list[LaneSpec]:
    """Build lane specs from a TOML/JSON roster or model ids.

    Precedence per field: lane entry > CLI value > file defaults > fallback.
    """
    defaults: dict = {}
    entries: list[dict] = []
    if lanes_file:
        defaults, entries = _read_lanes_file(lanes_file)
    else:
        entries = [{"model": model} for model in models]
    if not entries:
        raise ValueError(
            "no lanes given: pass --lanes, --lanes-file, or create lanes.toml"
        )

    specs: list[LaneSpec] = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("model"):
            raise ValueError("every lane entry needs a 'model'")
        specs.append(
            LaneSpec(
                id=len(specs) + 1,
                model=str(entry["model"]),
                temperature=_first(
                    entry.get("temperature"),
                    temperature,
                    defaults.get("temperature"),
                    FALLBACK_TEMPERATURE,
                ),
                reasoning_effort=_effort(
                    _first(
                        entry.get("reasoning_effort"),
                        reasoning_effort,
                        defaults.get("reasoning_effort"),
                        FALLBACK_REASONING_EFFORT,
                    )
                ),
                max_cost=_first(
                    entry.get("max_cost"),
                    max_cost,
                    defaults.get("max_cost"),
                    FALLBACK_MAX_COST,
                ),
                max_rounds=_first(
                    entry.get("max_rounds"),
                    max_rounds,
                    defaults.get("max_rounds"),
                    FALLBACK_MAX_ROUNDS,
                ),
                no_subtasks=bool(
                    _first(
                        entry.get("no_subtasks"),
                        no_subtasks,
                        defaults.get("no_subtasks"),
                        False,
                    )
                ),
            )
        )
    return specs


def _read_lanes_file(path: str | Path) -> tuple[dict, list[dict]]:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() == ".toml":
        if tomllib is None:
            raise ValueError("TOML rosters need Python 3.11+; use JSON instead")
        data = tomllib.loads(text)
    else:
        data = json.loads(text)
    if isinstance(data, list):
        return {}, data
    if not isinstance(data, dict):
        raise ValueError("roster must be a [[lanes]] table or a JSON list")
    defaults = data.get("defaults") or {}
    entries = data.get("lanes") or []
    if not isinstance(defaults, dict) or not isinstance(entries, list):
        raise ValueError("roster needs a [defaults] table and [[lanes]] entries")
    return defaults, entries


def _first(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _effort(value) -> str:
    if value is None:
        return FALLBACK_REASONING_EFFORT
    effort = str(value).strip().lower()
    if effort not in REASONING_EFFORTS:
        raise ValueError(
            f"invalid reasoning effort {value!r}; use one of: {', '.join(REASONING_EFFORTS)}"
        )
    return effort


class Lane:
    """One solver attempt with its own sandbox, board, ledger, and transcript."""

    def __init__(
        self,
        spec: LaneSpec,
        challenge: Challenge,
        *,
        llm: OpenRouterLLM,
        compaction_llm: OpenRouterLLM | None = None,
        docker: str | None = None,
        image: str = "ctfenv:csaw26",
        network: str = "ctfnet",
        memory: str = "4g",
        platform: str = "linux/amd64",
        timeout: float = 300.0,
        log_dir: str | Path,
        sandbox_label: str,
        emit: Callable[[str], None] | None = None,
        hints: HintBus | None = None,
        compaction: CompactionConfig | None = None,
        stall: StallConfig | None = None,
        max_subtasks: int = 4,
    ) -> None:
        self.spec = spec
        self.challenge = challenge
        self.llm = llm
        self.compaction_llm = compaction_llm
        self.docker = docker
        self.image = image
        self.network = network
        self.memory = memory
        self.platform = platform
        self.timeout = timeout
        self.log_dir = Path(log_dir)
        self.sandbox_label = sandbox_label
        self.emit = emit
        self.hints = hints
        self.compaction = compaction or CompactionConfig()
        self.stall = stall or StallConfig()
        self.max_subtasks = max_subtasks
        self.ledger = CostLedger()
        self.blackboard: Blackboard | None = None
        self.status = "running"
        self._stop = threading.Event()
        self._sandbox: Sandbox | None = None
        self._inbox: list[str] = []
        self._inbox_lock = threading.Lock()
        self._started_at = 0.0
        self._rounds = 0
        self._remote = 0
        self._failed_submits = 0
        self._last_tool_at = 0.0
        self._recent: list[str] = []
        self._trace: list[str] = []

    def run(self) -> LaneResult:
        ledger = self.ledger
        self.status = "running"
        self._started_at = time.time()
        sandbox = Sandbox(
            image=self.image,
            docker=self.docker,
            network=self.network,
            platform=self.platform,
            memory=self.memory,
            label=self.sandbox_label,
        )
        self._sandbox = sandbox
        try:
            sandbox.start()
            self.challenge.install_files(sandbox)
            blackboard = Blackboard(sandbox)
            self.blackboard = blackboard
            allow_subtasks = not self.spec.no_subtasks
            agent = Agent(
                self.challenge,
                sandbox,
                self.llm,
                build_main_tools(
                    blackboard, hints=self.hints, enable_subtasks=allow_subtasks
                ),
                max_rounds=self.spec.max_rounds or FALLBACK_MAX_ROUNDS,
                max_cost=self.spec.max_cost or FALLBACK_MAX_COST,
                timeout=self.timeout,
                log_dir=self.log_dir,
                emit=self.emit,
                should_stop=self._stop.is_set,
                on_event=self._handle_event,
                poll_messages=self._drain_inbox,
                ledger=ledger,
                blackboard=blackboard,
                hints=self.hints,
                bus_source=self.spec.label,
                compaction=self.compaction,
                compaction_llm=self.compaction_llm,
                stall=self.stall,
                allow_subtasks=allow_subtasks,
                max_subtasks=self.max_subtasks,
            )
            result = agent.run()
            status = _status_for(result)
            self.status = status
            if self.emit is not None:
                transcript = (
                    f" transcript={result.transcript_path}"
                    if result.transcript_path
                    else ""
                )
                self.emit(
                    f"finished: {status} reason={result.reason} rounds={result.rounds} "
                    f"cost=${ledger.total:.4f}{transcript}"
                )
            return LaneResult(
                spec=self.spec,
                status=status,
                cost=round(ledger.total, 6),
                rounds=result.rounds,
                transcript=(
                    str(result.transcript_path) if result.transcript_path else None
                ),
                error=result.error,
            )
        except Exception as exc:  # noqa: BLE001 - reported as a lane failure
            self.status = "error"
            return LaneResult(
                spec=self.spec, status="error", error=f"{type(exc).__name__}: {exc}"
            )
        finally:
            sandbox.stop()
            self._sandbox = None

    def abort(self) -> None:
        self._stop.set()

    def force_stop(self) -> None:
        self._stop.set()
        if self._sandbox is not None:
            self._sandbox.stop()

    def deliver(self, text: str) -> None:
        with self._inbox_lock:
            self._inbox.append(str(text))

    def _drain_inbox(self) -> list[str]:
        with self._inbox_lock:
            pending, self._inbox = self._inbox, []
        return pending

    def _handle_event(self, kind: str, payload: dict) -> None:
        if kind == "assistant":
            self._rounds = max(self._rounds, int(payload.get("round") or 0))
            return
        if kind == "tool_call":
            self._last_tool_at = time.time()
            name = str(payload.get("name") or "")
            arguments = " ".join(str(payload.get("arguments") or "").split())
            self._recent.append(f"{name}: {arguments[:120]}")
            del self._recent[:-20]
            self._trace.append(f"{name} {arguments[:200]}")
            del self._trace[:-80]
            host = self.challenge.server_host
            if (
                (host and host in arguments)
                or "remote(" in arguments
                or "connect(" in arguments
            ):
                self._remote += 1
            return
        if kind == "tool_result":
            self._trace.append(f"  -> {str(payload.get('preview') or '')[:300]}")
            del self._trace[:-80]
            return
        if kind == "submit_failed":
            self._failed_submits += 1

    def state(self) -> LaneState:
        return LaneState(
            label=self.spec.label,
            model=self.spec.model,
            status=self.status,
            rounds=self._rounds,
            cost=round(self.ledger.total, 6),
            started_at=self._started_at,
            last_tool_at=self._last_tool_at,
            remote_attempts=self._remote,
            failed_submits=self._failed_submits,
            recent=list(self._recent[-5:]),
        )

    def notes(self) -> list[str]:
        if self.blackboard is None:
            return []
        return [" ".join(line.split())[:160] for line in self.blackboard.lines()[-5:]]

    def recent_slice(self, limit: int = 40) -> str:
        return "\n".join(self._trace[-limit:])


def _status_for(result: RunResult) -> str:
    if result.solved:
        return "solved"
    if result.reason == "error":
        return "error"
    if result.reason == "aborted":
        return "aborted"
    return "unsolved"


class RaceCoordinator:
    """Runs lane objects in parallel; the first solved lane wins."""

    def __init__(
        self,
        challenge: Challenge,
        specs: Sequence[LaneSpec],
        *,
        race_dir: str | Path,
        lane_factory: Callable[
            [LaneSpec, Path, Callable[[str], None], HintBus], LaneRunner
        ],
        docker: str | None = None,
        network: str = "ctfnet",
        memory: str = "4g",
        start_service: bool = True,
        follow: bool = True,
        overseer: Overseer | None = None,
        max_minutes: float | None = None,
        poll_seconds: float = POLL_SECONDS,
        quiet: bool = False,
    ) -> None:
        self.challenge = challenge
        self.specs = list(specs)
        self.race_dir = Path(race_dir).resolve()
        self.lane_factory = lane_factory
        self.docker = docker
        self.network = network
        self.memory = memory
        self.start_service = start_service
        self.follow = follow
        self.max_minutes = max_minutes
        self.poll_seconds = poll_seconds
        self.quiet = quiet
        self.hints = HintBus()
        self.overseer = overseer
        self._overseer_log: list[dict] = []
        self._review_running = False
        self._review_thread: threading.Thread | None = None
        self._last_review = 0.0
        self._lock = threading.RLock()
        self._results: dict[int, LaneResult] = {}
        self._lanes: dict[int, LaneRunner] = {}
        self._logs: dict[int, object] = {}

    def _say(self, message: str) -> None:
        if self.quiet:
            return
        with self._lock:
            print(message, flush=True)

    def _lane_output(self, spec: LaneSpec, line: str) -> None:
        with self._lock:
            handle = self._logs.get(spec.id)
            if handle is not None and not handle.closed:
                try:
                    handle.write(line + "\n")
                    handle.flush()
                except (OSError, ValueError):
                    pass
            if self.follow:
                print(f"[{spec.label} {spec.short_model}] {line}", flush=True)

    def _run_lane(self, lane: LaneRunner) -> None:
        try:
            result = lane.run()
        except Exception as exc:  # noqa: BLE001 - reported as a lane failure
            result = LaneResult(
                spec=lane.spec, status="error", error=f"{type(exc).__name__}: {exc}"
            )
        with self._lock:
            self._results[lane.spec.id] = result
        self._say(
            f"[race] {lane.spec.label} ({lane.spec.short_model}) {result.status} "
            f"cost=${result.cost:.3f} rounds={result.rounds}"
        )

    def run(self) -> dict:
        self.race_dir.mkdir(parents=True, exist_ok=True)
        race_id = self.race_dir.name
        started = time.time()
        winner: LaneSpec | None = None
        timed_out = False
        service = None

        if self.start_service:
            if self.docker is None:
                self.docker = find_docker()
            daemon_version(self.docker)
            ensure_network(self.docker, self.network)
            _warn_memory(self.docker, self.memory, len(self.specs), self._say)
            if self.challenge.compose_file():
                self._say(
                    f"starting challenge service ({self.challenge.compose_file()})"
                )
                service = self.challenge.start_service(self.docker)

        threads: list[threading.Thread] = []
        interrupted = False
        try:
            for spec in self.specs:
                lane_dir = self.race_dir / spec.label
                lane_dir.mkdir(parents=True, exist_ok=True)
                (lane_dir / "lane.json").write_text(
                    json.dumps(
                        {
                            "model": spec.model,
                            "temperature": spec.temperature,
                            "reasoning_effort": spec.reasoning_effort,
                            "max_cost": spec.max_cost,
                            "max_rounds": spec.max_rounds,
                            "no_subtasks": spec.no_subtasks,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                with self._lock:
                    self._logs[spec.id] = open(
                        lane_dir / "run.log", "w", encoding="utf-8"
                    )
                emit = lambda line, lane_spec=spec: self._lane_output(lane_spec, line)
                try:
                    lane = self.lane_factory(spec, lane_dir, emit, self.hints)
                except Exception as exc:  # noqa: BLE001 - reported as a lane failure
                    with self._lock:
                        self._results[spec.id] = LaneResult(
                            spec=spec,
                            status="error",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    self._say(f"[race] {spec.label} failed to start: {exc}")
                    continue
                self._lanes[spec.id] = lane
                thread = threading.Thread(
                    target=self._run_lane, args=(lane,), daemon=True
                )
                threads.append(thread)
                thread.start()
                self._say(f"[race] started {spec.label} ({spec.short_model})")

            last_heartbeat = time.time()
            while True:
                time.sleep(self.poll_seconds)
                with self._lock:
                    finished = dict(self._results)
                if winner is None:
                    for lane_id, result in finished.items():
                        if result.status == "solved":
                            winner = next(
                                spec for spec in self.specs if spec.id == lane_id
                            )
                            break
                if winner is not None:
                    for lane in self._lanes.values():
                        lane.abort()
                    break
                if len(finished) >= len(self.specs):
                    break
                now = time.time()
                if now - last_heartbeat >= HEARTBEAT_SECONDS:
                    last_heartbeat = now
                    running = len(self.specs) - len(finished)
                    self._say(
                        f"[race] {running} lane(s) still running ({now - started:.0f}s elapsed)"
                    )
                if self.overseer is not None:
                    self._maybe_review(now, started)
                if self.max_minutes and now - started >= self.max_minutes * 60:
                    timed_out = True
                    for lane in self._lanes.values():
                        lane.abort()
                    break

        except KeyboardInterrupt:
            interrupted = True
            self._say("[race] interrupted; aborting lanes")
        finally:
            for lane in self._lanes.values():
                lane.abort()
            deadline = time.time() + ABORT_GRACE_SECONDS
            for thread in threads:
                thread.join(timeout=max(deadline - time.time(), 0.1))
            stragglers = [thread for thread in threads if thread.is_alive()]
            if stragglers:
                self._say(f"[race] force-stopping {len(stragglers)} straggler lane(s)")
                for lane in self._lanes.values():
                    lane.force_stop()
                for thread in stragglers:
                    thread.join(timeout=10)
            if self._review_thread is not None:
                self._review_thread.join(timeout=3)
            with self._lock:
                for handle in self._logs.values():
                    try:
                        handle.close()
                    except OSError:
                        pass
            if service is not None:
                service.stop()
            if self.docker:
                for spec in self.specs:
                    cleanup_orphans(self.docker, lane_label(race_id, spec))

        fallback = "aborted" if (interrupted or winner is not None) else "unknown"
        ordered = [
            self._results.get(spec.id) or LaneResult(spec=spec, status=fallback)
            for spec in self.specs
        ]
        total_cost = sum(result.cost for result in ordered)
        elapsed = time.time() - started
        overseer_report: dict = {"enabled": self.overseer is not None}
        if self.overseer is not None:
            overseer_report.update(
                {
                    "model": self.overseer.config.model,
                    "calls": self.overseer.calls,
                    "hints": len(self._overseer_log),
                    "cost": round(self.overseer.spent, 6),
                    "log": self._overseer_log,
                }
            )
        report = {
            "challenge": self.challenge.name,
            "category": self.challenge.category,
            "solved": winner is not None,
            "solved_by": winner.id if winner else None,
            "solved_by_model": winner.model if winner else None,
            "k": len(self.specs),
            "pass_at_k": 1 if winner else 0,
            "interrupted": interrupted,
            "hints": self.hints.stats(),
            "overseer": overseer_report,
            "total_cost": round(total_cost, 6),
            "elapsed_seconds": round(elapsed, 1),
            "started_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "lanes": [result.as_dict() for result in ordered],
        }
        summary_path = self.race_dir / "race-summary.json"
        summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        self._say("")
        if winner is not None:
            self._say(
                f"race: {self.challenge.name} solved by {winner.label} ({winner.short_model}) "
                f"in {elapsed / 60:.1f} min | cost ${total_cost:.3f} | "
                f"pass@k 1/{len(self.specs)}"
            )
        elif timed_out:
            self._say(
                f"race: {self.challenge.name} timed out after {self.max_minutes:.0f} min "
                f"| cost ${total_cost:.3f}"
            )
        elif interrupted:
            self._say(
                f"race: {self.challenge.name} interrupted | cost ${total_cost:.3f}"
            )
        else:
            self._say(
                f"race: {self.challenge.name} unsolved by {len(self.specs)} lane(s) "
                f"| cost ${total_cost:.3f}"
            )
        for result in ordered:
            self._say(
                f"  {result.spec.label} {result.spec.short_model:<28} {result.status:<9} "
                f"${result.cost:>7.3f} {result.rounds:>3} rounds  {result.transcript or ''}"
            )
        hint_stats = self.hints.stats()
        if hint_stats["count"]:
            self._say(
                f"hints: {hint_stats['count']} published ({hint_stats['sources']})"
            )
        if self.overseer is not None:
            self._say(
                f"overseer: {self.overseer.calls} call(s), {len(self._overseer_log)} hint(s), "
                f"${self.overseer.spent:.4f} ({self.overseer.config.model})"
            )
        self._say(f"summary: {summary_path}")
        return report

    def _lane_by_label(self, label: str) -> LaneRunner | None:
        for lane in self._lanes.values():
            if lane.spec.label == label:
                return lane
        return None

    def _maybe_review(self, now: float, started: float) -> None:
        overseer = self.overseer
        if overseer is None or self._review_running:
            return
        config = overseer.config
        if overseer.calls >= config.max_calls or overseer.spent >= config.max_cost:
            return
        states = [lane.state() for lane in self._lanes.values()]
        triggers = evaluate_triggers(config, states, self.hints.stats(), now)
        gap = max(config.min_gap_seconds, MIN_REVIEW_GAP_SECONDS)
        interval_due = now - self._last_review >= max(
            config.interval_seconds, MIN_REVIEW_GAP_SECONDS
        )
        trigger_due = bool(triggers) and now - self._last_review >= gap
        if not (interval_due or trigger_due):
            return
        self._last_review = now
        self._review_running = True
        self._review_thread = threading.Thread(
            target=self._review, args=(states, triggers, now, started), daemon=True
        )
        self._review_thread.start()

    def _review(
        self, states: list[LaneState], triggers: list[str], now: float, started: float
    ) -> None:
        overseer = self.overseer
        assert overseer is not None
        try:
            for state in states:
                lane = self._lane_by_label(state.label)
                if lane is not None:
                    state.notes = lane.notes()
            lane_spend = sum(lane.state().cost for lane in self._lanes.values())
            digest = build_digest(
                challenge=self.challenge,
                states=states,
                bus_stats=self.hints.stats(),
                triggers=triggers,
                elapsed_seconds=now - started,
                lane_spend=lane_spend,
                overseer_spend=overseer.spent,
                max_chars=overseer.config.max_digest_chars,
            )
            advice = overseer.review(digest)
            if advice is not None and advice.action == "inspect" and advice.target:
                target_lane = self._lane_by_label(advice.target)
                if target_lane is not None:
                    advice = overseer.review(
                        digest,
                        inspect_target=advice.target,
                        inspect_slice=target_lane.recent_slice(),
                    )
            if advice is not None and advice.action == "hint" and advice.text:
                self._deliver_hint(advice, now - started)
        except (
            Exception
        ) as exc:  # noqa: BLE001 - a broken review must not kill the race
            self._say(f"[overseer] review failed: {type(exc).__name__}: {exc}")
        finally:
            self._review_running = False

    def _deliver_hint(self, advice: Advice, elapsed: float) -> None:
        overseer = self.overseer
        assert overseer is not None
        target = advice.target or "*"
        if target != "*" and self._lane_by_label(target) is None:
            self._say(f"[overseer] hint for unknown lane {target!r}; broadcasting")
            target = "*"
        already = sum(1 for entry in self._overseer_log if entry["target"] == target)
        if target != "*" and already >= overseer.config.max_hints_per_lane:
            return
        text = str(advice.text)
        try:
            self.hints.publish(source="overseer", kind="hint", text=text)
        except ValueError:
            pass
        for lane in self._lanes.values():
            if target == "*" or lane.spec.label == target:
                lane.deliver(f"[overseer] {text}")
        self._overseer_log.append(
            {
                "at": round(elapsed, 1),
                "target": target,
                "text": text,
                "reason": advice.reason,
            }
        )
        self._say(f"[overseer] -> {target}: {text[:140]}")


def _parse_memory(value: str) -> int | None:
    text = value.strip().lower()
    if text.isdigit():
        return int(text)
    if text and text[-1] in MEMORY_UNITS:
        try:
            return int(float(text[:-1]) * MEMORY_UNITS[text[-1]])
        except ValueError:
            return None
    return None


def _warn_memory(
    docker: str, memory: str, lanes: int, say: Callable[[str], None]
) -> None:
    per_lane = _parse_memory(memory)
    if per_lane is None:
        return
    probe = subprocess.run(
        [docker, "info", "--format", "{{.MemTotal}}"], capture_output=True
    )
    try:
        total = int(probe.stdout.decode("utf-8", "replace").strip())
    except ValueError:
        return
    if total and lanes * per_lane > total * 0.9:
        say(
            f"[race] warning: {lanes} lanes x {memory} exceeds the Docker VM's "
            f"{total / 1024**3:.1f} GiB; lower --lane-memory or use fewer lanes"
        )
