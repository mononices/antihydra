"""Evaluation harness: run the racing solver across a challenge set."""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

from .challenge import Challenge, safe_dir_name, ChallengeError

COMPOSE_PORT_PATTERN = re.compile(r'^\s*-\s*"?(\d+):\d+"?\s*$')
CONTAINER_NAME_PATTERN = re.compile(r"^\s*container_name:\s*(\S+)\s*$")


@dataclass
class EvalConfig:
    lane_models: list[str] = field(default_factory=list)
    max_minutes_per_challenge: float | None = None
    max_total_cost: float | None = None
    skip_existing: bool = False
    concurrency: int = 3
    comment: str = ""
    # Recorded for audit: which dataset, which docker network and which flag
    # crib produced these numbers decides whether they can be compared.
    root: str = ""
    network: str = "ctfnet"
    flag_hint: str = "generic"


def run_dir_name(index: int, challenge: Challenge) -> str:
    """Per-challenge artifact directory: ``01__challenge-name`` (path-safe)."""
    return f"{index:02d}__{safe_dir_name(challenge.name)}"


def exclusive_resources(challenge: Challenge) -> set[str]:
    """Host ports and container names only one running challenge may hold."""
    compose = challenge.compose_file()
    if compose is None:
        return set()
    try:
        text = compose.read_text(encoding="utf-8")
    except OSError:
        return set()
    resources: set[str] = set()
    for line in text.splitlines():
        port = COMPOSE_PORT_PATTERN.match(line)
        if port:
            resources.add(f"port:{port.group(1)}")
        name = CONTAINER_NAME_PATTERN.match(line)
        if name:
            resources.add(f"container:{name.group(1)}")
    return resources


def prebuild_images(
    docker: str,
    challenges: Sequence[Challenge],
    eval_dir: str | Path,
    *,
    skip_existing: bool = False,
    say: Callable[[str], None] = print,
) -> dict[str, str]:
    """Build every challenge image up front; return failures by challenge directory.

    Doing this before the race turns a mid-run build failure into an immediate,
    visible error (and a ``build.log`` next to the race summary) instead of a
    challenge that silently scores zero.
    """
    root = Path(eval_dir)
    failures: dict[str, str] = {}
    total = len(challenges)
    for index, challenge in enumerate(challenges, start=1):
        if challenge.compose_file() is None:
            continue
        run_dir = root / run_dir_name(index, challenge)
        if skip_existing and (run_dir / "race-summary.json").is_file():
            say(f"[build {index}/{total}] {challenge.name}: skipped (existing run)")
            continue
        log_path = run_dir / "build.log"
        say(f"[build {index}/{total}] {challenge.name}: building ...")
        try:
            challenge.build_images(docker, log_path=log_path)
        except ChallengeError as exc:
            failures[str(challenge.directory)] = str(exc)
            say(f"[build {index}/{total}] {challenge.name}: FAILED -> {log_path}")
        else:
            say(f"[build {index}/{total}] {challenge.name}: ok")
    return failures


def run_eval(
    challenges: Sequence[Challenge],
    *,
    eval_dir: str | Path,
    coordinator_factory: Callable[[Challenge, Path], object],
    config: EvalConfig | None = None,
    quiet: bool = False,
    stop_event: threading.Event | None = None,
) -> dict:
    """Run one race per challenge, up to ``concurrency`` challenges at a time."""
    settings = config or EvalConfig()
    eval_dir = Path(eval_dir).resolve()
    eval_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    total = len(challenges)
    state_lock = threading.Lock()
    results: dict[int, dict] = {}
    active: dict[int, threading.Thread | None] = {}
    active_resources: set[str] = set()
    resource_cache: dict[int, set[str]] = {}
    stop = stop_event or threading.Event()
    halted = False

    def say(message: str) -> None:
        if not quiet:
            with state_lock:
                print(message, flush=True)

    def resources_for(index: int, challenge: Challenge) -> set[str]:
        if index not in resource_cache:
            resource_cache[index] = exclusive_resources(challenge)
        return resource_cache[index]

    def finish(index: int, challenge: Challenge, report: dict) -> None:
        run_dir = eval_dir / run_dir_name(index, challenge)
        flag = report.get("flag")
        if flag:
            (run_dir / "flag.txt").write_text(str(flag) + "\n", encoding="utf-8")
        record = {
            "challenge": challenge.name,
            "category": challenge.category,
            "solved": bool(report.get("solved")),
            "solved_by": report.get("solved_by_model"),
            "flag": flag,
            "target": report.get("target"),
            "targets": report.get("targets") or [],
            "cost": report.get("total_cost"),
            "overseer_cost": (report.get("overseer") or {}).get("cost"),
            "elapsed_seconds": report.get("elapsed_seconds"),
            "k": report.get("k"),
            "race_dir": str(run_dir),
            "race_summary": str(run_dir / "race-summary.json"),
            "lanes": report.get("lanes") or [],
            "error": report.get("error"),
        }
        with state_lock:
            results[index] = record
            active_resources.difference_update(resources_for(index, challenge))
            active.pop(index, None)
        cost = record["cost"] or 0.0
        elapsed = record["elapsed_seconds"] or 0.0
        outcome = f"solved by {record['solved_by']}" if record["solved"] else "unsolved"
        say(
            f"[{index}/{total}] {challenge.name}: {outcome} "
            f"(${cost:.3f}, {elapsed:.0f}s)" + (" -> flag.txt" if flag else "")
        )

    def run_challenge(index: int, challenge: Challenge) -> None:
        run_dir = eval_dir / run_dir_name(index, challenge)
        summary_path = run_dir / "race-summary.json"
        try:
            if settings.skip_existing and summary_path.is_file():
                report = json.loads(summary_path.read_text(encoding="utf-8"))
                say(f"[{index}/{total}] {challenge.name}: skipped (existing run)")
            else:
                say(f"[{index}/{total}] {challenge.name}: racing ...")
                try:
                    report = coordinator_factory(challenge, run_dir).run()
                except Exception as exc:  # noqa: BLE001 - keep the eval alive
                    report = {
                        "challenge": challenge.name,
                        "category": challenge.category,
                        "solved": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    say(f"[{index}/{total}] {challenge.name}: error: {exc}")
                if not summary_path.is_file():
                    summary_path.parent.mkdir(parents=True, exist_ok=True)
                    summary_path.write_text(
                        json.dumps(report, indent=2), encoding="utf-8"
                    )
            finish(index, challenge, report)
        except Exception as exc:  # noqa: BLE001 - never leave a slot occupied
            # A challenge must never vanish from the leaderboard: record the
            # failure even when it happened after the race itself finished.
            with state_lock:
                active_resources.difference_update(resources_for(index, challenge))
                active.pop(index, None)
                results[index] = {
                    "challenge": challenge.name,
                    "category": challenge.category,
                    "solved": False,
                    "solved_by": None,
                    "flag": None,
                    "cost": None,
                    "overseer_cost": None,
                    "elapsed_seconds": None,
                    "k": None,
                    "race_dir": str(run_dir),
                    "race_summary": str(summary_path),
                    "lanes": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            say(f"[{index}/{total}] {challenge.name}: unexpected error: {exc}")

    pending: list[tuple[int, Challenge]] = list(enumerate(challenges, start=1))
    try:
        while pending or active:
            if stop.is_set() and not halted:
                halted = True
                say("stopping: no new challenges will start")
            launched = False
            while not halted and len(active) < settings.concurrency and pending:
                if settings.max_total_cost is not None:
                    with state_lock:
                        spent = sum(item["cost"] or 0.0 for item in results.values())
                    if spent >= settings.max_total_cost:
                        halted = True
                        say(
                            f"total cost cap ${settings.max_total_cost:.2f} reached; "
                            "stopping"
                        )
                        break
                chosen: tuple[int, int, Challenge] | None = None
                with state_lock:
                    for position, (index, challenge) in enumerate(pending):
                        if resources_for(index, challenge) & active_resources:
                            continue
                        chosen = (position, index, challenge)
                        active_resources |= resources_for(index, challenge)
                        active[index] = None
                        break
                if chosen is None:
                    break
                position, index, challenge = chosen
                del pending[position]
                thread = threading.Thread(
                    target=run_challenge, args=(index, challenge), daemon=True
                )
                with state_lock:
                    active[index] = thread
                thread.start()
                launched = True
            if not launched and (pending or active):
                time.sleep(0.2)
    except KeyboardInterrupt:
        stop.set()
        halted = True
        say("interrupted; waiting for running challenges to stop")
        while active:
            time.sleep(0.2)

    for thread in list(active.values()):
        if thread is not None:
            thread.join()

    ordered = [results[index] for index in sorted(results)]
    solved = [record for record in ordered if record["solved"]]
    lane_total = sum(len(record["lanes"]) for record in ordered)
    lane_solved = sum(
        1
        for record in ordered
        for lane in record["lanes"]
        if lane.get("status") == "solved"
    )
    by_category: dict[str, dict] = {}
    for record in ordered:
        bucket = by_category.setdefault(record["category"], {"total": 0, "solved": 0})
        bucket["total"] += 1
        bucket["solved"] += 1 if record["solved"] else 0
    k = max((record["k"] or 0 for record in ordered), default=0)
    total_cost = round(
        sum(
            (record["cost"] or 0.0) + (record["overseer_cost"] or 0.0)
            for record in ordered
        ),
        6,
    )
    elapsed_seconds = round(time.time() - started, 1)
    summary = {
        "started_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "challenges": len(ordered),
        "planned": total,
        "solved": len(solved),
        "pass_at_k": round(len(solved) / len(ordered), 4) if ordered else 0.0,
        "k": k,
        "lane_solve_rate": round(lane_solved / lane_total, 4) if lane_total else 0.0,
        "by_category": by_category,
        "total_cost": total_cost,
        "elapsed_seconds": elapsed_seconds,
        "config": {
            "lane_models": settings.lane_models,
            "max_minutes_per_challenge": settings.max_minutes_per_challenge,
            "max_total_cost": settings.max_total_cost,
            "concurrency": settings.concurrency,
            "skip_existing": settings.skip_existing,
            "root": settings.root,
            "network": settings.network,
            "flag_hint": settings.flag_hint,
        },
        "results": ordered,
    }
    (eval_dir / "eval-summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    leaderboard = {
        "metadata": {
            "agent": "antihydra-racing",
            "comment": settings.comment or f"pass@{k}",
            "model": ", ".join(settings.lane_models),
            "link": "",
            "date": datetime.now().strftime("%Y/%m/%d"),
        },
        "results": {record["challenge"]: record["solved"] for record in ordered},
    }
    (eval_dir / "summary.json").write_text(
        json.dumps(leaderboard, indent=2), encoding="utf-8"
    )

    with (eval_dir / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for record in ordered:
            for lane in record["lanes"]:
                handle.write(
                    json.dumps(
                        {
                            "challenge": record["challenge"],
                            "category": record["category"],
                            "lane": lane.get("id"),
                            "model": lane.get("model"),
                            "status": lane.get("status"),
                            "cost": lane.get("cost"),
                            "rounds": lane.get("rounds"),
                            "flag": lane.get("flag"),
                            "transcript": lane.get("transcript"),
                        }
                    )
                    + "\n"
                )

    say("")
    say(
        f"eval: {len(solved)}/{len(ordered)} challenges solved (k={k}, "
        f"concurrency={settings.concurrency}) | ${total_cost:.2f} "
        f"| {elapsed_seconds / 60:.1f} min"
    )
    for category, bucket in sorted(by_category.items()):
        say(f"  {category:<10} {bucket['solved']}/{bucket['total']}")
    say(f"summary:     {eval_dir / 'eval-summary.json'}")
    say(f"leaderboard: {eval_dir / 'summary.json'}")
    say(f"traces:      {eval_dir / 'traces.jsonl'}")
    return summary
