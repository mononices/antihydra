#!/usr/bin/env python3
"""Print a readable report for a race directory (default: newest under logs/races)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PATTERNS = (
    "Commander",
    "Setting up qemu",
    "no-qemu",
    "exec format error",
    "refused",
    "csawctf{",
)


def newest_race(root: Path) -> Path:
    races = sorted(
        (path for path in root.glob("*/") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    if not races:
        raise SystemExit(f"no races under {root}")
    return races[-1]


def scan(messages: list[dict], needle: str) -> tuple[int, str]:
    hits, sample = 0, ""
    for message in messages:
        if message.get("role") != "tool":
            continue
        content = str(message.get("content") or "")
        if needle.lower() in content.lower():
            hits += 1
            if not sample:
                index = content.lower().index(needle.lower())
                sample = " ".join(content[max(0, index - 60) : index + 140].split())
    return hits, sample


def report_lane(lane_dir: Path) -> None:
    print("=" * 88)
    print(lane_dir.name)
    transcript_path = next(iter(sorted(lane_dir.glob("*_main_*.json"))), None)
    if transcript_path is None:
        print("  (no main transcript)")
        return
    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    events = data.get("events") or []
    finished = next(
        (event for event in reversed(events) if event.get("kind") == "finished"), {}
    )
    print(
        f"  finished: {finished.get('reason')} rounds={finished.get('rounds')} "
        f"cost=${finished.get('cost')}"
    )
    calls = [event for event in events if event.get("kind") == "tool_call"]
    tools = ", ".join(
        f"{name}:{count}"
        for name, count in Counter(c.get("name") for c in calls).most_common()
    )
    print("  tools:", tools)
    remote = sum(
        1
        for call in calls
        if "remote(" in str((call.get("arguments") or {}).get("command", ""))
    )
    stalls = sum(1 for event in events if event.get("kind") == "stall")
    print(f"  remote() attempts: {remote} | stalls: {stalls}")
    messages = data.get("messages") or []
    for pattern in PATTERNS:
        hits, sample = scan(messages, pattern)
        if hits:
            print(f"  {pattern!r}: {hits}x | {sample[:150]}")
    thoughts = [
        message["content"]
        for message in messages
        if message.get("role") == "assistant" and (message.get("content") or "").strip()
    ]
    print("  last thoughts:")
    for thought in thoughts[-3:]:
        print("   *", " ".join(thought.split())[:300])


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "race",
        nargs="?",
        default=None,
        help="a race directory, or a directory of races (default: logs/races)",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv[1:])
    base = Path(args.race) if args.race else Path("logs/races")
    if not base.is_dir():
        print(f"error: no such race directory: {base}", file=sys.stderr)
        return 2
    race = base if (base / "race-summary.json").is_file() else newest_race(base)
    summary_path = race / "race-summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(
            f"race: {race.name} | solved={summary.get('solved')} "
            f"| interrupted={summary.get('interrupted')} | cost=${summary.get('total_cost')} "
            f"| {summary.get('elapsed_seconds')}s"
        )
        hints = summary.get("hints") or {}
        if hints.get("count"):
            print(f"  hints published: {hints['count']} {hints['sources']}")
    for lane_dir in sorted(race.glob("lane-*")):
        report_lane(lane_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
