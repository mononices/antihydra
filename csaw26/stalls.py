"""Detect repeated identical tool calls and nudge or stop the agent."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class StallConfig:
    enabled: bool = True
    window: int = 30  # tool calls considered for duplicates
    repeat: int = 3  # identical occurrences that count as a stall
    abort_after: int = 3  # stall events before the run is stopped


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def fingerprint(name: str, arguments: Any) -> str:
    """Canonical identity of a tool call (whitespace-insensitive for strings)."""
    canonical = json.dumps(
        _normalize(arguments or {}), sort_keys=True, ensure_ascii=False
    )
    return f"{name}:{canonical}"


def fingerprint_id(fingerprint_text: str) -> str:
    """Short stable id for transcripts."""
    return hashlib.sha1(fingerprint_text.encode("utf-8")).hexdigest()[:12]


def preview(name: str, arguments: Any, limit: int = 140) -> str:
    """Short human-readable rendering of a call for nudges and findings."""
    text = json.dumps(arguments, ensure_ascii=False) if arguments else ""
    return " ".join(f"{name} {text}".split())[:limit]


class StallTracker:
    """Rolling window of tool-call fingerprints with occurrence counting."""

    def __init__(self, window: int = 30) -> None:
        self._calls: deque[str] = deque(maxlen=max(int(window), 1))

    def add(self, fingerprint_text: str) -> int:
        """Record one call and return its occurrence count within the window."""
        self._calls.append(fingerprint_text)
        return self.count(fingerprint_text)

    def count(self, fingerprint_text: str) -> int:
        return sum(1 for item in self._calls if item == fingerprint_text)

    def clear(self) -> None:
        self._calls.clear()
