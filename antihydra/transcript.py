"""Run transcripts (JSON) for later analysis and leaderboard packaging."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class Transcript:
    """Collects timestamped events and writes a JSON record of a run."""

    def __init__(
        self, *, challenge: str, category: str, model: str, sandbox_image: str
    ) -> None:
        self.started = time.time()
        self.data: dict[str, Any] = {
            "challenge": challenge,
            "category": category,
            "model": model,
            "sandbox_image": sandbox_image,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "events": [],
        }

    def event(self, kind: str, **fields: Any) -> None:
        self.data["events"].append(
            {"t": round(time.time() - self.started, 2), "kind": kind, **fields}
        )

    def save(self, directory: str | Path, filename: str, **extra: Any) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        body = {
            **self.data,
            "minutes": round((time.time() - self.started) / 60, 2),
            **extra,
        }
        path.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
        return path
