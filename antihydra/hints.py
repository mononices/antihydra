"""In-process hint bus: lanes publish findings and read their peers'."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Hint:
    id: int
    source: str
    kind: str
    text: str
    time: float

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "kind": self.kind,
            "text": self.text,
        }


class HintBus:
    """Thread-safe, bounded bus shared by the lanes of one race."""

    def __init__(
        self, *, max_entries: int = 200, max_per_source: int = 25, max_text: int = 500
    ) -> None:
        self.max_entries = max_entries
        self.max_per_source = max_per_source
        self.max_text = max_text
        self._lock = threading.Lock()
        self._hints: list[Hint] = []
        self._next_id = 1
        self._reads = 0

    def publish(self, *, source: str, text: str, kind: str = "fact") -> Hint:
        clean = " ".join(str(text).split())[: self.max_text]
        if not clean:
            raise ValueError("hint text must not be empty")
        with self._lock:
            own = [hint for hint in self._hints if hint.source == source]
            if any(hint.text == clean for hint in own):
                raise ValueError("this hint was already published")
            if len(own) >= self.max_per_source:
                raise ValueError(
                    f"publish limit reached ({self.max_per_source}); share only durable, "
                    "cross-lane-useful findings"
                )
            hint = Hint(
                id=self._next_id,
                source=str(source),
                kind=str(kind)[:24],
                text=clean,
                time=round(time.time(), 2),
            )
            self._next_id += 1
            self._hints.append(hint)
            if len(self._hints) > self.max_entries:
                del self._hints[: len(self._hints) - self.max_entries]
            return hint

    def since(
        self, cursor: int = 0, *, exclude_source: str | None = None
    ) -> list[Hint]:
        with self._lock:
            hints = [hint for hint in self._hints if hint.id > int(cursor)]
        if exclude_source is not None:
            hints = [hint for hint in hints if hint.source != exclude_source]
        return hints

    def last_id(self) -> int:
        with self._lock:
            return self._next_id - 1

    def note_read(self) -> None:
        with self._lock:
            self._reads += 1

    def stats(self) -> dict:
        with self._lock:
            counts: dict[str, int] = {}
            for hint in self._hints:
                counts[hint.source] = counts.get(hint.source, 0) + 1
            return {"count": len(self._hints), "sources": counts, "reads": self._reads}
