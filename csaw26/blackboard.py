"""Markdown findings board shared by the agents on one challenge."""

from __future__ import annotations

from .sandbox import Sandbox, SandboxError

DEFAULT_PATH = "/home/ctfplayer/notes.md"
HEADER = "# Findings\n"


class Blackboard:
    """One markdown notes board backed by a file inside a sandbox."""

    def __init__(self, sandbox: Sandbox, path: str = DEFAULT_PATH) -> None:
        self.sandbox = sandbox
        self.path = path

    def read(self) -> str:
        """Full board contents (empty string if it does not exist yet)."""
        try:
            return self.sandbox.read_text(self.path)
        except SandboxError:
            return ""

    def append(self, text: str, *, kind: str = "fact", source: str = "main") -> str:
        """Append one finding bullet and return the line that was written."""
        text = " ".join(str(text).split())
        kind = (str(kind).strip()[:24] or "fact") if kind else "fact"
        line = f"- [{kind}] ({source}) {text}"
        body = self.read()
        if not body.strip():
            body = HEADER + "\n"
        if not body.endswith("\n"):
            body += "\n"
        self.sandbox.write_text(self.path, body + line + "\n")
        return line

    def lines(self) -> list[str]:
        """All finding bullets, in order."""
        return [line for line in self.read().splitlines() if line.startswith("- ")]

    def search(self, query: str | None = None) -> list[str]:
        """Bullets containing ``query`` (case-insensitive); all bullets if empty."""
        lines = self.lines()
        if not query:
            return lines
        needle = str(query).lower()
        return [line for line in lines if needle in line.lower()]
