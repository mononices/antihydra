"""Challenge discovery, metadata, and challenge-service lifecycle."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .sandbox import Sandbox

COMPOSE_NAMES = (
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yml",
    "compose.yaml",
)


class ChallengeError(RuntimeError):
    """Raised when challenge data is missing or malformed."""


@dataclass(frozen=True)
class Challenge:
    """One challenge loaded from ``challenge.json``."""

    directory: Path
    metadata: dict[str, Any] = field(repr=False)

    @property
    def name(self) -> str:
        return str(self.metadata.get("name") or self.directory.name)

    @property
    def category(self) -> str:
        value = self.metadata.get("category")
        return str(value) if value else self.directory.parent.name

    @property
    def description(self) -> str:
        return str(self.metadata.get("description") or "No description provided.")

    @property
    def flag(self) -> str | None:
        value = self.metadata.get("flag")
        return str(value) if value else None

    @property
    def files(self) -> tuple[str, ...]:
        raw = self.metadata.get("files") or []
        return tuple(str(item) for item in raw)

    @property
    def server_host(self) -> str | None:
        value = self.metadata.get("box")
        return str(value) if value else None

    @property
    def server_port(self) -> int | None:
        value = self.metadata.get("internal_port")
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def description_text(self) -> str:
        text = self.description
        if self.server_host:
            text = text.replace("{box}", self.server_host)
        if self.server_port:
            text = text.replace("{port}", str(self.server_port))
        return text

    def flag_format_hint(self) -> str | None:
        flag = self.flag
        if not flag or "{" not in flag or "}" not in flag:
            return None
        prefix = flag[: flag.index("{")]
        return f"{prefix}{{...}}"

    def target_line(self) -> str | None:
        if not self.server_host or not self.server_port:
            return None
        return f"{self.server_host}:{self.server_port}"

    def local_files(self) -> list[Path]:
        found: list[Path] = []
        for entry in self.files:
            candidate = (self.directory / entry).resolve()
            if candidate.exists():
                found.append(candidate)
        return found

    def missing_files(self) -> list[str]:
        return [entry for entry in self.files if not (self.directory / entry).exists()]

    def install_files(self, sandbox: "Sandbox") -> None:
        """Copy the challenge's files into the sandbox."""
        for entry in self.missing_files():
            print(
                f"[warn] challenge file listed but missing on disk: {entry}",
                file=sys.stderr,
            )
        for path in self.local_files():
            relative = path.relative_to(self.directory).as_posix()
            sandbox.copy_in(path, f"/home/ctfplayer/ctf_files/{relative}")

    def matches_flag(self, text: str | None) -> bool:
        """True when ``text`` contains the flag (full or inner secret)."""
        flag = self.flag
        if not flag or not text:
            return False
        if flag in text:
            return True
        if "{" in flag and "}" in flag:
            inner = flag[flag.index("{") + 1 : flag.rindex("}")]
            if len(inner) >= 8 and inner in text:
                return True
        return False

    def compose_file(self) -> Path | None:
        for base in (self.directory, self.directory / "infra"):
            for name in COMPOSE_NAMES:
                candidate = base / name
                if candidate.is_file():
                    return candidate
        return None

    def start_service(
        self, docker: str, *, wait: float = 0.0
    ) -> "ChallengeService | None":
        """Start the challenge's docker-compose stack, if it has one."""
        compose = self.compose_file()
        if compose is None:
            return None
        command = _compose_command(docker) + [
            "-f",
            str(compose),
            "up",
            "-d",
            "--remove-orphans",
        ]
        proc = subprocess.run(command, cwd=str(compose.parent), capture_output=True)
        if proc.returncode != 0:
            raise ChallengeError(
                "failed to start challenge service:\n"
                + proc.stderr.decode("utf-8", "replace").strip()
            )
        if wait > 0:
            time.sleep(wait)
        return ChallengeService(compose_file=compose, docker=docker)


def _compose_command(docker: str) -> list[str]:
    """Prefer the compose v2 plugin, fall back to legacy docker-compose."""
    probe = subprocess.run([docker, "compose", "version"], capture_output=True)
    if probe.returncode == 0:
        return [docker, "compose"]
    legacy = shutil.which("docker-compose")
    if legacy:
        return [legacy]
    raise ChallengeError("neither 'docker compose' nor 'docker-compose' is available")


@dataclass
class ChallengeService:
    """A started ``docker compose`` project for one challenge."""

    compose_file: Path
    docker: str

    def logs(self, tail: int = 200) -> str:
        command = _compose_command(self.docker) + [
            "-f",
            str(self.compose_file),
            "logs",
            "--tail",
            str(tail),
        ]
        proc = subprocess.run(
            command, cwd=str(self.compose_file.parent), capture_output=True
        )
        return proc.stdout.decode("utf-8", "replace")

    def stop(self) -> None:
        command = _compose_command(self.docker) + [
            "-f",
            str(self.compose_file),
            "down",
            "--volumes",
            "--remove-orphans",
        ]
        subprocess.run(command, cwd=str(self.compose_file.parent), capture_output=True)


def load(path: str | Path) -> Challenge:
    """Load a challenge from a directory or a ``challenge.json`` path."""
    directory = Path(path)
    if directory.is_file():
        directory = directory.parent
    directory = directory.resolve()
    meta_file = directory / "challenge.json"
    if not meta_file.is_file():
        raise ChallengeError(f"no challenge.json in {directory}")
    try:
        metadata = json.loads(meta_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ChallengeError(f"invalid challenge.json in {directory}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ChallengeError(f"challenge.json in {directory} is not an object")
    return Challenge(directory=directory, metadata=metadata)


def discover(root: str | Path) -> list[Challenge]:
    """Find every challenge under ``root``, sorted by category and name."""
    root_path = Path(root).resolve()
    if not root_path.exists():
        raise ChallengeError(f"root directory does not exist: {root_path}")
    challenges: list[Challenge] = []
    for meta_file in sorted(root_path.rglob("challenge.json")):
        try:
            challenges.append(load(meta_file.parent))
        except ChallengeError:
            continue
    challenges.sort(key=lambda challenge: (challenge.category, challenge.name.lower()))
    return challenges


def find_by_name(root: str | Path, name: str) -> Challenge:
    """Find a single challenge by name or directory name."""
    wanted = name.strip().lower()
    candidates = [
        challenge
        for challenge in discover(root)
        if challenge.name.lower() == wanted
        or challenge.directory.name.lower() == wanted
    ]
    if not candidates:
        raise ChallengeError(f"no challenge named {name!r} under {root}")
    if len(candidates) > 1:
        directories = ", ".join(str(challenge.directory) for challenge in candidates)
        raise ChallengeError(f"challenge name {name!r} is ambiguous: {directories}")
    return candidates[0]


def resolve(root: str | Path, value: str) -> Challenge:
    """Resolve a challenge from a directory, a challenge.json, or a name."""
    candidate = Path(value)
    if candidate.is_dir() and (candidate / "challenge.json").is_file():
        return load(candidate)
    if candidate.is_file() and candidate.name == "challenge.json":
        return load(candidate)
    return find_by_name(root, value)
