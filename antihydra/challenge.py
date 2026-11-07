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

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 has no tomllib
    tomllib = None

if TYPE_CHECKING:
    from .sandbox import Sandbox

COMPOSE_NAMES = (
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yml",
    "compose.yaml",
)

OVERRIDES_FILE = Path(__file__).resolve().parent.parent / "overrides.toml"
OVERRIDE_FIELDS = ("files", "name")


class ChallengeError(RuntimeError):
    """Raised when challenge data is missing or malformed."""


@dataclass
class Challenge:
    """One challenge loaded from ``challenge.json``."""

    directory: Path
    metadata: dict[str, Any] = field(repr=False)
    local_target: str | None = None

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
        """The public address from ``challenge.json`` (not necessarily reachable here)."""
        if not self.server_host or not self.server_port:
            return None
        return f"{self.server_host}:{self.server_port}"

    def is_remote_only(self) -> bool:
        """Declares a public service but ships no compose stack to run locally."""
        return self.server_host is not None and self.compose_file() is None

    def reachable_target(self) -> str | None:
        """Address the sandbox can actually use: the started compose service."""
        if self.local_target:
            return self.local_target
        if self.is_remote_only():
            return None
        return self.target_line()

    def discover_local_target(self, docker: str) -> str | None:
        """Resolve the started stack to an address usable from inside ctfnet.

        Compose publishes host ports, but the sandbox sits on the challenge
        network, where the container listens on its own port. The stack's network
        aliases carry the public hostname.
        """
        compose = self.compose_file()
        if compose is None:
            return None
        command = _compose_command(docker) + ["-f", str(compose), "ps", "-q"]
        proc = subprocess.run(command, cwd=str(compose.parent), capture_output=True)
        if proc.returncode != 0:
            return None
        ids = [line for line in proc.stdout.decode("utf-8", "replace").split() if line]
        if not ids:
            return None
        proc = subprocess.run(
            [docker, "inspect", *ids], capture_output=True, timeout=60
        )
        if proc.returncode != 0:
            return None
        try:
            containers = json.loads(proc.stdout.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            return None
        self.local_target = _local_target_from(containers, self)
        return self.local_target

    def local_files(self) -> list[Path]:
        found: list[Path] = []
        for entry in self.files:
            candidate = (self.directory / entry).resolve()
            if candidate.exists():
                found.append(candidate)
        return found

    def missing_files(self) -> list[str]:
        return [entry for entry in self.files if not (self.directory / entry).exists()]

    def require_delivered_files(self) -> None:
        """Fail loudly when the manifest lists files that are not on disk."""
        missing = self.missing_files()
        if not missing:
            return
        key = override_key(self.directory, self.category)
        raise ChallengeError(
            f"{key} lists files that do not exist: {', '.join(missing)}; "
            f'add a ["{key}"] entry with a corrected files list to '
            f"{OVERRIDES_FILE.name} (the dataset itself is read-only)"
        )

    def install_files(self, sandbox: "Sandbox") -> None:
        """Copy the challenge's files into the sandbox."""
        self.require_delivered_files()
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


def _local_target_from(containers: Any, challenge: "Challenge") -> str | None:
    """Pick ``<network alias or service>:<container port>`` from docker inspect."""
    for container in containers if isinstance(containers, list) else []:
        config = container.get("Config") or {}
        exposed = [
            str(key).split("/")[0]
            for key in (config.get("ExposedPorts") or {})
            if str(key).split("/")[0].isdigit()
        ]
        if not exposed:
            continue
        port = (
            str(challenge.server_port)
            if str(challenge.server_port) in exposed
            else exposed[0]
        )
        labels = config.get("Labels") or {}
        service = str(labels.get("com.docker.compose.service") or "")
        networks = (container.get("NetworkSettings") or {}).get("Networks") or {}
        aliases: list[str] = []
        for network in networks.values():
            aliases.extend(str(name) for name in (network.get("Aliases") or []))
        unordered = [
            name
            for name in aliases
            if name and name not in {"localhost", container.get("Name", "").lstrip("/")}
        ]
        unordered.sort(key=lambda name: name != challenge.server_host)
        host = unordered[0] if unordered else (service or "localhost")
        return f"{host}:{port}"
    return None


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


def override_key(directory: Path, category: str) -> str:
    """Stable key for overrides: ``<category>/<directory name>``."""
    return f"{category.lower()}/{directory.name}"


def _validate_override(key: str, entry: Any) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ChallengeError(f"override {key!r} must be a table")
    unknown = sorted(set(entry) - set(OVERRIDE_FIELDS))
    if unknown:
        raise ChallengeError(
            f"override {key!r} sets unsupported field(s) {unknown}; "
            f"allowed: {list(OVERRIDE_FIELDS)}"
        )
    return dict(entry)


def load_overrides(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Per-challenge corrections kept in our repo because the dataset is read-only."""
    file_path = Path(path) if path is not None else OVERRIDES_FILE
    if not file_path.is_file():
        return {}
    if tomllib is None:
        raise ChallengeError(f"{file_path.name} needs Python 3.11 or newer")
    try:
        data = tomllib.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ChallengeError(f"invalid {file_path.name}: {exc}") from exc
    return {
        str(key): _validate_override(str(key), entry) for key, entry in data.items()
    }


def _apply_overrides(
    metadata: dict[str, Any], directory: Path, overrides: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    category = str(metadata.get("category") or directory.parent.name)
    key = override_key(directory, category)
    entry = overrides.get(key)
    if not entry:
        return metadata
    entry = _validate_override(key, entry)
    merged = dict(metadata)
    if "name" in entry:
        merged["name"] = str(entry["name"])
    if "files" in entry:
        raw = entry["files"]
        if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
            raise ChallengeError(f"override {key!r}: files must be a list of paths")
        merged["files"] = [str(item) for item in raw]
    return merged


def load(
    path: str | Path, *, overrides: dict[str, dict[str, Any]] | None = None
) -> Challenge:
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
    if overrides is None:
        overrides = load_overrides()
    metadata = _apply_overrides(metadata, directory, overrides)
    return Challenge(directory=directory, metadata=metadata)


def discover(root: str | Path) -> list[Challenge]:
    """Find every challenge under ``root``, sorted by category and name."""
    root_path = Path(root).resolve()
    if not root_path.exists():
        raise ChallengeError(f"root directory does not exist: {root_path}")
    overrides = load_overrides()
    challenges: list[Challenge] = []
    for meta_file in sorted(root_path.rglob("challenge.json")):
        try:
            challenges.append(load(meta_file.parent, overrides=overrides))
        except ChallengeError:
            continue
    challenges.sort(key=lambda challenge: (challenge.category, challenge.name.lower()))
    used = {
        override_key(challenge.directory, challenge.category)
        for challenge in challenges
    }
    for key in sorted(set(overrides) - used):
        print(f"[warn] dataset override {key!r} matches no challenge", file=sys.stderr)
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
