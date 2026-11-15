"""Challenge discovery, metadata, and challenge-service lifecycle."""

from __future__ import annotations

import json
import re
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

# Compose files in the dataset are YAML or JSON; match the key in both flavours.
COMPOSE_BUILD_PATTERN = re.compile(r"""(?:^|[{,])\s*['"]?build['"]?\s*:""", re.M)

REPO_ROOT = Path(__file__).resolve().parent.parent
OVERRIDES_FILE = REPO_ROOT / "overrides.toml"
OVERRIDE_FIELDS = ("files", "name", "build_files")
# Staged copies of a challenge's build context live here, never in competition/.
BUILD_CONTEXT_ROOT = REPO_ROOT / ".cache" / "build-contexts"

# Characters no filesystem path component may contain; Windows is strictest.
ILLEGAL_PATH_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def safe_dir_name(name: str, *, fallback: str = "challenge") -> str:
    """Filesystem-safe directory name for a challenge (names may contain ':'").

    ``Breach: Zero Day`` cannot be a Windows path component, and a dropped
    challenge is worse than an ugly directory name.
    """
    safe = ILLEGAL_PATH_CHARS.sub("-", str(name)).strip().strip(".")
    if safe.split(".")[0].upper() in WINDOWS_RESERVED_NAMES:
        safe = f"{safe}_"
    return safe or fallback


class ChallengeError(RuntimeError):
    """Raised when challenge data is missing or malformed."""


@dataclass
class Challenge:
    """One challenge loaded from ``challenge.json``."""

    directory: Path
    metadata: dict[str, Any] = field(repr=False)
    local_target: str | None = None
    local_services: tuple["ServiceTarget", ...] = ()
    local_discovery: bool = False
    staged_context: Path | None = field(default=None, repr=False, compare=False)

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

    def reachable_services(self) -> list["ServiceTarget"]:
        """Every service the sandbox can use, best first (empty when unknown).

        A started stack wins over the public address from ``challenge.json``,
        which is not necessarily reachable from inside the sandbox network.
        """
        if self.local_services:
            return list(self.local_services)
        if self.local_target:
            parsed = _target_from_address(self.local_target)
            return [parsed] if parsed else []
        if self.local_discovery:
            return []  # asked docker: nothing of this stack sits on our network
        if self.is_remote_only():
            return []
        if self.server_host and self.server_port:
            return [ServiceTarget(self.server_host, self.server_port, "")]
        return []

    def reachable_targets(self) -> list[str]:
        """Addresses the sandbox can use, best first."""
        return [service.address for service in self.reachable_services()]

    def reachable_target(self) -> str | None:
        """Address the sandbox can actually use: the started compose service."""
        targets = self.reachable_targets()
        return targets[0] if targets else None

    def discover_local_targets(
        self, docker: str, network: str | None = None
    ) -> list["ServiceTarget"]:
        """Resolve the started stack to services usable inside ``network``.

        Only containers attached to the sandbox's own network are reachable, and
        only the aliases they carry on that network resolve. A private backing
        service (an admin relay on its own network) must never be advertised as
        the challenge target.
        """
        self.local_services = ()
        self.local_target = None
        containers = self._service_containers(docker)
        if containers is None:
            return []
        self.local_discovery = True
        services = _local_targets_from(containers, self, network)
        if services:
            self.local_services = tuple(services)
            self.local_target = services[0].address
        return services

    def discover_local_target(
        self, docker: str, network: str | None = None
    ) -> str | None:
        """Primary reachable address of the started stack (see discovery above)."""
        self.discover_local_targets(docker, network)
        return self.local_target

    def _service_containers(self, docker: str) -> Any | None:
        """``docker inspect`` payload for the stack's containers, or None."""
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
            return json.loads(proc.stdout.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            return None

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
        """Compose file to build and run from (staged copy when patched)."""
        root = self.build_context()
        for base in (root, root / "infra"):
            for name in COMPOSE_NAMES:
                candidate = base / name
                if candidate.is_file():
                    return candidate
        return None

    @property
    def build_files(self) -> dict[str, str]:
        """Context files replaced from host copies (dataset corrections)."""
        raw = self.metadata.get("build_files") or {}
        if not isinstance(raw, dict):
            return {}
        return {str(key): str(value) for key, value in raw.items()}

    def build_context(self) -> Path:
        """Directory docker builds from: the dataset, or a patched staged copy.

        ``competition/`` is read-only, so a broken build file is corrected by
        copying the context under ``.cache/build-contexts/`` and overwriting the
        listed files there.
        """
        patches = self.build_files
        if not patches:
            return self.directory
        if self.staged_context is not None:
            return self.staged_context
        staged = BUILD_CONTEXT_ROOT / safe_dir_name(
            f"{self.category}_{self.directory.name}"
        )
        if staged.exists():
            shutil.rmtree(staged)
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.directory, staged)
        for relative, source in patches.items():
            origin = Path(source)
            if not origin.is_absolute():
                origin = (REPO_ROOT / origin).resolve()
                try:
                    origin.relative_to(REPO_ROOT)
                except ValueError as exc:
                    raise ChallengeError(
                        f"build_files source {source!r} must live inside the repo"
                    ) from exc
            if not origin.is_file():
                raise ChallengeError(
                    f"build_files source {source!r} for {self.name!r} does not exist"
                )
            target = (staged / relative).resolve()
            try:
                target.relative_to(staged.resolve())
            except ValueError as exc:
                raise ChallengeError(
                    f"build_files target {relative!r} escapes the build context"
                ) from exc
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, target)
        self.staged_context = staged
        return staged

    def start_service(
        self, docker: str, *, wait: float = 0.0, build: bool = True
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
        if not build:
            command.append("--no-build")
        proc = subprocess.run(command, cwd=str(compose.parent), capture_output=True)
        if proc.returncode != 0:
            raise ChallengeError(
                "failed to start challenge service:\n" + _compose_output(proc)
            )
        if wait > 0:
            time.sleep(wait)
        return ChallengeService(compose_file=compose, docker=docker)

    def build_images(self, docker: str, *, log_path: Path | None = None) -> None:
        """Build the challenge's compose images; nothing starts yet.

        Pull-only stacks (no ``build`` key) are left alone: their images are
        fetched by ``up``, and ``--no-build`` does not stop compose pulling.
        """
        compose = self.compose_file()
        if compose is None:
            return
        try:
            text = compose.read_text(encoding="utf-8")
        except OSError as exc:
            raise ChallengeError(f"cannot read {compose}: {exc}") from exc
        if not COMPOSE_BUILD_PATTERN.search(text):
            if log_path is not None:
                _write_log(log_path, "no build section; image comes from a registry\n")
            return
        command = _compose_command(docker) + ["-f", str(compose), "build"]
        proc = subprocess.run(command, cwd=str(compose.parent), capture_output=True)
        output = _compose_output(proc)
        if log_path is not None:
            _write_log(log_path, output)
        if proc.returncode != 0:
            raise ChallengeError(
                f"failed to build challenge images in {compose.parent}:\n"
                + _error_tail(output)
            )


@dataclass(frozen=True)
class ServiceTarget:
    """One started compose service, addressed the way the sandbox must reach it."""

    host: str
    port: int
    service: str = ""

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def label(self) -> str:
        """``host:port`` plus the compose service name when that adds information."""
        if self.service and self.service != self.host:
            return f"{self.address} ({self.service})"
        return self.address


def _target_from_address(address: str) -> "ServiceTarget | None":
    """Parse ``host:port`` back into a target (for externally supplied values)."""
    host, separator, port = str(address).rpartition(":")
    if not separator or not host or not port.isdigit():
        return None
    return ServiceTarget(host, int(port))


def _exposed_ports(container: Any) -> list[int]:
    """Container ports from ``ExposedPorts`` (compose ``expose`` or Dockerfile)."""
    config = container.get("Config") or {}
    ports = {
        int(str(key).split("/")[0])
        for key in (config.get("ExposedPorts") or {})
        if str(key).split("/")[0].isdigit()
    }
    return sorted(ports)


def _network_aliases(container: Any, network: str | None) -> list[str]:
    """Aliases a container carries, restricted to ``network`` when one is given."""
    settings = (container.get("NetworkSettings") or {}).get("Networks") or {}
    aliases: list[str] = []
    for name, entry in settings.items():
        if network is not None and str(name) != network:
            continue
        aliases.extend(str(alias) for alias in (entry or {}).get("Aliases") or [])
    return aliases


def _local_targets_from(
    containers: Any, challenge: "Challenge", network: str | None = None
) -> list[ServiceTarget]:
    """Pick ``<network alias or service>:<container port>`` from docker inspect.

    Without ``network`` every attached network counts (legacy behaviour). With
    one, containers outside it are skipped: they exist, but the sandbox has no
    route to them, and naming one sends the agent chasing an unresolvable host.
    """
    found: list[ServiceTarget] = []
    seen: set[str] = set()
    for container in containers if isinstance(containers, list) else []:
        ports = _exposed_ports(container)
        if not ports:
            continue
        aliases = _network_aliases(container, network)
        if network is not None and not aliases:
            continue
        config = container.get("Config") or {}
        labels = config.get("Labels") or {}
        service = str(labels.get("com.docker.compose.service") or "")
        skip = {"localhost", str(container.get("Name") or "").lstrip("/")}
        candidates = [name for name in aliases if name and name not in skip]
        # Docker always adds the service and container names as aliases; the
        # alias a human wrote in compose is the dotted one, so prefer it.
        candidates.sort(
            key=lambda name: (name != challenge.server_host, "." not in name, name)
        )
        if candidates:
            host = candidates[0]
        elif service:
            host = service
        elif network is None:
            host = "localhost"
        else:
            continue
        port = (
            challenge.server_port
            if challenge.server_port in ports
            else ports[0]
        )
        address = f"{host}:{port}"
        if address in seen:
            continue
        seen.add(address)
        found.append(ServiceTarget(host, int(port), service))

    def rank(target: ServiceTarget) -> tuple[int, str, int]:
        preferred = (
            challenge.server_host is not None and target.host == challenge.server_host
        ) or (
            challenge.server_host is None
            and challenge.server_port is not None
            and target.port == challenge.server_port
        )
        return (0 if preferred else 1, target.host, target.port)

    found.sort(key=rank)
    return found


def _compose_output(proc: subprocess.CompletedProcess) -> str:
    """Compose writes build progress to stdout and failures to stderr."""
    chunks = [
        proc.stdout.decode("utf-8", "replace"),
        proc.stderr.decode("utf-8", "replace"),
    ]
    return "\n".join(chunk.strip() for chunk in chunks if chunk.strip())


def _error_tail(output: str, limit: int = 2500) -> str:
    """The failing step is at the end of a long build log."""
    text = output.strip()
    if len(text) <= limit:
        return text
    return "...\n" + text[-limit:]


def _write_log(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


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
    build_files = entry.get("build_files")
    if build_files is not None:
        if not isinstance(build_files, dict) or not all(
            isinstance(name, str) and isinstance(source, str)
            for name, source in build_files.items()
        ):
            raise ChallengeError(
                f"override {key!r}: build_files must map context paths to source "
                "files, e.g. { \"build/Dockerfile\" = \"fixes/x/Dockerfile\" }"
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
    if "build_files" in entry:
        merged["build_files"] = dict(entry["build_files"])
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
