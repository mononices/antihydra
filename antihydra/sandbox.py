"""Docker sandbox management using the docker CLI (no third-party SDK)."""

from __future__ import annotations

import base64
import binascii
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, IO

LEASE_LABEL = "antihydra.sandbox"
KEEP_ALIVE = "tail -f /dev/null"

FLOOD_REPEAT = 60  # identical lines inside the recent window that count as a loop
FLOOD_WINDOW = 500  # how many recent lines the window holds
FLOOD_BYTES = 2_000_000  # backstop for output that never repeats a line
FLOOD_MIN_LINE = 8  # ignore blank/short lines (progress bars, brace soup)

_WINDOWS_DOCKER_HINTS = (
    r"C:\Program Files\Docker\Docker\resources\bin\docker.exe",
    r"C:\ProgramData\chocolatey\bin\docker.exe",
)


class SandboxError(RuntimeError):
    """Sandbox lifecycle or command execution failure."""


def _windows_docker_candidates() -> list[Path]:
    """Common Docker Desktop CLI locations, including per-user installs."""
    local_appdata = Path(
        os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
    )
    per_user = [
        local_appdata
        / "Programs"
        / "DockerDesktop"
        / "resources"
        / "bin"
        / "docker.exe",
        local_appdata
        / "Programs"
        / "Docker"
        / "Docker"
        / "resources"
        / "bin"
        / "docker.exe",
    ]
    return [*per_user, *(Path(hint) for hint in _WINDOWS_DOCKER_HINTS)]


def find_docker() -> str:
    """Locate the docker CLI, including common Windows install locations."""
    found = shutil.which("docker")
    if found:
        return found
    if sys.platform == "win32":
        for candidate in _windows_docker_candidates():
            if candidate.is_file():
                return str(candidate)
    raise SandboxError(
        "docker CLI not found; install Docker Desktop (or add docker to PATH)"
    )


def daemon_version(docker: str) -> str:
    """Return the docker server version, or raise if the daemon is unreachable."""
    proc = subprocess.run(
        [docker, "version", "--format", "{{.Server.Version}}"], capture_output=True
    )
    if proc.returncode != 0:
        raise SandboxError(
            "docker CLI found but the daemon is unreachable:\n"
            + proc.stderr.decode("utf-8", "replace").strip()
        )
    return proc.stdout.decode("utf-8", "replace").strip()


def ensure_network(docker: str, name: str) -> None:
    """Create the docker network if it does not exist yet."""
    if (
        subprocess.run(
            [docker, "network", "inspect", name], capture_output=True
        ).returncode
        == 0
    ):
        return
    proc = subprocess.run([docker, "network", "create", name], capture_output=True)
    if proc.returncode != 0:
        raise SandboxError(
            "failed to create docker network:\n"
            + proc.stderr.decode("utf-8", "replace").strip()
        )


def cleanup_orphans(docker: str, label: str = LEASE_LABEL) -> int:
    """Force-remove leftover sandbox containers from previous runs."""
    proc = subprocess.run(
        [docker, "ps", "-aq", "--filter", f"label={label}"], capture_output=True
    )
    ids = proc.stdout.decode("utf-8", "replace").split()
    if ids:
        subprocess.run([docker, "rm", "-f", *ids], capture_output=True)
    return len(ids)


MEMORY_UNITS = {"k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}


def parse_memory(value: str) -> int | None:
    """Parse docker-style memory strings like ``4g`` into bytes."""
    text = value.strip().lower()
    if text.isdigit():
        return int(text)
    if text and text[-1] in MEMORY_UNITS:
        try:
            return int(float(text[:-1]) * MEMORY_UNITS[text[-1]])
        except ValueError:
            return None
    return None


def warn_memory(
    docker: str, memory: str, containers: int, say: Callable[[str], None]
) -> None:
    """Warn when ``containers`` sandboxes of ``memory`` each exceed the Docker VM."""
    per_container = parse_memory(memory)
    if per_container is None:
        return
    probe = subprocess.run(
        [docker, "info", "--format", "{{.MemTotal}}"], capture_output=True
    )
    try:
        total = int(probe.stdout.decode("utf-8", "replace").strip())
    except ValueError:
        return
    if total and containers * per_container > total * 0.9:
        say(
            f"[warn] {containers} sandbox(es) x {memory} exceeds the Docker VM's "
            f"{total / 1024**3:.1f} GiB; lower the memory limit or concurrency"
        )


def _decode(data: bytes | None) -> str:
    if not data:
        return ""
    return data.decode("utf-8", "replace").replace("\r\n", "\n")


class OutputFlood:
    """Trips when a command keeps printing the same output.

    An interactive binary handed a closed stdin spins forever ("no such
    option"), which otherwise costs the whole command timeout and floods the
    transcript. Both pipes get their own guard.
    """

    def __init__(
        self,
        repeat: int = FLOOD_REPEAT,
        window: int = FLOOD_WINDOW,
        max_bytes: int = FLOOD_BYTES,
    ) -> None:
        self.repeat = repeat
        self.max_bytes = max_bytes
        self.bytes = 0
        self.line: str | None = None
        self.count = 0
        self._window: deque[str] = deque(maxlen=window)
        self._counts: Counter[str] = Counter()
        self._pending = ""

    def feed(self, text: str) -> bool:
        """Consume one decoded chunk; True once the command should be killed."""
        self.bytes += len(text)
        if self.bytes > self.max_bytes:
            self.count = self.bytes
            return True
        self._pending += text
        lines = self._pending.split("\n")
        self._pending = lines.pop()
        return any(self._add(line.strip()) for line in lines)

    def describe(self) -> str:
        """Short reason for the kill, safe to hand back to the model."""
        if self.line:
            return f"line repeated {self.count}x: {self.line[:120]!r}"
        return f"output passed {self.bytes} bytes without finishing"

    def _add(self, line: str) -> bool:
        if len(line) < FLOOD_MIN_LINE:
            return False
        if len(self._window) == self._window.maxlen:
            dropped = self._window[0]
            self._counts[dropped] -= 1
            if self._counts[dropped] <= 0:
                del self._counts[dropped]
        self._window.append(line)
        self._counts[line] += 1
        if self._counts[line] >= self.repeat:
            self.line = line
            self.count = self._counts[line]
            return True
        return False


@dataclass
class ExecResult:
    """Result of a command executed inside the sandbox."""

    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool
    killed: str | None = None
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def to_dict(self) -> dict:
        payload = {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
        }
        if self.killed:
            payload["killed"] = self.killed
            payload["killed_note"] = self.note
        return payload


def _run_streaming(argv: list[str], timeout: float) -> ExecResult:
    """Run ``argv`` while watching both pipes so runaway output dies early.

    ``subprocess.run`` can only kill on the timeout, so a command that loops
    printing the same line (an interactive binary with no stdin) used to eat
    the full timeout. Reader threads keep the guard fed line by line instead.
    """
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    guards = {"out": OutputFlood(), "err": OutputFlood()}
    trip = threading.Event()
    stdout: list[str] = []
    stderr: list[str] = []

    def kill() -> None:
        try:
            proc.kill()
        except OSError:
            pass

    def pump(stream: IO[bytes] | None, parts: list[str], guard: OutputFlood) -> None:
        try:
            while True:
                chunk = stream.readline(65536) if stream else b""
                if not chunk:
                    break
                text = _decode(chunk)
                if trip.is_set():
                    continue  # keep draining so the dying process never blocks
                parts.append(text)
                if guard.feed(text):
                    trip.set()
                    kill()
        except (OSError, ValueError):
            pass

    readers = [
        threading.Thread(
            target=pump, args=(proc.stdout, stdout, guards["out"]), daemon=True
        ),
        threading.Thread(
            target=pump, args=(proc.stderr, stderr, guards["err"]), daemon=True
        ),
    ]
    for reader in readers:
        reader.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        kill()
        proc.wait()
    for reader in readers:
        reader.join(timeout=5)
    for stream in (proc.stdout, proc.stderr):
        try:
            if stream:
                stream.close()
        except OSError:
            pass

    killed: str | None = "timeout" if timed_out else None
    note = ""
    if trip.is_set():
        guard = next(
            (item for item in guards.values() if item.line),
            max(guards.values(), key=lambda item: item.bytes),
        )
        killed, note = "output_flood", guard.describe()
    return ExecResult(
        stdout="".join(stdout),
        stderr="".join(stderr),
        exit_code=None if killed else proc.returncode,
        timed_out=timed_out,
        killed=killed,
        note=note,
    )


class Sandbox:
    """One long-lived sandbox container driven via ``docker exec``."""

    def __init__(
        self,
        image: str,
        *,
        docker: str | None = None,
        name: str | None = None,
        network: str | None = None,
        platform: str | None = None,
        memory: str | None = None,
        user: str | None = None,
        workdir: str | None = None,
        label: str = LEASE_LABEL,
    ) -> None:
        self.docker = docker or find_docker()
        self.image = image
        self.name = name or f"antihydra-{uuid.uuid4().hex[:10]}"
        self.network = network
        self.platform = platform
        self.memory = memory
        self.user = user
        self.workdir = workdir
        self.label = label
        self.container_id: str | None = None
        self._stop_lock = threading.Lock()
        self.shell = "bash"

    def start(self, timeout: float = 120) -> "Sandbox":
        """Start the container and detect the available shell."""
        if self.container_id:
            return self
        self._require_image(timeout)
        command = [
            self.docker,
            "run",
            "-d",
            "--rm",
            "--name",
            self.name,
            "--label",
            self.label,
        ]
        if self.network:
            command += ["--network", self.network]
        if self.platform:
            command += ["--platform", self.platform]
        if self.memory:
            command += ["--memory", self.memory]
        if self.user:
            command += ["--user", self.user]
        if self.workdir:
            command += ["-w", self.workdir]
        command += [self.image, "sh", "-c", KEEP_ALIVE]
        proc = subprocess.run(command, capture_output=True, timeout=timeout)
        silent = not _decode(proc.stderr).strip() and not _decode(proc.stdout).strip()
        if proc.returncode != 0 and silent:
            # A silent failure is a daemon hiccup rather than a bad image or flag.
            time.sleep(1.0)
            proc = subprocess.run(command, capture_output=True, timeout=timeout)
        if proc.returncode != 0:
            detail = (
                _decode(proc.stderr).strip()
                or _decode(proc.stdout).strip()
                or "no output"
            )
            raise SandboxError(
                f"failed to start sandbox container (exit {proc.returncode}):\n{detail}"
            )
        self.container_id = _decode(proc.stdout).strip()
        self.shell = self._detect_shell()
        return self

    def _require_image(self, timeout: float) -> None:
        proc = subprocess.run(
            [self.docker, "image", "inspect", self.image],
            capture_output=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            raise SandboxError(
                f"sandbox image {self.image!r} is not available locally; "
                "build it (bash docker/build_sandbox.sh) or pass --image with an existing tag"
            )

    def _detect_shell(self) -> str:
        assert self.container_id
        for candidate in ("bash", "sh"):
            proc = subprocess.run(
                [self.docker, "exec", self.container_id, candidate, "-c", "true"],
                capture_output=True,
                timeout=60,
            )
            if proc.returncode == 0:
                return candidate
        return "sh"

    def stop(self) -> None:
        """Stop and remove the container; idempotent and safe under concurrent calls."""
        with self._stop_lock:
            container_id, self.container_id = self.container_id, None
        if not container_id:
            return
        subprocess.run(
            [self.docker, "stop", "-t", "10", container_id],
            capture_output=True,
            timeout=120,
        )
        subprocess.run(
            [self.docker, "rm", "-f", container_id],
            capture_output=True,
            timeout=120,
        )

    def running(self) -> bool:
        if not self.container_id:
            return False
        proc = subprocess.run(
            [self.docker, "inspect", "-f", "{{.State.Running}}", self.container_id],
            capture_output=True,
        )
        return _decode(proc.stdout).strip() == "true"

    def __enter__(self) -> "Sandbox":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def exec(self, command: str, timeout: float = 300) -> ExecResult:
        """Run a shell command inside the container."""
        if not self.container_id:
            raise SandboxError("sandbox is not running")
        return _run_streaming(
            [self.docker, "exec", self.container_id, self.shell, "-c", command],
            timeout=timeout,
        )

    def copy_in(self, host_path: str | Path, container_path: str) -> None:
        """Copy a host file or directory into the container."""
        if not self.container_id:
            raise SandboxError("sandbox is not running")
        source = Path(host_path).resolve()
        if not source.exists():
            raise SandboxError(f"no such host path: {source}")
        parent = str(PurePosixPath(container_path).parent)
        mkdir = self.exec(f"mkdir -p {shlex.quote(parent)}")
        if not mkdir.ok:
            raise SandboxError(
                f"failed to create {parent} in sandbox: {mkdir.stderr.strip()}"
            )
        proc = subprocess.run(
            [self.docker, "cp", str(source), f"{self.container_id}:{container_path}"],
            capture_output=True,
            timeout=300,
        )
        if proc.returncode != 0:
            raise SandboxError("docker cp failed:\n" + _decode(proc.stderr).strip())

    def write_text(self, container_path: str, content: str) -> None:
        """Write text to a file inside the container."""
        if not self.container_id:
            raise SandboxError("sandbox is not running")
        parent = str(PurePosixPath(container_path).parent)
        script = (
            f"mkdir -p {shlex.quote(parent)} && cat > {shlex.quote(container_path)}"
        )
        proc = subprocess.run(
            [self.docker, "exec", "-i", self.container_id, "sh", "-c", script],
            input=content.encode("utf-8"),
            capture_output=True,
            timeout=300,
        )
        if proc.returncode != 0:
            raise SandboxError("failed to write file:\n" + _decode(proc.stderr).strip())

    def file_size(self, container_path: str) -> int:
        """Size of a file inside the container, or -1 when it does not exist."""
        result = self.exec(f"wc -c < {shlex.quote(container_path)}")
        if result.exit_code != 0:
            return -1
        try:
            return int(result.stdout.strip().split()[0])
        except (IndexError, ValueError):
            return -1

    def read_bytes(self, container_path: str, limit: int = 4_000_000) -> bytes:
        """Read a container file as bytes, refusing anything over ``limit``."""
        size = self.file_size(container_path)
        if size < 0:
            raise SandboxError(f"no such file: {container_path}")
        if size > limit:
            raise SandboxError(
                f"{container_path} is {size} bytes; the read limit is {limit} bytes"
            )
        result = self.exec(f"base64 -w0 {shlex.quote(container_path)}", timeout=120)
        if result.exit_code != 0:
            raise SandboxError(
                f"failed to read {container_path}: {result.stderr.strip()}"
            )
        try:
            return base64.b64decode(result.stdout, validate=False)
        except (ValueError, binascii.Error) as exc:
            raise SandboxError(f"failed to decode {container_path}: {exc}") from exc

    def read_text(self, container_path: str, limit: int | None = None) -> str:
        """Read a file from the container (optionally truncated to ``limit`` chars)."""
        result = self.exec(f"cat {shlex.quote(container_path)}")
        if result.exit_code != 0:
            raise SandboxError(
                f"failed to read {container_path}: {result.stderr.strip()}"
            )
        text = result.stdout
        if limit is not None and len(text) > limit:
            text = text[:limit]
        return text
