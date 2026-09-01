"""
Hermetic container execution, for the eval harness.

The eval runs model-generated code against fixture repositories, so it needs
real isolation and a clean slate per task — otherwise one task's leftovers
change the next task's result, and the numbers stop meaning anything.

The timeout is enforced *inside* the container by coreutils `timeout`. Wrapping
`exec_run` in `asyncio.wait_for` looks equivalent and is not: it cancels the
await, not the thread and not the process, so a runaway `while True:` keeps
burning the container's CPU while the caller is told it timed out — and the
thread-pool slot never comes back.
"""

from __future__ import annotations

import asyncio
import io
import logging
import shlex
import tarfile
import time
import uuid
from pathlib import Path

from codepilot.sandbox.base import CommandResult
from codepilot.sandbox.pytest_parse import TestOutcome, parse

log = logging.getLogger(__name__)

WORKDIR = "/workspace"
DEFAULT_IMAGE = "codepilot-sandbox:latest"
#: GNU timeout's exit code when it kills the child.
TIMEOUT_EXIT = 124


class DockerUnavailable(RuntimeError):
    pass


class DockerSandbox:
    """Implements `Sandbox` against a throwaway container."""

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        *,
        memory: str = "512m",
        cpus: int = 1,
        network: str = "none",
    ) -> None:
        try:
            import docker  # noqa: PLC0415 - optional dependency, only for evals
        except ImportError as exc:
            raise DockerUnavailable(
                "the docker SDK is not installed — `pip install docker`"
            ) from exc
        try:
            self._docker = docker.from_env()
            self._docker.ping()
        except Exception as exc:  # noqa: BLE001 - any failure means unusable
            raise DockerUnavailable(f"the Docker daemon is not reachable: {exc}") from exc

        self._image = image
        self._memory = memory
        self._cpus = cpus
        self._network = network
        self._container = None
        self.id = uuid.uuid4().hex[:12]

    async def start(self, files: dict[str, str] | None = None) -> str:
        def _start():
            return self._docker.containers.run(
                self._image,
                command="sleep infinity",
                detach=True,
                remove=True,
                mem_limit=self._memory,
                nano_cpus=self._cpus * 1_000_000_000,
                network_mode=self._network,
                user="nobody",
                working_dir=WORKDIR,
                # uid must match the user we exec as, or every write is denied.
                tmpfs={WORKDIR: "size=256m,uid=65534,mode=1777"},
                name=f"codepilot_{self.id}",
            )

        self._container = await asyncio.to_thread(_start)
        for path, content in (files or {}).items():
            await self.write(path, content)
        return self.id

    async def write(self, path: str, content: str) -> None:
        if self._container is None:
            raise DockerUnavailable("container not started — call start() first")
        # tar member names carry directories, so nested paths arrive intact. A
        # `..` in the name would escape WORKDIR on extraction, so reject it here
        # rather than trusting whatever produced the path.
        clean = Path(path.replace("\\", "/"))
        if clean.is_absolute() or ".." in clean.parts:
            raise ValueError(f"refusing to write outside the sandbox: {path!r}")

        data = content.encode("utf-8")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name=clean.as_posix())
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
        payload = buf.getvalue()
        await asyncio.to_thread(self._container.put_archive, WORKDIR, payload)

    async def run(self, command: str, timeout_seconds: int = 60) -> CommandResult:
        if self._container is None:
            raise DockerUnavailable("container not started — call start() first")
        started = time.monotonic()
        # The kernel enforces this, so a runaway loop actually dies.
        wrapped = f"timeout -k 2 {timeout_seconds} sh -c {shlex.quote(command)}"

        def _exec():
            return self._container.exec_run(
                cmd=["sh", "-c", wrapped], workdir=WORKDIR, demux=True
            )

        result = await asyncio.to_thread(_exec)
        stdout_b, stderr_b = result.output or (b"", b"")
        timed_out = result.exit_code == TIMEOUT_EXIT
        return CommandResult(
            command=command,
            stdout=(stdout_b or b"").decode("utf-8", errors="replace"),
            stderr=(stderr_b or b"").decode("utf-8", errors="replace")
            + (f"\n[timed out after {timeout_seconds}s]" if timed_out else ""),
            exit_code=result.exit_code,
            duration_ms=int((time.monotonic() - started) * 1000),
            timed_out=timed_out,
        )

    async def run_tests(
        self, command: str = "pytest -q", timeout_seconds: int = 300
    ) -> TestOutcome:
        result = await self.run(command, timeout_seconds=timeout_seconds)
        outcome = parse(
            result.combined,
            result.exit_code,
            command=command,
            duration_ms=result.duration_ms,
        )
        outcome.timed_out = result.timed_out
        return outcome

    async def close(self) -> None:
        if self._container is None:
            return
        container, self._container = self._container, None
        try:
            await asyncio.to_thread(container.stop, timeout=5)
        except Exception:  # noqa: BLE001 - teardown must not mask the real error
            # A container that will not stop leaks resources across an eval
            # sweep, which is worth knowing about even though it must not
            # replace whatever error brought us here.
            log.warning("could not stop container %s", self.id, exc_info=True)
