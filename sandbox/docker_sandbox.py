"""
DockerSandbox — executes code inside an isolated Docker container.

Security model:
  - Fresh container per session (--rm)
  - Memory capped at 512 MB
  - CPU capped at 1 core
  - Network disabled (--network=none) by default
  - Non-root user (nobody)
  - /workspace is the only writable directory (tmpfs)
  - Host filesystem mounted read-only (not mounted at all — files are copied in)

Requires: Docker daemon running locally and 'docker' SDK installed.
"""

from __future__ import annotations

import asyncio
import io
import re
import tarfile
import time
import uuid

from models.schemas import ExecutionResult, TestResult

try:
    import docker
    from docker.models.containers import Container

    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False


SANDBOX_IMAGE = "codepilot-sandbox:latest"  # built from sandbox/docker/Dockerfile


class DockerSandbox:
    """
    Runs code in a Docker container with full resource and network isolation.
    Implements the SandboxBackend protocol.
    """

    def __init__(
        self,
        image: str = SANDBOX_IMAGE,
        memory_limit: str = "512m",
        cpu_quota: int = 100_000,  # 1 CPU core (100% of one CPU)
        network_mode: str = "none",
        timeout_default: int = 30,
    ) -> None:
        if not DOCKER_AVAILABLE:
            raise RuntimeError(
                "docker SDK not installed. Run: pip install docker\n"
                "Or use sandbox_type='subprocess' for local development."
            )
        self._client = docker.from_env()
        self._image = image
        self._memory_limit = memory_limit
        self._cpu_quota = cpu_quota
        self._network_mode = network_mode
        self._timeout_default = timeout_default

        self.sandbox_id: str = ""
        self._container: Container | None = None
        self._workdir_in_container = "/workspace"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def setup(self, workspace_files: dict[str, str]) -> str:
        self.sandbox_id = str(uuid.uuid4())[:12]

        # Start the container — long-running so we can exec into it
        loop = asyncio.get_event_loop()
        self._container = await loop.run_in_executor(
            None,
            lambda: self._client.containers.run(
                self._image,
                command="tail -f /dev/null",  # keep alive
                detach=True,
                remove=True,
                mem_limit=self._memory_limit,
                cpu_quota=self._cpu_quota,
                network_mode=self._network_mode,
                user="nobody",
                working_dir=self._workdir_in_container,
                tmpfs={self._workdir_in_container: "size=256m,uid=65534"},
                name=f"codepilot_{self.sandbox_id}",
            ),
        )

        # Copy workspace files into the container
        for rel_path, content in workspace_files.items():
            await self.write_file(rel_path, content)

        return self.sandbox_id

    async def teardown(self) -> None:
        if self._container:
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(
                    None, lambda: self._container.stop(timeout=5)
                )
            except Exception:
                pass
            self._container = None

    # ------------------------------------------------------------------
    # File I/O (via Docker archive API)
    # ------------------------------------------------------------------

    async def write_file(self, path: str, content: str) -> None:
        self._assert_ready()
        data = content.encode("utf-8")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name=path)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        buf.seek(0)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self._container.put_archive(self._workdir_in_container, buf.read()),
        )

    async def read_file(self, path: str) -> str:
        self._assert_ready()
        result = await self.run_command(f"cat {path}", timeout_seconds=10)
        if result.exit_code != 0:
            raise FileNotFoundError(f"File not found in sandbox: {path}")
        return result.stdout

    async def list_files(self, directory: str = ".") -> list[str]:
        self._assert_ready()
        result = await self.run_command(
            f"find {directory} -type f ! -path '*/.git/*' ! -path '*/__pycache__/*'",
            timeout_seconds=10,
        )
        if result.exit_code != 0:
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(
        self,
        code: str,
        filename: str = "run.py",
        timeout_seconds: int = 30,
    ) -> ExecutionResult:
        await self.write_file(filename, code)
        return await self.run_command(f"python {filename}", timeout_seconds)

    async def run_command(
        self,
        command: str,
        timeout_seconds: int = 60,
    ) -> ExecutionResult:
        self._assert_ready()
        start = time.monotonic()
        loop = asyncio.get_event_loop()

        try:
            exec_result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self._container.exec_run(
                        cmd=["sh", "-c", command],
                        workdir=self._workdir_in_container,
                        demux=True,
                    ),
                ),
                timeout=timeout_seconds,
            )
            stdout_b, stderr_b = exec_result.output or (b"", b"")
            exit_code = exec_result.exit_code
            timed_out = False
        except asyncio.TimeoutError:
            stdout_b, stderr_b = b"", b"Execution timed out"
            exit_code = -1
            timed_out = True

        duration_ms = int((time.monotonic() - start) * 1000)
        return ExecutionResult(
            stdout=(stdout_b or b"").decode("utf-8", errors="replace"),
            stderr=(stderr_b or b"").decode("utf-8", errors="replace"),
            exit_code=exit_code,
            timed_out=timed_out,
            duration_ms=duration_ms,
        )

    async def run_tests(
        self,
        test_command: str = "pytest tests/ -v --tb=short",
        timeout_seconds: int = 120,
    ) -> TestResult:
        start = time.monotonic()
        result = await self.run_command(test_command, timeout_seconds)
        duration_ms = int((time.monotonic() - start) * 1000)
        passed, failed, errors = self._parse_pytest_output(
            result.stdout + result.stderr
        )
        return TestResult(
            command=test_command,
            passed=passed,
            failed=failed,
            errors=errors,
            output=result.stdout + result.stderr,
            duration_ms=duration_ms,
            success=(failed == 0 and errors == 0 and result.exit_code == 0),
            no_tests=(result.exit_code in (4, 5) and passed == 0 and failed == 0),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_ready(self) -> None:
        if self._container is None:
            raise RuntimeError("Docker sandbox not initialised — call setup() first")

    @staticmethod
    def _parse_pytest_output(output: str) -> tuple[int, int, int]:
        passed = errors = failed = 0
        for match in re.finditer(r"(\d+)\s+(passed|failed|error)", output):
            count, label = int(match.group(1)), match.group(2)
            if label == "passed":
                passed = count
            elif label == "failed":
                failed = count
            elif label == "error":
                errors = count
        return passed, failed, errors
