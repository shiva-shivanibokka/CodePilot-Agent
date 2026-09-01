"""
SubprocessSandbox — executes code in a temporary directory on the host machine
using Python subprocess. No Docker required. Used for local development and demo.

Security: this does NOT isolate the host. Use DockerSandbox for any untrusted code.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from models.schemas import ExecutionResult, TestResult


class SubprocessSandbox:
    """
    Runs code in a managed temp directory via asyncio subprocess.
    Implements the SandboxBackend protocol.
    """

    def __init__(self) -> None:
        self.sandbox_id: str = ""
        self._workdir: Path | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def setup(self, workspace_files: dict[str, str]) -> str:
        self.sandbox_id = str(uuid.uuid4())[:12]
        self._workdir = Path(tempfile.mkdtemp(prefix=f"codepilot_{self.sandbox_id}_"))

        # Write all workspace files into the temp directory
        for rel_path, content in workspace_files.items():
            dest = self._workdir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

        return self.sandbox_id

    async def teardown(self) -> None:
        if self._workdir and self._workdir.exists():
            shutil.rmtree(self._workdir, ignore_errors=True)
        self._workdir = None

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    async def write_file(self, path: str, content: str) -> None:
        self._assert_ready()
        dest = self._wd / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")

    async def read_file(self, path: str) -> str:
        self._assert_ready()
        target = self._wd / path
        if not target.exists():
            raise FileNotFoundError(f"File not found in sandbox: {path}")
        return target.read_text(encoding="utf-8")

    async def list_files(self, directory: str = ".") -> list[str]:
        self._assert_ready()
        base = self._wd / directory
        if not base.exists():
            return []
        wd = self._wd
        return [
            str(p.relative_to(wd))
            for p in base.rglob("*")
            if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts
        ]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(
        self,
        code: str,
        filename: str = "run.py",
        timeout_seconds: int = 30,
    ) -> ExecutionResult:
        self._assert_ready()
        script = self._wd / filename
        script.write_text(code, encoding="utf-8")
        return await self.run_command(f"python {filename}", timeout_seconds)

    async def run_command(
        self,
        command: str,
        timeout_seconds: int = 60,
    ) -> ExecutionResult:
        self._assert_ready()
        start = time.monotonic()
        timed_out = False

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._workdir),
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                timed_out = True
                stdout_b, stderr_b = b"", b"Execution timed out"

            duration_ms = int((time.monotonic() - start) * 1000)
            return ExecutionResult(
                stdout=stdout_b.decode("utf-8", errors="replace"),
                stderr=stderr_b.decode("utf-8", errors="replace"),
                exit_code=(proc.returncode if proc.returncode is not None else -1)
                if not timed_out
                else -1,
                timed_out=timed_out,
                duration_ms=duration_ms,
            )

        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            return ExecutionResult(
                stdout="",
                stderr=str(exc),
                exit_code=-1,
                timed_out=False,
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
        if self._workdir is None:
            raise RuntimeError("Sandbox not initialised — call setup() first")

    @property
    def _wd(self) -> Path:
        """Narrowed non-None workdir path for use after _assert_ready()."""
        if self._workdir is None:
            raise RuntimeError("Sandbox not initialised — call setup() first")
        return self._workdir

    @staticmethod
    def _parse_pytest_output(output: str) -> tuple[int, int, int]:
        """Parse pytest summary line: '3 passed, 1 failed, 0 errors'."""
        passed = errors = failed = 0
        # e.g. "3 passed" / "1 failed" / "2 errors"
        for match in re.finditer(r"(\d+)\s+(passed|failed|error)", output):
            count, label = int(match.group(1)), match.group(2)
            if label == "passed":
                passed = count
            elif label == "failed":
                failed = count
            elif label == "error":
                errors = count
        return passed, failed, errors
