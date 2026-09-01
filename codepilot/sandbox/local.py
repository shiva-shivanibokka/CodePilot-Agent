"""
Runs commands in the real working directory.

The tool's backend. Safety here comes from the permission gate upstream, not
from isolation — the code being run is yours, in your repo, at your request.

Windows matters: this project's author develops on it, so the shell is
resolved rather than assumed, `sys.executable` is preferred to a bare `python`
(which may not exist, or may be the wrong interpreter), and output is decoded
with a replacement policy so a stray byte cannot kill a run.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from codepilot.sandbox.base import CommandResult
from codepilot.sandbox.pytest_parse import TestOutcome, parse


class LocalSandbox:
    """Implements `Sandbox` against the host, in `root`."""

    def __init__(self, root: Path | str, env: dict[str, str] | None = None) -> None:
        self.root = Path(root).resolve()
        self._env = {**os.environ, **(env or {})}
        # Unbuffered and UTF-8: without these, a child process's output arrives
        # in one lump at exit and non-ASCII arrives mangled on a cp1252 console.
        self._env.setdefault("PYTHONUNBUFFERED", "1")
        self._env.setdefault("PYTHONIOENCODING", "utf-8")

    def _normalise(self, command: str) -> str:
        """Point a bare `python`/`pytest` at the interpreter actually running us.

        On Windows `python` frequently resolves to a store stub, and inside a
        virtualenv a bare `pytest` may be a different environment's. Both
        produce failures that look like the agent's fault.
        """
        try:
            parts = shlex.split(command, posix=False)
        except ValueError:
            return command
        if not parts:
            return command
        head = parts[0].strip('"')
        if head in ("python", "python3", "py"):
            parts[0] = f'"{sys.executable}"'
        elif head == "pytest":
            parts[0:1] = [f'"{sys.executable}"', "-m", "pytest"]
        else:
            return command
        return " ".join(parts)

    async def run(self, command: str, timeout_seconds: int = 60) -> CommandResult:
        """Run to completion in a worker thread.

        Deliberately `subprocess.run` in a thread rather than
        `asyncio.create_subprocess_shell`: on Windows the latter raises
        NotImplementedError under a SelectorEventLoop, which is what several
        async test runners install. Depending on the caller's event-loop policy
        to be able to run a command is a fragile way to build a coding agent,
        and nothing here streams command output incrementally anyway.
        """
        started = time.monotonic()
        actual = self._normalise(command)

        def _run() -> tuple[int, bytes, bytes, bool]:
            try:
                proc = subprocess.run(  # noqa: S602 - a shell is the point
                    actual,
                    shell=True,
                    cwd=str(self.root),
                    env=self._env,
                    capture_output=True,
                    timeout=timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                # TimeoutExpired carries whatever was captured before the kill.
                return (
                    -1,
                    exc.stdout or b"",
                    (exc.stderr or b"")
                    + f"\n[timed out after {timeout_seconds}s]".encode(),
                    True,
                )
            except OSError as exc:
                return -1, b"", f"could not start command: {exc}".encode(), False
            return proc.returncode, proc.stdout, proc.stderr, False

        code, out, err, timed_out = await asyncio.to_thread(_run)
        return CommandResult(
            command=command,
            stdout=out.decode("utf-8", errors="replace"),
            stderr=err.decode("utf-8", errors="replace"),
            exit_code=code,
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
        """Nothing to release: no container, no temp directory, no connection."""
        return None
