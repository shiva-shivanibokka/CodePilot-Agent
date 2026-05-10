"""
SandboxBackend Protocol — the abstraction layer between the agent
and whatever executes the code (Docker, subprocess, E2B).

Every backend implements the same interface so agents never need
to know where code is actually running.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from models.schemas import ExecutionResult, TestResult


@runtime_checkable
class SandboxBackend(Protocol):
    """
    Interface every sandbox backend must satisfy.
    The agent tools call these methods exclusively —
    no direct subprocess or Docker calls anywhere in agent code.
    """

    async def setup(self, workspace_files: dict[str, str]) -> str:
        """
        Initialise the sandbox, write workspace_files into it.
        Returns a sandbox_id string that identifies this session.
        """
        ...

    async def execute(
        self,
        code: str,
        filename: str = "run.py",
        timeout_seconds: int = 30,
    ) -> ExecutionResult:
        """
        Run a Python snippet inside the sandbox.
        Returns stdout, stderr, exit_code, duration.
        """
        ...

    async def run_command(
        self,
        command: str,
        timeout_seconds: int = 60,
    ) -> ExecutionResult:
        """
        Run an arbitrary shell command inside the sandbox
        (e.g. 'pytest tests/ -v', 'pip install requests').
        """
        ...

    async def write_file(self, path: str, content: str) -> None:
        """Write or overwrite a file inside the sandbox workspace."""
        ...

    async def read_file(self, path: str) -> str:
        """Read a file from the sandbox workspace."""
        ...

    async def list_files(self, directory: str = ".") -> list[str]:
        """List all files recursively under directory."""
        ...

    async def teardown(self) -> None:
        """Clean up sandbox resources (stop container, delete temp dir, etc.)."""
        ...

    async def run_tests(
        self,
        test_command: str = "pytest tests/ -v --tb=short",
        timeout_seconds: int = 120,
    ) -> TestResult:
        """
        Run the test suite and parse structured results.
        This wraps run_command with pytest output parsing.
        """
        ...
