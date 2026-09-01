"""
Where commands run.

Split by purpose rather than by "safe" and "unsafe":

* **Local** — the tool. Commands run in your own working directory behind the
  permission gate. Isolating your code from your own machine is theatre; the
  agent is running your tests, at your request, on files you already own.
* **Docker** — the eval harness. It executes model-generated code against
  fixture repositories, which genuinely is untrusted, and it has to be hermetic
  for results to mean anything.

Both satisfy one protocol, and the protocol is used as a type — unlike the old
`SandboxBackend`, which no module imported while the factory returned one
concrete class with a `# type: ignore`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from codepilot.sandbox.pytest_parse import TestOutcome


@dataclass
class CommandResult:
    command: str
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int
    timed_out: bool = False

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    @property
    def combined(self) -> str:
        return (self.stdout + ("\n" + self.stderr if self.stderr else "")).strip()


@runtime_checkable
class Sandbox(Protocol):
    """Everything the tools need in order to run something."""

    async def run(self, command: str, timeout_seconds: int = 60) -> CommandResult: ...

    async def run_tests(
        self, command: str = "pytest -q", timeout_seconds: int = 300
    ) -> TestOutcome: ...

    async def close(self) -> None: ...
