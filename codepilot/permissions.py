"""
The single choke point for "should this be allowed, and can we still afford it".

Both agent arms route every tool call through here, so a safety rule written
once applies to both. Splitting these checks across the arms is how one arm ends
up quietly less safe than the other.

Two independent gates:

* **Permission** — is this command allowed to run? Read-only and test commands
  pass silently; anything else asks. There is no allowlist that is safe against
  a determined shell string, so the goal is to stop the ordinary accident, not
  an adversary: the agent is running your code on your machine at your request.
* **Budget** — tokens, dollars and turns. Checked before each call, so a runaway
  loop stops at a number you chose instead of one you discover on an invoice.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum


class Decision(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


#: Commands that only read, or that run the project's tests. Matched on the
#: executable name after shell parsing, not by substring.
DEFAULT_ALLOWED = {
    "ls", "dir", "cat", "head", "tail", "wc", "find", "grep", "rg", "which",
    "pwd", "echo", "tree", "file", "stat", "diff",
    "pytest", "python", "python3", "py", "tox", "nox", "coverage",
    "node", "npm", "npx", "yarn", "pnpm", "go", "cargo", "make",
}

#: Always refused. Not a security boundary — a guard against the obvious
#: accident, since a shell string has unlimited ways to say the same thing.
DEFAULT_DENIED = [
    re.compile(r"\brm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)*/(?:\s|$)"),  # rm -rf /
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s+.*\bof=/dev/"),
    re.compile(r":\(\)\s*\{.*\|.*&\s*\}\s*;"),  # fork bomb
    re.compile(r"\bgit\s+push\b.*--force"),
    re.compile(r"\bshutdown\b|\breboot\b"),
    re.compile(r"\bcurl\b.*\|\s*(ba)?sh\b"),  # pipe-to-shell
]


class BudgetExceeded(RuntimeError):
    """Raised when a ceiling is hit. Ends the turn; never silently ignored."""


@dataclass
class Budget:
    max_usd: float = 1.00
    max_turns: int = 40
    max_tokens: int = 400_000

    spent_usd: float = 0.0
    turns: int = 0
    tokens: int = 0
    #: Calls whose model had no published price. Their real cost is unknown, so
    #: spent_usd is a lower bound and saying so is part of the report.
    unpriced_calls: int = 0

    def record(self, cost_usd: float | None, tokens: int) -> None:
        if cost_usd is None:
            self.unpriced_calls += 1
        else:
            self.spent_usd += cost_usd
        self.tokens += tokens
        self.turns += 1

    def check(self) -> None:
        if self.spent_usd >= self.max_usd:
            raise BudgetExceeded(
                f"Cost ceiling reached: ${self.spent_usd:.4f} of ${self.max_usd:.2f}."
            )
        if self.turns >= self.max_turns:
            raise BudgetExceeded(
                f"Turn ceiling reached: {self.turns} of {self.max_turns}."
            )
        if self.tokens >= self.max_tokens:
            raise BudgetExceeded(
                f"Token ceiling reached: {self.tokens:,} of {self.max_tokens:,}."
            )

    def summary(self) -> str:
        note = (
            f" (+{self.unpriced_calls} unpriced call(s), so this is a lower bound)"
            if self.unpriced_calls
            else ""
        )
        return (
            f"${self.spent_usd:.4f} / ${self.max_usd:.2f}{note} · "
            f"{self.tokens:,} tokens · {self.turns} calls"
        )


@dataclass
class PermissionGate:
    allowed: set[str] = field(default_factory=lambda: set(DEFAULT_ALLOWED))
    denied: list[re.Pattern[str]] = field(default_factory=lambda: list(DEFAULT_DENIED))
    #: Skip every prompt. Used by the eval runner, never interactively.
    auto_approve: bool = False
    #: Asked when a decision is ASK. Returns True to allow. Without one, ASK
    #: becomes DENY — a non-interactive run must not block forever on a prompt.
    prompt: Callable[[str, str], bool] | None = None
    #: Test files are protected during debugging: the classic agent shortcut is
    #: to edit the test until it passes.
    protected_globs: list[str] = field(
        default_factory=lambda: ["test_*.py", "*_test.py", "tests/*", "conftest.py"]
    )

    def classify(self, command: str) -> tuple[Decision, str]:
        for pattern in self.denied:
            if pattern.search(command):
                return Decision.DENY, f"matches a refused pattern: {pattern.pattern}"
        if self.auto_approve:
            return Decision.ALLOW, "auto-approved"
        try:
            parts = shlex.split(command, posix=False)
        except ValueError:
            return Decision.ASK, "the command could not be parsed as a shell string"
        if not parts:
            return Decision.DENY, "empty command"

        # Chained commands are only as safe as their least safe segment.
        if any(sep in command for sep in ("&&", "||", ";", "|", "`", "$(")):
            return Decision.ASK, "chains or substitutes commands"

        executable = parts[0].rsplit("/", 1)[-1].rsplit("\\", 1)[-1].removesuffix(".exe")
        if executable in self.allowed:
            return Decision.ALLOW, f"{executable} is on the allowlist"
        return Decision.ASK, f"{executable} is not on the allowlist"

    def check_command(self, command: str) -> tuple[bool, str]:
        decision, reason = self.classify(command)
        if decision is Decision.ALLOW:
            return True, reason
        if decision is Decision.DENY:
            return False, f"refused — {reason}"
        if self.prompt is None:
            return False, f"needs approval ({reason}) but this session cannot ask"
        return self.prompt(command, reason), f"asked — {reason}"

    def is_protected(self, path: str) -> bool:
        from fnmatch import fnmatch
        from pathlib import PurePosixPath

        p = PurePosixPath(path.replace("\\", "/"))
        return any(
            fnmatch(p.as_posix(), g) or fnmatch(p.name, g) for g in self.protected_globs
        )

    def check_edit(self, path: str, *, debugging: bool = False) -> tuple[bool, str]:
        """Editing a test while debugging needs a reason.

        Making the failing test pass by changing the test is the single most
        common way an agent reports success without having fixed anything.
        """
        if not (debugging and self.is_protected(path)):
            return True, "ok"
        if self.auto_approve:
            return True, "auto-approved"
        if self.prompt is None:
            return False, (
                f"{path} is a test file and this is a debugging turn. Fix the code "
                "under test, not the test."
            )
        return (
            self.prompt(f"edit {path}", "editing a test file while debugging"),
            "asked",
        )
