"""
One pytest parser, shared by every backend.

It used to be duplicated byte-for-byte in the subprocess and docker sandboxes,
which meant a parser fix landed in one of them and the two backends could
silently disagree about whether a run passed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: pytest's own exit codes. 1 is "tests failed", which is a real result;
#: 4 and 5 mean there was nothing to run, which is not.
EXIT_TESTS_FAILED = 1
EXIT_USAGE_ERROR = 4  # e.g. `pytest tests/` where tests/ does not exist
EXIT_NO_TESTS_COLLECTED = 5

#: Decorated summary, as printed by default:
#:   "===== 1 failed, 3 passed, 2 warnings in 0.42s ====="
_DECORATED = re.compile(r"^=+\s(.*?)\s=+$")
#: Quiet summary, as printed by `-q`, with no decoration at all:
#:   "1 failed, 3 passed in 0.42s"
#: Anchoring only on the decorated form silently reports 0 passed for every
#: `-q` run — which is the default command this agent uses.
_UNDECORATED = re.compile(r"^[^=].*\bin\s+\d+(?:\.\d+)?\s*s(?:econds)?\s*$")
_COUNT = re.compile(r"(\d+)\s+(passed|failed|error|errors|skipped|xfailed|xpassed)")


@dataclass
class TestOutcome:
    command: str
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    exit_code: int = 0
    duration_ms: int = 0
    output: str = ""
    timed_out: bool = False

    @property
    def no_tests(self) -> bool:
        """Nothing was collected — a missing suite, not a failing one.

        Routing this to a debugger asks a model to root-cause a missing
        directory inside application code, which it will attempt, expensively.
        """
        return (
            self.exit_code in (EXIT_USAGE_ERROR, EXIT_NO_TESTS_COLLECTED)
            and self.passed == 0
            and self.failed == 0
        )

    @property
    def success(self) -> bool:
        return (
            self.exit_code == 0
            and not self.timed_out
            and self.failed == 0
            and self.errors == 0
        )

    @property
    def summary(self) -> str:
        if self.timed_out:
            return "timed out"
        if self.no_tests:
            return "no tests collected"
        bits = [f"{self.passed} passed"]
        if self.failed:
            bits.append(f"{self.failed} failed")
        if self.errors:
            bits.append(f"{self.errors} errors")
        if self.skipped:
            bits.append(f"{self.skipped} skipped")
        return ", ".join(bits)


def parse(output: str, exit_code: int, command: str = "pytest", duration_ms: int = 0) -> TestOutcome:
    """Read pytest's summary line, not any line that happens to say "passed".

    Anchoring on the `=== ... ===` summary matters: a traceback that prints the
    string "3 passed" from a fixture would otherwise beat the real totals, since
    a scan of the whole output takes the last match it finds.
    """
    outcome = TestOutcome(
        command=command, exit_code=exit_code, duration_ms=duration_ms, output=output
    )

    # Scan upward for the last line that is *shaped* like a summary and carries
    # counts. Shape matters: a traceback can print "9 passed" as an assertion
    # message, and a scan for counts alone would take that over pytest's totals.
    for raw in reversed(output.splitlines()):
        line = raw.rstrip()
        decorated = _DECORATED.match(line)
        candidate = decorated.group(1) if decorated else (line if _UNDECORATED.match(line) else None)
        if candidate is None:
            continue
        counts = _COUNT.findall(candidate)
        if not counts:
            continue  # e.g. "=== FAILURES ===" or "no tests ran in 0.01s"
        for value, label in counts:
            n = int(value)
            if label == "passed":
                outcome.passed = n
            elif label == "failed":
                outcome.failed = n
            elif label in ("error", "errors"):
                outcome.errors = n
            elif label == "skipped":
                outcome.skipped = n
        break

    return outcome
