"""
The tool surface, and the one place tool calls are executed.

Two rules the old registry broke, both structural rather than stylistic:

1. **The sandbox is never a tool parameter.** It arrives in `ToolContext`, via a
   closure. The previous version declared `sandbox: Any` as an argument, which
   would require the model to serialise a live object into JSON — the tools
   could not have worked even if anything had called them.
2. **A failing tool returns a result, never an exception.** The model can read
   an error and recover; it cannot recover from a traceback that ended the turn.
   Everything below `execute()` is caught.

Both agent arms call `execute()`, so the permission gate, the budget and the
event stream apply identically to both. A comparison between the arms is only
meaningful if the thing that differs is the control flow.
"""

from __future__ import annotations

import difflib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from codepilot.events import EventStream, EventType
from codepilot.llm import ToolCall
from codepilot.permissions import PermissionGate
from codepilot.sandbox.base import Sandbox
from codepilot.workspace import Workspace, WorkspaceError

#: Tool output above this is truncated before it reaches the model. A 50k-line
#: test log crowds out the code it is about.
MAX_TOOL_OUTPUT = 12_000


@dataclass
class ToolContext:
    workspace: Workspace
    sandbox: Sandbox
    permissions: PermissionGate
    events: EventStream
    #: True while recovering from a failing test. Gates edits to test files.
    debugging: bool = False
    #: Set by `finish`, read by the loop.
    finished: bool = False
    final_message: str = ""
    #: Set by `propose_plan`, rendered by the CLI.
    plan: list[str] = field(default_factory=list)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    required: list[str]
    run: Callable[..., Awaitable[str]]
    #: Tools that only read can skip the permission prompt entirely.
    read_only: bool = False

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required,
            },
        }


REGISTRY: dict[str, Tool] = {}


def tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    required: list[str],
    read_only: bool = False,
) -> Callable[[Callable[..., Awaitable[str]]], Callable[..., Awaitable[str]]]:
    def decorate(fn: Callable[..., Awaitable[str]]):
        REGISTRY[name] = Tool(name, description, parameters, required, fn, read_only)
        return fn

    return decorate


def schemas(names: list[str] | None = None) -> list[dict[str, Any]]:
    """Tool schemas for the request.

    Order is fixed (sorted), because the tool block sits in front of the cache
    breakpoint: a set that reorders between turns invalidates the whole prefix.
    """
    chosen = sorted(names) if names else sorted(REGISTRY)
    return [REGISTRY[n].schema() for n in chosen]


def _truncate(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    head, tail = text[: limit // 2], text[-limit // 2 :]
    dropped = len(text) - limit
    return f"{head}\n\n[... {dropped:,} characters omitted ...]\n\n{tail}"


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


@tool(
    "read_file",
    "Read a file from the repository. Read a file before editing it — edits to "
    "a file you have not read are refused.",
    {"path": {"type": "string", "description": "Path relative to the repository root"}},
    ["path"],
    read_only=True,
)
async def read_file(ctx: ToolContext, path: str) -> str:
    content = ctx.workspace.read(path)
    numbered = "\n".join(
        f"{i:5d} | {line}" for i, line in enumerate(content.splitlines(), 1)
    )
    return _truncate(f"{path} ({content.count(chr(10)) + 1} lines)\n\n{numbered}")


@tool(
    "edit_file",
    "Replace an exact string in a file. Prefer this over write_file: it costs "
    "tokens proportional to the change rather than to the file, and it cannot "
    "silently drop code you did not reproduce. `old` must match byte for byte, "
    "whitespace included, and must be unique unless replace_all is true.",
    {
        "path": {"type": "string"},
        "old": {"type": "string", "description": "Exact text to replace"},
        "new": {"type": "string", "description": "Replacement text"},
        "replace_all": {"type": "boolean", "description": "Replace every occurrence"},
    },
    ["path", "old", "new"],
)
async def edit_file(
    ctx: ToolContext, path: str, old: str, new: str, replace_all: bool = False
) -> str:
    allowed, reason = ctx.permissions.check_edit(path, debugging=ctx.debugging)
    if not allowed:
        raise PermissionError(reason)
    # Read the bytes directly rather than through workspace.read(): that would
    # record the read and hand this edit the freshness it is supposed to prove.
    target = ctx.workspace.resolve(path)
    before = target.read_text(encoding="utf-8", errors="replace") if target.is_file() else ""
    result = ctx.workspace.edit(path, old, new, replace_all=replace_all)
    after = target.read_text(encoding="utf-8", errors="replace")
    _emit_diff(ctx, path, before, after)
    return result


@tool(
    "write_file",
    "Write a whole file. Use for new files; for changes to an existing file use "
    "edit_file instead.",
    {"path": {"type": "string"}, "content": {"type": "string"}},
    ["path", "content"],
)
async def write_file(ctx: ToolContext, path: str, content: str) -> str:
    allowed, reason = ctx.permissions.check_edit(path, debugging=ctx.debugging)
    if not allowed:
        raise PermissionError(reason)
    target = ctx.workspace.resolve(path)
    before = (
        target.read_text(encoding="utf-8", errors="replace") if target.is_file() else ""
    )
    result = ctx.workspace.write(path, content)
    _emit_diff(ctx, path, before, content)
    return result


def _emit_diff(ctx: ToolContext, path: str, before: str, after: str) -> None:
    diff = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=2,
        )
    )
    added = sum(1 for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
    removed = sum(1 for ln in diff.splitlines() if ln.startswith("-") and not ln.startswith("---"))
    ctx.events.emit(
        EventType.DIFF,
        f"{path}  +{added} -{removed}",
        path=path,
        diff=_truncate(diff, 4000),
        added=added,
        removed=removed,
    )


@tool(
    "list_files",
    "List the repository's files, honouring .gitignore.",
    {"subdirectory": {"type": "string", "description": "Optional path prefix filter"}},
    [],
    read_only=True,
)
async def list_files(ctx: ToolContext, subdirectory: str = "") -> str:
    files = ctx.workspace.list_files()
    if subdirectory:
        prefix = subdirectory.strip("/") + "/"
        files = [f for f in files if f.startswith(prefix)]
    if not files:
        return "No files matched."
    return _truncate(f"{len(files)} files:\n" + "\n".join(files))


@tool(
    "search",
    "Search file contents for a regular expression. Returns path:line: text.",
    {
        "pattern": {"type": "string", "description": "Python regular expression"},
        "glob": {"type": "string", "description": "Filename filter, e.g. *.py"},
    },
    ["pattern"],
    read_only=True,
)
async def search(ctx: ToolContext, pattern: str, glob: str = "*") -> str:
    import re
    from fnmatch import fnmatch

    try:
        regex = re.compile(pattern)
    except re.error as exc:
        # An invalid pattern is the model's mistake to fix, so tell it precisely.
        raise ValueError(f"{pattern!r} is not a valid regular expression: {exc}") from exc

    hits: list[str] = []
    for rel in ctx.workspace.list_files():
        if not fnmatch(rel, glob) and not fnmatch(rel.rsplit("/", 1)[-1], glob):
            continue
        try:
            content = (ctx.workspace.root / rel).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            continue
        for i, line in enumerate(content.splitlines(), 1):
            if regex.search(line):
                hits.append(f"{rel}:{i}: {line.strip()[:200]}")
                if len(hits) >= 200:
                    return _truncate(
                        "\n".join(hits) + "\n[stopped at 200 matches — narrow the pattern]"
                    )
    return _truncate("\n".join(hits)) if hits else f"No matches for {pattern!r}."


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@tool(
    "run_command",
    "Run a shell command in the repository. Commands outside the allowlist need "
    "the user's approval.",
    {
        "command": {"type": "string"},
        "timeout_seconds": {"type": "integer", "description": "Default 60"},
    },
    ["command"],
)
async def run_command(ctx: ToolContext, command: str, timeout_seconds: int = 60) -> str:
    allowed, reason = ctx.permissions.check_command(command)
    ctx.events.emit(
        EventType.PERMISSION_DECISION,
        f"{'allowed' if allowed else 'refused'}: {command}",
        command=command,
        allowed=allowed,
        reason=reason,
    )
    if not allowed:
        raise PermissionError(f"`{command}` was not run: {reason}")
    result = await ctx.sandbox.run(command, timeout_seconds=timeout_seconds)
    return _truncate(
        f"$ {command}\nexit {result.exit_code} in {result.duration_ms}ms\n\n"
        f"{result.combined or '(no output)'}"
    )


@tool(
    "run_tests",
    "Run the test suite and get structured pass/fail counts.",
    {"command": {"type": "string", "description": "Default: pytest -q"}},
    [],
)
async def run_tests(ctx: ToolContext, command: str = "pytest -q") -> str:
    allowed, reason = ctx.permissions.check_command(command)
    if not allowed:
        raise PermissionError(f"`{command}` was not run: {reason}")
    outcome = await ctx.sandbox.run_tests(command)
    ctx.events.emit(
        EventType.TEST_RESULT,
        f"{outcome.summary} ({outcome.duration_ms}ms)",
        passed=outcome.passed,
        failed=outcome.failed,
        errors=outcome.errors,
        no_tests=outcome.no_tests,
        success=outcome.success,
    )
    # Debugging turns protect test files; a passing run lifts that.
    ctx.debugging = not outcome.success and not outcome.no_tests

    if outcome.no_tests:
        return (
            f"$ {command}\nNo tests were collected. This is a missing suite, not a "
            "failing one — write the tests, do not debug the code."
        )
    return _truncate(
        f"$ {command}\n{outcome.summary} (exit {outcome.exit_code})\n\n{outcome.output}"
    )


# ---------------------------------------------------------------------------
# Control
# ---------------------------------------------------------------------------


@tool(
    "propose_plan",
    "State the steps you intend to take, before editing anything. Use for any "
    "task touching more than one file.",
    {"steps": {"type": "array", "items": {"type": "string"}}},
    ["steps"],
    read_only=True,
)
async def propose_plan(ctx: ToolContext, steps: list[str]) -> str:
    ctx.plan = list(steps)
    ctx.events.emit(
        EventType.PLAN,
        "\n".join(f"  {i}. {s}" for i, s in enumerate(steps, 1)),
        steps=steps,
    )
    return f"Plan of {len(steps)} steps recorded. Proceed."


@tool(
    "finish",
    "Call when the task is complete, or when you cannot continue. Summarise what "
    "changed and the state of the tests.",
    {"summary": {"type": "string"}},
    ["summary"],
    read_only=True,
)
async def finish(ctx: ToolContext, summary: str) -> str:
    ctx.finished = True
    ctx.final_message = summary
    return "Acknowledged."


# ---------------------------------------------------------------------------
# Execution of a call
# ---------------------------------------------------------------------------


async def execute(ctx: ToolContext, call: ToolCall) -> dict[str, Any]:
    """Run one tool call and return a `tool_result` block.

    Never raises. Every failure mode — unknown tool, bad arguments, a refused
    permission, a workspace rule, an unexpected bug — comes back as an error
    result the model can read and respond to.
    """
    from codepilot.context import Conversation

    ctx.events.emit(
        EventType.TOOL_CALL,
        f"{call.name}({_brief(call.arguments)})",
        tool=call.name,
        arguments=call.arguments,
    )

    spec = REGISTRY.get(call.name)
    if spec is None:
        return Conversation.tool_result(
            call.id,
            f"No such tool: {call.name!r}. Available: {', '.join(sorted(REGISTRY))}.",
            is_error=True,
        )

    try:
        output = await spec.run(ctx, **call.arguments)
    except TypeError as exc:
        # Wrong or missing arguments — recoverable, and worth being precise about.
        output, is_error = f"Bad arguments for {call.name}: {exc}", True
    except (WorkspaceError, PermissionError, ValueError) as exc:
        output, is_error = str(exc), True
    except Exception as exc:  # noqa: BLE001 - a tool bug must not end the turn
        output, is_error = f"{call.name} failed unexpectedly: {type(exc).__name__}: {exc}", True
    else:
        is_error = False

    ctx.events.emit(
        EventType.TOOL_RESULT,
        _first_line(output),
        tool=call.name,
        is_error=is_error,
        output=_truncate(output, 4000),
    )
    return Conversation.tool_result(call.id, output, is_error=is_error)


def _brief(args: dict[str, Any], limit: int = 90) -> str:
    parts = []
    for k, v in args.items():
        text = str(v).replace("\n", "⏎")
        parts.append(f"{k}={text[:40] + '…' if len(text) > 40 else text}")
    joined = ", ".join(parts)
    return joined[:limit] + ("…" if len(joined) > limit else "")


def _first_line(text: str, limit: int = 120) -> str:
    line = text.strip().splitlines()[0] if text.strip() else "(empty)"
    return line[:limit] + ("…" if len(line) > limit else "")
