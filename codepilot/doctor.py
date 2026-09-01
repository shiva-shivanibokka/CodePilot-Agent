"""
`codepilot doctor` — does this install actually work, end to end?

Unit tests prove each part behaves. This proves the parts are connected: that
the workspace can really checkpoint through git, that the sandbox can really
start a process on this OS, that every tool schema is one the API will accept,
and — with --live — that the configured models exist and the cache engages.

Run it after every change to the wiring. Most of it needs no API key; the
checks that spend money are opt-in and cost well under a cent.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Result:
    name: str
    ok: bool
    detail: str = ""
    skipped: bool = False


@dataclass
class Doctor:
    live: bool = False
    results: list[Result] = field(default_factory=list)

    def record(self, name: str, ok: bool, detail: str = "", skipped: bool = False) -> None:
        self.results.append(Result(name, ok, detail, skipped))
        mark = "SKIP" if skipped else ("PASS" if ok else "FAIL")
        print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))

    async def check(self, name: str, fn: Callable[[], Awaitable[str] | str]) -> None:
        try:
            detail = fn()
            if asyncio.iscoroutine(detail):
                detail = await detail
            self.record(name, True, str(detail or ""))
        except Exception as exc:  # noqa: BLE001 - a doctor reports, never crashes
            line = traceback.extract_tb(exc.__traceback__)[-1]
            self.record(
                name, False, f"{type(exc).__name__}: {exc} ({line.filename.split(os.sep)[-1]}:{line.lineno})"
            )

    @property
    def failures(self) -> list[Result]:
        return [r for r in self.results if not r.ok and not r.skipped]


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


async def _imports() -> str:
    import codepilot.context  # noqa: F401
    import codepilot.events  # noqa: F401
    import codepilot.llm  # noqa: F401
    import codepilot.permissions  # noqa: F401
    import codepilot.tools  # noqa: F401
    import codepilot.workspace  # noqa: F401

    return "all core modules import"


async def _git_present() -> str:
    if not shutil.which("git"):
        raise RuntimeError("git is not on PATH — checkpoints and undo cannot work")
    version = subprocess.run(
        ["git", "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return version


async def _workspace_round_trip() -> str:
    from codepilot.workspace import Workspace

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "d@example.com"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "doctor"], cwd=root, check=True)
        (root / "a.py").write_text("value = 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-qm", "init"], cwd=root, check=True)

        ws = Workspace(root=root, session_id="doctor")
        ws.checkpoint("before")
        ws.read("a.py")
        ws.write("a.py", "value = 2\n")
        ws.write("new.py", "junk = 1\n")
        ws.undo()

        if (root / "a.py").read_text() != "value = 1\n":
            raise RuntimeError("undo did not restore the edited file")
        if (root / "new.py").exists():
            raise RuntimeError("undo left a file the agent created")
    return "checkpoint -> edit -> undo restores and cleans up"


async def _containment() -> str:
    from codepilot.workspace import Workspace, WorkspaceError

    with tempfile.TemporaryDirectory() as tmp:
        ws = Workspace(root=Path(tmp))
        for bad in ("../escape.py", "/etc/passwd", "a/../../b.py"):
            try:
                ws.resolve(bad)
            except WorkspaceError:
                continue
            raise RuntimeError(f"{bad!r} was not refused")
    return "paths escaping the root are refused"


async def _sandbox_runs() -> str:
    from codepilot.sandbox.local import LocalSandbox

    with tempfile.TemporaryDirectory() as tmp:
        sb = LocalSandbox(root=Path(tmp))
        result = await sb.run("python -c \"print('hello from the sandbox')\"")
        if not result.success or "hello from the sandbox" not in result.stdout:
            raise RuntimeError(
                f"exit={result.exit_code} stdout={result.stdout!r} stderr={result.stderr!r}"
            )
    return "a process starts and its output comes back"


async def _sandbox_timeout() -> str:
    from codepilot.sandbox.local import LocalSandbox

    with tempfile.TemporaryDirectory() as tmp:
        sb = LocalSandbox(root=Path(tmp))
        result = await sb.run("python -c \"import time; time.sleep(30)\"", timeout_seconds=3)
        if not result.timed_out:
            raise RuntimeError("a 30s command was not stopped by a 3s timeout")
    return "a runaway command is actually killed"


async def _sandbox_tests() -> str:
    from codepilot.sandbox.local import LocalSandbox

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "test_x.py").write_text("def test_a():\n    assert True\n", encoding="utf-8")
        sb = LocalSandbox(root=root)
        good = await sb.run_tests("pytest -q")
        if good.passed != 1 or not good.success:
            raise RuntimeError(f"expected 1 passed, got {good.summary} (exit {good.exit_code})")
        missing = await sb.run_tests("pytest does_not_exist/ -q")
        if not missing.no_tests:
            raise RuntimeError("a missing suite was not reported as no_tests")
    return "pytest parses, and a missing suite is distinguished from a failure"


async def _tool_schemas() -> str:
    from codepilot.tools import REGISTRY, schemas

    specs = schemas()
    if not specs:
        raise RuntimeError("no tools are registered")
    for spec in specs:
        props = spec["input_schema"]["properties"]
        for leaked in ("sandbox", "workspace", "ctx", "permissions"):
            if leaked in props:
                raise RuntimeError(f"{spec['name']} exposes {leaked!r} as a model-facing argument")
        for name in spec["input_schema"]["required"]:
            if name not in props:
                raise RuntimeError(f"{spec['name']} requires undeclared {name!r}")
        if not spec["description"].strip():
            raise RuntimeError(f"{spec['name']} has no description")
    if schemas() != specs:
        raise RuntimeError("tool order is unstable, which invalidates the prompt cache")
    return f"{len(REGISTRY)} tools: {', '.join(sorted(REGISTRY))}"


async def _tool_execution() -> str:
    from codepilot.events import EventStream
    from codepilot.llm import ToolCall
    from codepilot.permissions import PermissionGate
    from codepilot.sandbox.local import LocalSandbox
    from codepilot.tools import ToolContext, execute
    from codepilot.workspace import Workspace

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "m.py").write_text("x = 1\n", encoding="utf-8")
        ctx = ToolContext(
            workspace=Workspace(root=root),
            sandbox=LocalSandbox(root=root),
            permissions=PermissionGate(auto_approve=True),
            events=EventStream(session_id="doctor"),
        )
        await execute(ctx, ToolCall(id="1", name="read_file", arguments={"path": "m.py"}))
        edit = await execute(
            ctx,
            ToolCall(id="2", name="edit_file", arguments={"path": "m.py", "old": "x = 1", "new": "x = 42"}),
        )
        if edit.get("is_error"):
            raise RuntimeError(f"edit failed: {edit['content'][:200]}")
        if (root / "m.py").read_text() != "x = 42\n":
            raise RuntimeError("the edit did not reach disk")

        # Every failure mode must come back as a result, never an exception.
        for name, args in (
            ("no_such_tool", {}),
            ("read_file", {"bogus": 1}),
            ("read_file", {"path": "../../escape"}),
        ):
            r = await execute(ctx, ToolCall(id="3", name=name, arguments=args))
            if not r.get("is_error"):
                raise RuntimeError(f"{name}{args} should have produced an error result")
        if not ctx.events.events:
            raise RuntimeError("no events were emitted for any tool call")
    return "read -> edit reaches disk; every failure returns a result"


async def _events_round_trip() -> str:
    from codepilot.events import EventStream, EventType

    seen: list[str] = []
    stream = EventStream(session_id="doctor")
    stream.subscribe(lambda e: seen.append(e.message))
    stream.subscribe(lambda e: (_ for _ in ()).throw(RuntimeError("bad subscriber")))
    stream.emit(EventType.TOOL_CALL, "one")
    stream.emit(EventType.DONE, "two")
    if seen != ["one", "two"]:
        raise RuntimeError("a broken subscriber stopped a working one")
    restored = EventStream.from_jsonl(stream.to_jsonl())
    if [e.message for e in restored.events] != ["one", "two"]:
        raise RuntimeError("JSONL round trip lost events")
    return "subscribers isolated; JSONL round trip is lossless"


async def _context_prefix() -> str:
    from codepilot.context import Conversation

    convo = Conversation(system_prompt="S" * 100, project_instructions="P" * 100)
    blocks = convo.system_blocks()
    if "cache_control" in blocks[0] or "cache_control" not in blocks[-1]:
        raise RuntimeError("the cache breakpoint is not on the last stable block")
    convo.user("something volatile")
    if convo.system_blocks() != blocks:
        raise RuntimeError("the cached prefix changed between turns")
    return "cache breakpoint placed once, prefix stable across turns"


async def _loop_terminates() -> str:
    """A loop that never calls finish must still stop, and stop cheaply."""
    import tempfile as _tf

    from codepilot.agent.loop import AgentLoop, new_conversation
    from codepilot.events import EventStream
    from codepilot.llm import Reply, ToolCall, Usage
    from codepilot.permissions import Budget, PermissionGate
    from codepilot.sandbox.local import LocalSandbox
    from codepilot.tools import ToolContext
    from codepilot.workspace import Workspace

    class Endless:
        model = "stub"
        calls = 0

        async def chat(self, messages, **kw):
            Endless.calls += 1
            return Reply(
                text="", content=[], stop_reason="tool_use",
                tool_calls=[ToolCall(id="t", name="list_files", arguments={})],
                model="stub", usage=Usage(input_tokens=10, output_tokens=10),
                latency_ms=1, cost_usd=0.01,
            )

        async def count_tokens(self, messages, **kw):
            return 10

    with _tf.TemporaryDirectory() as tmp:
        root = Path(tmp)
        stream = EventStream(session_id="doctor")
        ctx = ToolContext(
            workspace=Workspace(root=root),
            sandbox=LocalSandbox(root=root),
            permissions=PermissionGate(auto_approve=True),
            events=stream,
        )
        budget = Budget(max_usd=0.05, max_turns=1000)
        loop = AgentLoop(Endless(), ctx, new_conversation(), budget, effort=None)
        result = await loop.run("never finish")
    if result.stopped_by != "budget":
        raise RuntimeError(f"stopped by {result.stopped_by!r}, expected the budget")
    return f"an endless loop stopped at the ceiling after {result.steps} calls"


async def _session_round_trip() -> str:
    """A resumed conversation must contain messages, not their reprs."""
    import tempfile as _tf

    from pydantic import BaseModel

    from codepilot.context import Conversation
    from codepilot.session import SessionStore

    class Block(BaseModel):
        type: str = "text"
        text: str = "remembered"

    with _tf.TemporaryDirectory() as tmp:
        store = SessionStore.create(Path(tmp))
        convo = Conversation(system_prompt="S")
        convo.user("a task")
        convo.assistant([Block()])
        store.record_messages(convo.messages)

        restored = SessionStore.create(Path(tmp), store.session_id).messages()
        if len(restored) != 2:
            raise RuntimeError(f"expected 2 messages, got {len(restored)}")
        block = restored[1]["content"][0]
        if not isinstance(block, dict) or block.get("text") != "remembered":
            raise RuntimeError(f"content block did not survive: {block!r}")
    return "messages and content blocks survive save -> load"


async def _api_key() -> str:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set (put it in .env)")
    return f"present ({key[:7]}…, {len(key)} chars)"


async def _models_live() -> str:
    from codepilot.llm import FAST_MODEL, STRONG_MODEL, LLMClient

    client = LLMClient()
    await client.validate(STRONG_MODEL, FAST_MODEL)
    return f"{STRONG_MODEL} and {FAST_MODEL} are both available"


async def _tool_call_live() -> str:
    from codepilot.llm import FAST_MODEL, LLMClient
    from codepilot.tools import schemas

    client = LLMClient(model=FAST_MODEL)
    reply = await client.chat(
        [{"role": "user", "content": "List the files in this repository. Use a tool."}],
        system="You are a coding agent. Always use tools rather than guessing.",
        tools=schemas(["list_files", "read_file"]),
        max_tokens=512,
    )
    if not reply.wants_tools:
        raise RuntimeError(f"the model did not call a tool (stop_reason={reply.stop_reason})")
    names = [t.name for t in reply.tool_calls]
    cost = f"${reply.cost_usd:.5f}" if reply.cost_usd is not None else "unpriced"
    return f"model called {names} — {cost}"


# ---------------------------------------------------------------------------


async def run(live: bool = False) -> int:
    doctor = Doctor(live=live)

    print("\ncodepilot doctor\n" + "=" * 60)
    print("\nCore")
    await doctor.check("modules import", _imports)
    await doctor.check("events stream", _events_round_trip)
    await doctor.check("context prefix", _context_prefix)

    print("\nWorkspace")
    await doctor.check("git available", _git_present)
    await doctor.check("checkpoint and undo", _workspace_round_trip)
    await doctor.check("path containment", _containment)

    print("\nSandbox")
    await doctor.check("runs a process", _sandbox_runs)
    await doctor.check("enforces a timeout", _sandbox_timeout)
    await doctor.check("runs and parses pytest", _sandbox_tests)

    print("\nTools")
    await doctor.check("schemas are model-safe", _tool_schemas)
    await doctor.check("execution and error handling", _tool_execution)

    print("\nAgent")
    await doctor.check("loop stops at the budget", _loop_terminates)
    await doctor.check("session save and resume", _session_round_trip)

    print("\nAPI" + ("" if live else "  (--live to include billed checks)"))
    await doctor.check("api key", _api_key)
    if live:
        await doctor.check("models available", _models_live)
        await doctor.check("live tool call", _tool_call_live)
    else:
        doctor.record("models available", True, "not checked", skipped=True)
        doctor.record("live tool call", True, "not checked", skipped=True)

    failures = doctor.failures
    total = len([r for r in doctor.results if not r.skipped])
    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} of {total} checks FAILED:")
        for r in failures:
            print(f"  - {r.name}: {r.detail}")
        return 1
    print(f"all {total} checks passed")
    return 0


def main() -> int:
    return asyncio.run(run(live="--live" in sys.argv))


if __name__ == "__main__":
    raise SystemExit(main())
