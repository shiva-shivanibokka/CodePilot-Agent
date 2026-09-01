"""Arm A: termination, budgets, interrupts, and recovery from tool errors."""

from __future__ import annotations

import subprocess

import pytest

from codepilot.agent.interrupt import InterruptChannel
from codepilot.agent.loop import MAX_STEPS, AgentLoop, new_conversation
from codepilot.events import EventStream, EventType
from codepilot.llm import Reply, ToolCall, Usage
from codepilot.permissions import Budget, PermissionGate
from codepilot.sandbox.local import LocalSandbox
from codepilot.tools import ToolContext
from codepilot.workspace import Workspace


class StubClient:
    """Replays a scripted sequence of replies. No network."""

    def __init__(self, script: list[Reply]) -> None:
        self.script = list(script)
        self.calls = 0
        self.model = "stub"

    async def chat(self, messages, **kw) -> Reply:
        reply = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return reply

    async def count_tokens(self, messages, **kw) -> int:
        return 100


def tool_reply(*calls: tuple[str, dict], text: str = "") -> Reply:
    return Reply(
        text=text,
        content=[{"type": "text", "text": text}] if text else [],
        tool_calls=[
            ToolCall(id=f"tu_{i}", name=name, arguments=args)
            for i, (name, args) in enumerate(calls)
        ],
        stop_reason="tool_use",
        model="stub",
        usage=Usage(input_tokens=100, output_tokens=50),
        latency_ms=10,
        cost_usd=0.001,
    )


def text_reply(text: str, stop: str = "end_turn") -> Reply:
    return Reply(
        text=text,
        content=[{"type": "text", "text": text}],
        tool_calls=[],
        stop_reason=stop,
        model="stub",
        usage=Usage(input_tokens=100, output_tokens=50),
        latency_ms=10,
        cost_usd=0.001,
    )


@pytest.fixture
def harness(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "fib.py").write_text("def fib(n):\n    return n\n", encoding="utf-8")
    stream = EventStream(session_id="t")
    ctx = ToolContext(
        workspace=Workspace(root=tmp_path, session_id="t"),
        sandbox=LocalSandbox(root=tmp_path),
        permissions=PermissionGate(auto_approve=True),
        events=stream,
    )
    return ctx, stream, tmp_path


def build(client, ctx, budget=None, interrupts=None):
    return AgentLoop(
        client,
        ctx,
        new_conversation(),
        budget or Budget(max_usd=1.0, max_turns=100),
        interrupts=interrupts,
        effort=None,
    )


# ------------------------------------------------------------------ finishing


@pytest.mark.asyncio
async def test_a_task_that_calls_finish_completes(harness):
    ctx, stream, _ = harness
    client = StubClient([tool_reply(("finish", {"summary": "done: nothing needed"}))])
    result = await build(client, ctx).run("do nothing")
    assert result.finished
    assert result.summary == "done: nothing needed"
    assert result.stopped_by == "finished"
    assert stream.events[-1].type is EventType.TURN_END


@pytest.mark.asyncio
async def test_a_full_read_edit_test_finish_sequence(harness):
    ctx, stream, root = harness
    client = StubClient(
        [
            tool_reply(("read_file", {"path": "fib.py"})),
            tool_reply(
                ("edit_file", {"path": "fib.py", "old": "return n", "new": "return n + 1"})
            ),
            tool_reply(("run_tests", {"command": "pytest -q"})),
            tool_reply(("finish", {"summary": "edited fib.py"})),
        ]
    )
    result = await build(client, ctx).run("bump fib")
    assert result.finished
    assert result.edited == ["fib.py"]
    assert "return n + 1" in (root / "fib.py").read_text()
    assert any(e.type is EventType.DIFF for e in stream.events)


@pytest.mark.asyncio
async def test_answering_without_calling_finish_still_terminates(harness):
    """The model saying its piece is an ending, not a reason to prod it again."""
    ctx, _, _ = harness
    client = StubClient([text_reply("The answer is 42.")])
    result = await build(client, ctx).run("a question")
    assert not result.finished
    assert result.stopped_by == "ended without finish"
    assert "42" in result.summary


# -------------------------------------------------------------------- ceilings


@pytest.mark.asyncio
async def test_the_cost_ceiling_stops_a_runaway_loop(harness):
    """A model that never finishes must be stopped by the budget, not by luck."""
    ctx, stream, _ = harness
    client = StubClient([tool_reply(("list_files", {}))])  # repeats forever
    budget = Budget(max_usd=0.005, max_turns=1000)
    result = await build(client, ctx, budget=budget).run("loop forever")
    assert result.stopped_by == "budget"
    assert client.calls < 20, f"kept calling {client.calls} times past the ceiling"
    assert any(e.type is EventType.BUDGET for e in stream.events)


@pytest.mark.asyncio
async def test_the_step_ceiling_stops_a_cheap_runaway(harness):
    ctx, _, _ = harness
    client = StubClient([tool_reply(("list_files", {}))])
    budget = Budget(max_usd=1000.0, max_turns=100_000, max_tokens=10**9)
    result = await build(client, ctx, budget=budget).run("loop forever cheaply")
    assert result.stopped_by == "step limit"
    assert result.steps == MAX_STEPS


# ------------------------------------------------------------------ interrupts


@pytest.mark.asyncio
async def test_a_stop_request_is_honoured_at_the_next_checkpoint(harness):
    ctx, stream, _ = harness
    interrupts = InterruptChannel()
    interrupts.stop()
    client = StubClient([tool_reply(("list_files", {}))])
    result = await build(client, ctx, interrupts=interrupts).run("something")
    assert result.stopped_by == "interrupted"
    assert client.calls == 0, "stopped before spending on a model call"


@pytest.mark.asyncio
async def test_a_queued_steer_reaches_the_conversation(harness):
    ctx, stream, _ = harness
    interrupts = InterruptChannel()
    interrupts.send("actually, use a generator")
    client = StubClient([tool_reply(("finish", {"summary": "ok"}))])
    loop = build(client, ctx, interrupts=interrupts)
    await loop.run("original task")
    injected = [
        m for m in loop.convo.messages
        if isinstance(m.get("content"), str) and "interruption" in m["content"]
    ]
    assert injected, "the steer never reached the model's context"
    assert "generator" in injected[0]["content"]
    assert any(e.type is EventType.INTERRUPT for e in stream.events)


# ------------------------------------------------------------------- recovery


@pytest.mark.asyncio
async def test_a_failing_tool_does_not_end_the_turn(harness):
    """An error is something the model reads and recovers from."""
    ctx, _, root = harness
    client = StubClient(
        [
            # Edit without reading first — refused.
            tool_reply(("edit_file", {"path": "fib.py", "old": "return n", "new": "return 0"})),
            tool_reply(("read_file", {"path": "fib.py"})),
            tool_reply(("edit_file", {"path": "fib.py", "old": "return n", "new": "return 0"})),
            tool_reply(("finish", {"summary": "recovered"})),
        ]
    )
    result = await build(client, ctx).run("edit it")
    assert result.finished
    assert "return 0" in (root / "fib.py").read_text()


@pytest.mark.asyncio
async def test_a_refusal_stops_cleanly(harness):
    ctx, stream, _ = harness
    client = StubClient([text_reply("", stop="refusal")])
    result = await build(client, ctx).run("something declined")
    assert result.stopped_by == "refused"
    assert any(e.type is EventType.ERROR for e in stream.events)


# ----------------------------------------------------------------- accounting


@pytest.mark.asyncio
async def test_every_model_call_emits_a_cost_event(harness):
    ctx, stream, _ = harness
    client = StubClient(
        [tool_reply(("list_files", {})), tool_reply(("finish", {"summary": "x"}))]
    )
    await build(client, ctx).run("count me")
    costs = [e for e in stream.events if e.type is EventType.COST]
    assert len(costs) == 2
    assert all(e.data["cost_usd"] == 0.001 for e in costs)
