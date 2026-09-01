"""Arm B: the fixed pipeline, and the defects the original had."""

from __future__ import annotations

import json
import subprocess

import pytest

from codepilot.agent.pipeline import (
    MAX_DEBUG_ROUNDS,
    PipelineState,
    Runtime,
    after_test,
    run_pipeline,
)
from codepilot.context import Conversation
from codepilot.events import EventStream, EventType
from codepilot.llm import Reply, ToolCall, Usage
from codepilot.permissions import Budget, PermissionGate
from codepilot.tools import ToolContext
from codepilot.workspace import Workspace


class FakeOutcome:
    def __init__(self, success=True, no_tests=False, summary="1 passed", output="1 passed"):
        self.success = success
        self.no_tests = no_tests
        self.summary = summary
        self.output = output
        self.duration_ms = 1


class FakeSandbox:
    """Returns a scripted sequence of test outcomes."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.runs = 0

    async def run(self, command, timeout_seconds=60):
        raise AssertionError("the pipeline should not shell out directly")

    async def run_tests(self, command="pytest -q", timeout_seconds=300):
        outcome = self.outcomes[min(self.runs, len(self.outcomes) - 1)]
        self.runs += 1
        return outcome

    async def close(self):
        return None


class ScriptedClient:
    """Replies by phase, recognised from the system prompt."""

    def __init__(self, steps=("step one", "step two"), verdicts=("APPROVED",)):
        self.steps = list(steps)
        self.verdicts = list(verdicts)
        self.phases: list[str] = []
        self.reviews = 0
        self.model = "stub"

    async def count_tokens(self, messages, **kw):
        return 50

    async def chat(self, messages, system=None, tools=None, **kw):
        text = system[0]["text"] if isinstance(system, list) else str(system)
        if "plan a coding task" in text:
            phase = "plan"
            body = "```json\n" + json.dumps({"steps": self.steps}) + "\n```"
        elif "one step of a plan" in text:
            phase, body = "code", "edited the file"
        elif "failing test run" in text:
            phase, body = "debug", "root cause: an off-by-one"
        else:
            phase = "review"
            verdict = self.verdicts[min(self.reviews, len(self.verdicts) - 1)]
            self.reviews += 1
            body = f"VERDICT: {verdict}\nlooks right"
        self.phases.append(phase)
        return Reply(
            text=body,
            content=[{"type": "text", "text": body}],
            tool_calls=[],
            stop_reason="end_turn",
            model="stub",
            usage=Usage(input_tokens=50, output_tokens=20),
            latency_ms=1,
            cost_usd=0.001,
        )


def runtime(tmp_path, client, sandbox, budget=None):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    stream = EventStream(session_id="t")
    ctx = ToolContext(
        workspace=Workspace(root=tmp_path, session_id="t"),
        sandbox=sandbox,
        permissions=PermissionGate(auto_approve=True),
        events=stream,
    )
    return Runtime(
        client=client,
        ctx=ctx,
        convo=Conversation(system_prompt="unused"),
        budget=budget or Budget(max_usd=1.0, max_turns=200),
        effort=None,
    ), stream


# ----------------------------------------------------------------- happy path


@pytest.mark.asyncio
async def test_a_two_step_plan_runs_both_steps_and_is_approved(tmp_path):
    client = ScriptedClient(steps=["step one", "step two"])
    rt, stream = runtime(tmp_path, client, FakeSandbox([FakeOutcome()]))
    state = await run_pipeline(rt, "do the thing")

    assert state.approved
    assert client.phases.count("code") == 2, (
        f"expected one code phase per step, got {client.phases}"
    )
    assert any(e.type is EventType.PLAN for e in stream.events)


@pytest.mark.asyncio
async def test_a_step_advances_only_when_its_tests_pass(tmp_path):
    """The original advanced on reviewer rejection, so a first-time approval
    ended the run after step one."""
    client = ScriptedClient(steps=["one", "two", "three"])
    rt, _ = runtime(tmp_path, client, FakeSandbox([FakeOutcome()]))
    state = await run_pipeline(rt, "three steps")
    assert state.current_step == 2
    assert client.phases.count("code") == 3


# ------------------------------------------------------------------ debugging


@pytest.mark.asyncio
async def test_failing_tests_route_to_the_debugger(tmp_path):
    client = ScriptedClient(steps=["only step"])
    sandbox = FakeSandbox(
        [FakeOutcome(success=False, summary="1 failed", output="AssertionError"), FakeOutcome()]
    )
    rt, _ = runtime(tmp_path, client, sandbox)
    state = await run_pipeline(rt, "fix it")
    assert "debug" in client.phases
    assert state.approved


@pytest.mark.asyncio
async def test_a_missing_suite_goes_to_the_coder_not_the_debugger(tmp_path):
    """Handing a missing directory to the debugger costs a model call to
    root-cause it inside application code."""
    client = ScriptedClient(steps=["only step"])
    sandbox = FakeSandbox(
        [FakeOutcome(success=False, no_tests=True, output="no tests ran"), FakeOutcome()]
    )
    rt, _ = runtime(tmp_path, client, sandbox)
    await run_pipeline(rt, "write some code")
    assert "debug" not in client.phases, f"phases were {client.phases}"
    assert client.phases.count("code") >= 2


def test_the_missing_suite_edge_is_chosen_directly():
    state = vars(
        PipelineState(
            task="t", steps=["a"], tests_passed=False,
            last_test="No tests were collected. Write the tests this step needs.",
        )
    )
    assert after_test(state) == "code"


def test_a_real_failure_edge_goes_to_debug():
    state = vars(
        PipelineState(task="t", steps=["a"], tests_passed=False, last_test="AssertionError")
    )
    assert after_test(state) == "debug"


# ------------------------------------------------------------------- ceilings


@pytest.mark.asyncio
async def test_endless_failures_stop_at_the_debug_ceiling(tmp_path):
    client = ScriptedClient(steps=["only step"], verdicts=["CHANGES_NEEDED"])
    sandbox = FakeSandbox([FakeOutcome(success=False, output="AssertionError")])
    rt, _ = runtime(tmp_path, client, sandbox)
    state = await run_pipeline(rt, "impossible")
    assert not state.approved
    assert state.stopped_by == "review limit"
    assert client.phases.count("debug") <= MAX_DEBUG_ROUNDS * 2


@pytest.mark.asyncio
async def test_the_budget_stops_the_pipeline(tmp_path):
    client = ScriptedClient(steps=["a", "b", "c"], verdicts=["CHANGES_NEEDED"])
    sandbox = FakeSandbox([FakeOutcome(success=False, output="boom")])
    rt, stream = runtime(
        tmp_path, client, sandbox, budget=Budget(max_usd=0.004, max_turns=500)
    )
    state = await run_pipeline(rt, "expensive")
    assert state.stopped_by == "budget"
    assert any(e.type is EventType.BUDGET for e in stream.events)


@pytest.mark.asyncio
async def test_an_unparseable_plan_falls_back_to_one_step(tmp_path):
    class NoJson(ScriptedClient):
        async def chat(self, messages, system=None, tools=None, **kw):
            text = system[0]["text"] if isinstance(system, list) else str(system)
            if "plan a coding task" in text:
                self.phases.append("plan")
                return Reply(
                    text="I would do a few things.", content=[], tool_calls=[],
                    stop_reason="end_turn", model="stub",
                    usage=Usage(input_tokens=10, output_tokens=5),
                    latency_ms=1, cost_usd=0.0,
                )
            return await super().chat(messages, system=system, tools=tools, **kw)

    client = NoJson()
    rt, stream = runtime(tmp_path, client, FakeSandbox([FakeOutcome()]))
    state = await run_pipeline(rt, "the original task")
    assert state.steps == ["the original task"]
    assert state.approved
    assert any(e.type is EventType.ERROR for e in stream.events)
