"""
Graph-level flow tests.

No API key, no network: ModelRouter.chat is replaced by StubRouter, which returns
canned replies per agent. These tests exercise the real compiled LangGraph, which
is where every critical defect in AUDIT.md lives.
"""

from __future__ import annotations

import json

import pytest

from agent.graph import build_graph
from models.schemas import (
    AgentName,
    AgentState,
    LLMCall,
    StepStatus,
    TaskComplexity,
    TestResult,
)


class StubRouter:
    """Canned replies per agent. Records the call sequence for assertions."""

    def __init__(self, reviewer_verdicts: tuple[str, ...] = ("APPROVED",)) -> None:
        self.calls: list[str] = []
        self._verdicts = list(reviewer_verdicts)

    async def chat(
        self,
        agent,
        messages,
        system,
        complexity=TaskComplexity.SIMPLE,
        tools=None,
        max_tokens=4096,
    ):
        self.calls.append(agent.value)
        if len(self.calls) > 200:
            raise AssertionError(
                f"runaway loop: {len(self.calls)} LLM calls, tail={self.calls[-6:]}"
            )
        call = LLMCall(
            agent=agent, model="stub", input_tokens=1, output_tokens=1, latency_ms=1
        )

        if agent is AgentName.ORCHESTRATOR:
            return '{"complexity": "complex", "reason": "stub"}', call

        if agent is AgentName.PLANNER:
            plan = {
                "task": "t",
                "complexity": "complex",
                "total_steps": 3,
                "steps": [
                    {
                        "index": i,
                        "description": f"step {i}",
                        "files_to_read": [],
                        "files_to_write": [f"s{i}.py"],
                        "test_command": "true",
                    }
                    for i in range(3)
                ],
            }
            return "```json\n" + json.dumps(plan) + "\n```", call

        if agent is AgentName.CODER:
            n = self.calls.count("coder")
            return f"WRITE_FILE: f{n}.py\n```python\nx = {n}\n```\nrationale", call

        if agent is AgentName.REVIEWER:
            n = self.calls.count("reviewer") - 1
            verdict = self._verdicts[min(n, len(self._verdicts) - 1)]
            return f"VERDICT: {verdict}\nREASON: stub", call

        return "ok", call


async def _run(router: StubRouter, **state_kwargs) -> AgentState:
    state = AgentState(user_request="build a thing", **state_kwargs)
    out = await build_graph().ainvoke(
        state.model_dump(mode="json"),
        config={"configurable": {"router": router, "sandbox": None}},
    )
    return AgentState.model_validate(
        {k: v for k, v in out.items() if not k.startswith("_")}
    )


@pytest.mark.asyncio
async def test_graph_runs_end_to_end():
    """F-1: the live router must survive every node hop, including the first."""
    final = await _run(StubRouter())
    assert final.is_complete


@pytest.mark.asyncio
async def test_reviewer_rejection_redoes_the_step_and_shows_the_reason():
    """A rejection means redo the current step, and the Coder must see why.

    It must NOT advance the plan: progress is earned by passing tests, not by
    being told the work is wrong.
    """
    router = StubRouter(reviewer_verdicts=("CHANGES_NEEDED", "APPROVED"))
    final = await _run(router)
    assert final.current_step == 0, "rejection skipped ahead instead of redoing"
    assert [a.step_index for a in final.coder_attempts] == [0, 0]
    assert final.reviewer_feedback, "the rejection reason was never recorded"


@pytest.mark.asyncio
async def test_reviewer_rejection_loop_terminates():
    """F-3: a reviewer that never approves must still end the run."""
    router = StubRouter(reviewer_verdicts=("CHANGES_NEEDED",))
    final = await _run(router)
    assert final.is_complete or final.error_message
    assert len(router.calls) < 40, f"unbounded loop: {len(router.calls)} LLM calls"


@pytest.mark.asyncio
async def test_missing_sandbox_is_not_reported_as_passing_tests():
    """F-9: no sandbox must not be reported as a green test run."""
    final = await _run(StubRouter())
    assert final.test_results, "tester never recorded a result"
    assert all(not t.success for t in final.test_results), (
        "tester reported success without executing anything"
    )


class StubSandbox:
    """Minimal sandbox: records writes, returns a scripted sequence of TestResults."""

    def __init__(self, results: list[TestResult]) -> None:
        self._results = list(results)
        self.files: dict[str, str] = {}
        self.runs = 0

    async def write_file(self, path: str, content: str) -> None:
        self.files[path] = content

    async def run_tests(self, command: str, timeout_seconds: int = 120) -> TestResult:
        self.runs += 1
        r = self._results[min(self.runs - 1, len(self._results) - 1)]
        return r.model_copy(update={"command": command})


def _passing(**kw) -> TestResult:
    base = dict(
        command="pytest", passed=3, failed=0, errors=0,
        output="3 passed", duration_ms=1, success=True, no_tests=False,
    )
    return TestResult(**(base | kw))


async def _run_with(router: StubRouter, sandbox, **state_kwargs) -> AgentState:
    state = AgentState(user_request="build a thing", **state_kwargs)
    out = await build_graph().ainvoke(
        state.model_dump(mode="json"),
        config={"configurable": {"router": router, "sandbox": sandbox}},
    )
    return AgentState.model_validate(
        {k: v for k, v in out.items() if not k.startswith("_")}
    )


@pytest.mark.asyncio
async def test_passing_tests_advance_the_plan_to_the_next_step():
    """A step is done when its tests pass — not when the reviewer rejects it."""
    sandbox = StubSandbox([_passing()])
    final = await _run_with(StubRouter(), sandbox)
    assert final.reviewer_approved
    assert [s.status for s in final.execution_plan.steps] == [StepStatus.DONE] * 3
    assert [a.step_index for a in final.coder_attempts] == [0, 1, 2], (
        "every plan step must be coded, not just the first"
    )


@pytest.mark.asyncio
async def test_no_tests_collected_goes_to_the_coder_not_the_debugger():
    """pytest exit 5 means the suite is missing; that is not a bug to debug."""
    sandbox = StubSandbox([_passing(passed=0, success=False, no_tests=True), _passing()])
    router = StubRouter()
    final = await _run_with(router, sandbox)
    assert "debugger" not in router.calls, (
        f"a missing test suite was sent to the Debugger: {router.calls}"
    )
    assert final.is_complete
