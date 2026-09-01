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
from models.schemas import AgentName, AgentState, LLMCall, TaskComplexity


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
async def test_every_plan_step_is_executed():
    """F-2: a 3-step plan must advance through steps 0, 1, 2 — not repeat step 0."""
    router = StubRouter(
        reviewer_verdicts=("CHANGES_NEEDED", "CHANGES_NEEDED", "APPROVED")
    )
    final = await _run(router)
    assert final.current_step == 2, "plan pointer never advanced"
    assert [a.step_index for a in final.coder_attempts] == [0, 1, 2]


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
