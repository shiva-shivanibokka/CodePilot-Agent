"""
Arm B — the fixed pipeline.

A LangGraph state machine: plan → code → test → (debug) → review, with the
phase deciding which tools are available and when. This is what the original
CodePilot-Agent was, rebuilt on the same substrate as arm A.

It shares arm A's client, tools, sandbox, workspace, permissions, context and
event stream. **The control flow is the only difference**, which is what makes
the comparison in the eval mean something: any other divergence and you would
be measuring two different agents rather than two ways of sequencing one.

The original's defects are not reproduced. Live objects travel in
`configurable` rather than being hand-copied into every node's return value; a
step advances when its tests pass rather than when the reviewer rejects it; a
missing suite goes back to the coder rather than to the debugger; and the
review loop has a ceiling that something actually increments.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from codepilot.context import Conversation
from codepilot.events import EventType
from codepilot.llm import LLMClient
from codepilot.permissions import Budget, BudgetExceeded
from codepilot.tools import ToolContext, execute, schemas

MAX_DEBUG_ROUNDS = 3
MAX_REVIEWS = 2
#: Tool calls allowed within a single phase before it is cut off.
MAX_PHASE_STEPS = 12

PLANNER_SYSTEM = """\
You plan a coding task, you do not implement it.

Produce a minimal, ordered list of steps. Every step must name the files it
touches. If code needs tests, the step that writes the code or the step right
after it must create them — never emit a step whose only action is "run the
tests", because the tests are run automatically after every step.

Reply with JSON only, no prose:
{"steps": ["...", "..."]}
"""

CODER_SYSTEM = """\
You implement exactly one step of a plan. Not the next step, not the whole task.

Read a file before editing it. Prefer edit_file to write_file. Match the
surrounding code's conventions. When the step is done, stop calling tools and
say in one sentence what you changed.
"""

DEBUGGER_SYSTEM = """\
You are given a failing test run. Find the root cause and fix it.

Fix the code under test, not the test. Read the failing file before editing.
Make the smallest change that addresses the cause rather than the symptom, and
state the root cause in one sentence.
"""

REVIEWER_SYSTEM = """\
You check whether the completed work actually does what was asked.

Reply with exactly one line, then a reason:
VERDICT: APPROVED
VERDICT: CHANGES_NEEDED

Approve only if the task is genuinely done. Do not approve mediocre work to
end the run, and do not demand changes that were never part of the task.
"""

#: Which tools each phase may use. Constraining this is the point of a
#: pipeline: the loop arm gets the whole set and decides for itself.
PHASE_TOOLS = {
    "code": ["read_file", "list_files", "search", "edit_file", "write_file"],
    "debug": ["read_file", "search", "edit_file", "write_file"],
}


@dataclass
class PipelineState:
    task: str
    steps: list[str] = field(default_factory=list)
    current_step: int = 0
    #: How many steps have passed their tests. Counted rather than inferred
    #: from current_step: the pointer is advanced before the routing check,
    #: so comparing it against len(steps) skipped the final step.
    steps_done: int = 0
    debug_rounds: int = 0
    reviews: int = 0
    approved: bool = False
    finished: bool = False
    summary: str = ""
    stopped_by: str = ""
    last_test: str = ""
    tests_passed: bool = False
    edited: list[str] = field(default_factory=list)


@dataclass
class Runtime:
    """The live objects. In `configurable`, never in graph state — graph state
    is serialised at every hop and these cannot survive that."""

    client: LLMClient
    ctx: ToolContext
    convo: Conversation
    budget: Budget
    effort: str | None = "high"
    model: str | None = None


def _runtime(config: RunnableConfig) -> Runtime:
    runtime = (config.get("configurable") or {}).get("runtime")
    if runtime is None:
        raise RuntimeError(
            'The pipeline needs config={"configurable": {"runtime": Runtime(...)}}.'
        )
    return runtime


async def _converse(
    rt: Runtime,
    system: str,
    prompt: str,
    tool_names: list[str] | None,
    max_steps: int = MAX_PHASE_STEPS,
) -> str:
    """One phase: talk to the model, run whatever tools it asks for, return text.

    Uses a private Conversation so a phase's tool chatter does not crowd the
    shared history; only the phase's conclusion is carried forward.
    """
    convo = Conversation(
        system_prompt=system, project_instructions=rt.convo.project_instructions
    )
    convo.user(prompt)
    tools = schemas(tool_names) if tool_names else None
    text = ""

    for _ in range(max_steps):
        rt.budget.check()
        reply = await rt.client.chat(
            convo.messages,
            system=convo.system_blocks(),
            tools=tools,
            model=rt.model,
            effort=rt.effort,
        )
        rt.budget.record(
            reply.cost_usd, reply.usage.input_tokens + reply.usage.output_tokens
        )
        cost = f"${reply.cost_usd:.5f}" if reply.cost_usd is not None else "unpriced"
        rt.ctx.events.emit(
            EventType.COST,
            f"{cost} · {reply.usage.input_tokens:,} in / {reply.usage.output_tokens:,} out",
            cost_usd=reply.cost_usd,
            cache_read=reply.usage.cache_read_tokens,
        )
        if reply.text.strip():
            text = reply.text.strip()
            rt.ctx.events.emit(EventType.ASSISTANT_TEXT, text)
        convo.assistant(reply.content)
        if not reply.wants_tools:
            break
        convo.tool_results([await execute(rt.ctx, call) for call in reply.tool_calls])
    return text


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


async def plan_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    rt = _runtime(config)
    s = PipelineState(**state)
    files = "\n".join(rt.ctx.workspace.list_files()[:80]) or "(empty repository)"
    text = await _converse(
        rt,
        PLANNER_SYSTEM,
        f"Task: {s.task}\n\nFiles in the repository:\n{files}",
        tool_names=None,
        max_steps=1,
    )
    try:
        s.steps = [str(x) for x in json.loads(_extract_json(text)).get("steps", [])]
    except Exception:
        s.steps = []
    if not s.steps:
        # A plan that failed to parse is not a reason to give up: one step is a
        # worse plan, not a broken run.
        s.steps = [s.task]
        rt.ctx.events.emit(
            EventType.ERROR, "could not parse a plan; falling back to a single step"
        )
    rt.ctx.events.emit(
        EventType.PLAN,
        "\n".join(f"  {i}. {step}" for i, step in enumerate(s.steps, 1)),
        steps=s.steps,
    )
    return vars(s)


async def code_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    rt = _runtime(config)
    s = PipelineState(**state)
    if s.current_step >= len(s.steps):
        s.stopped_by = "plan pointer out of range"
        s.finished = True
        return vars(s)

    step = s.steps[s.current_step]
    rt.ctx.events.emit(
        EventType.TURN_START, f"step {s.current_step + 1}/{len(s.steps)}: {step}"
    )
    hint = ""
    if s.last_test and not s.tests_passed:
        hint = f"\n\nThe previous test run reported:\n{s.last_test[:1500]}"
    before = set(rt.ctx.workspace.list_files())
    await _converse(
        rt,
        CODER_SYSTEM,
        f"Overall task: {s.task}\n\nYour step: {step}{hint}",
        tool_names=PHASE_TOOLS["code"],
    )
    s.edited = sorted(set(s.edited) | (set(rt.ctx.workspace.list_files()) - before))
    return vars(s)


async def test_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    rt = _runtime(config)
    s = PipelineState(**state)
    outcome = await rt.ctx.sandbox.run_tests("pytest -q")
    s.last_test = outcome.output[-3000:]
    s.tests_passed = outcome.success
    rt.ctx.events.emit(
        EventType.TEST_RESULT,
        f"{outcome.summary} ({outcome.duration_ms}ms)",
        success=outcome.success,
        no_tests=outcome.no_tests,
    )
    if outcome.no_tests:
        # Back to the coder. Handing a missing directory to the debugger asks a
        # model to root-cause it inside application code, which it will try.
        s.tests_passed = False
        s.last_test = "No tests were collected. Write the tests this step needs."
        rt.ctx.debugging = False
        return vars(s)

    rt.ctx.debugging = not outcome.success
    if outcome.success:
        s.steps_done = max(s.steps_done, s.current_step + 1)  # done when tests pass
        if s.current_step < len(s.steps) - 1:
            s.current_step += 1
    return vars(s)


async def debug_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    rt = _runtime(config)
    s = PipelineState(**state)
    s.debug_rounds += 1
    rt.ctx.events.emit(
        EventType.THINKING, f"debugging, round {s.debug_rounds}/{MAX_DEBUG_ROUNDS}"
    )
    await _converse(
        rt,
        DEBUGGER_SYSTEM,
        f"Overall task: {s.task}\n\nFailing test output:\n{s.last_test}",
        tool_names=PHASE_TOOLS["debug"],
    )
    return vars(s)


async def review_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    rt = _runtime(config)
    s = PipelineState(**state)
    s.reviews += 1
    text = await _converse(
        rt,
        REVIEWER_SYSTEM,
        f"Task: {s.task}\n\nFiles changed: {', '.join(s.edited) or 'none'}\n\n"
        f"Final test run:\n{s.last_test[:2000]}",
        tool_names=["read_file", "list_files", "search"],
    )
    s.approved = bool(re.search(r"VERDICT:\s*APPROVED", text, re.IGNORECASE))
    s.summary = text
    if s.approved or s.reviews >= MAX_REVIEWS:
        s.finished = True
        s.stopped_by = "finished" if s.approved else "review limit"
    return vars(s)


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


def after_test(state: dict[str, Any]) -> str:
    s = PipelineState(**state)
    if s.tests_passed:
        return "code" if s.steps_done < len(s.steps) else "review"
    if "No tests were collected" in s.last_test:
        return "code" if s.debug_rounds < MAX_DEBUG_ROUNDS else "review"
    if s.debug_rounds >= MAX_DEBUG_ROUNDS:
        return "review"
    return "debug"


def after_review(state: dict[str, Any]) -> str:
    s = PipelineState(**state)
    if s.approved or s.reviews >= MAX_REVIEWS:
        return END
    return "code"


def build_pipeline() -> Any:
    builder: Any = StateGraph(dict)
    builder.add_node("plan", plan_node)
    builder.add_node("code", code_node)
    builder.add_node("test", test_node)
    builder.add_node("debug", debug_node)
    builder.add_node("review", review_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "code")
    builder.add_edge("code", "test")
    builder.add_conditional_edges(
        "test", after_test, {"code": "code", "debug": "debug", "review": "review"}
    )
    builder.add_edge("debug", "test")
    builder.add_conditional_edges("review", after_review, {"code": "code", END: END})
    return builder.compile()


async def run_pipeline(rt: Runtime, task: str) -> PipelineState:
    """Run the whole pipeline for one task, matching AgentLoop.run's contract."""
    rt.ctx.events.emit(EventType.USER_MESSAGE, task)
    initial = vars(PipelineState(task=task))
    try:
        final = await build_pipeline().ainvoke(
            initial,
            config={"configurable": {"runtime": rt}, "recursion_limit": 60},
        )
        state = PipelineState(**{k: v for k, v in final.items() if k in initial})
    except BudgetExceeded as exc:
        rt.ctx.events.emit(EventType.BUDGET, str(exc))
        state = PipelineState(task=task, stopped_by="budget", summary=str(exc))
    except Exception as exc:  # noqa: BLE001 - report, do not crash the eval
        rt.ctx.events.emit(EventType.ERROR, f"{type(exc).__name__}: {exc}")
        state = PipelineState(task=task, stopped_by="error", summary=str(exc))

    rt.ctx.events.emit(
        EventType.DONE if state.approved else EventType.ERROR,
        state.summary or "(no summary)",
        stopped_by=state.stopped_by,
        edited=state.edited,
    )
    rt.ctx.events.emit(EventType.TURN_END, rt.budget.summary())
    return state


def _extract_json(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text
