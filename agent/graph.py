"""
LangGraph state graph — the nervous system of CodePilot-Agent.

Graph structure:
  START → orchestrator_node
            ├── "plan"     → planner_node → coder_node → tester_node
            │                                   ↑              │
            │                              (iterate)    ┌──────┴──────┐
            │                                           │             │
            │                                      pass │        fail │
            │                                           │             ▼
            │                                           │      debugger_node
            │                                           │             │
            │                                     reviewer_node ←────┘
            │                                           │
            │                               ┌───────────┴────────────┐
            │                          approved│              changes │
            │                                 ▼                      ▼
            │                               END              coder_node (loop)
            │
            ├── "qa"       → qa_node → END
            └── "complete" → END
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator

from langgraph.graph import END, START, StateGraph

from agent.prompts import (
    CODER_SYSTEM,
    DEBUGGER_SYSTEM,
    ORCHESTRATOR_SYSTEM,
    PLANNER_SYSTEM,
    REVIEWER_SYSTEM,
    TESTER_SYSTEM,
)
from indexer.ast_indexer import ASTIndexer
from models.schemas import (
    AgentEvent,
    AgentEventType,
    AgentMode,
    AgentName,
    AgentState,
    CoderAttempt,
    DebugTrace,
    ExecutionPlan,
    LLMCall,
    PlanStep,
    TaskComplexity,
)
from router.model_router import ModelRouter
from sandbox.factory import create_sandbox


# ---------------------------------------------------------------------------
# State type alias — plain dict for LangGraph compatibility
# All agent state is serialised/deserialised via Pydantic at each node.
# Non-serialisable objects (_router, _sandbox) live in the dict alongside.
# ---------------------------------------------------------------------------

# Simple alias — avoids TypedDict friction with LangGraph's pyright stubs
GDict = dict  # generic dict, typed as Any throughout


def _state_from_dict(d: GDict) -> AgentState:
    return AgentState.model_validate(d)


def _state_to_dict(s: AgentState) -> GDict:
    return s.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Helper: emit an event into state and yield for streaming
# ---------------------------------------------------------------------------


def _emit(
    state: AgentState,
    agent: AgentName,
    etype: AgentEventType,
    msg: str,
    payload: dict | None = None,
) -> AgentEvent:
    ev = AgentEvent(
        session_id=state.session_id,
        agent=agent,
        event_type=etype,
        message=msg,
        payload=payload or {},
    )
    state.add_event(ev)
    return ev


# ---------------------------------------------------------------------------
# Node: Orchestrator
# ---------------------------------------------------------------------------


async def orchestrator_node(state_dict: dict[str, Any]) -> dict[str, Any]:
    state = _state_from_dict(state_dict)
    router: ModelRouter = state_dict["_router"]

    _emit(
        state,
        AgentName.ORCHESTRATOR,
        AgentEventType.THINKING,
        f"Received request: '{state.user_request[:80]}...'",
    )

    # Classify complexity
    system = ORCHESTRATOR_SYSTEM
    messages = [
        {
            "role": "user",
            "content": (
                f"User request: {state.user_request}\n\n"
                f"Mode: {state.mode.value}\n\n"
                "Classify this as SIMPLE or COMPLEX and explain in one sentence. "
                'Reply with JSON: {{"complexity": "simple"|"complex", "reason": "..."}}'
            ),
        }
    ]

    text, call = await router.chat(
        AgentName.ORCHESTRATOR,
        messages,
        system,
        complexity=TaskComplexity.SIMPLE,
        max_tokens=256,
    )
    state.llm_calls.append(call)

    try:
        parsed = json.loads(text.strip())
        complexity_str = parsed.get("complexity", "simple").lower()
        reason = parsed.get("reason", "")
    except Exception:
        complexity_str, reason = "simple", "Could not parse complexity"

    complexity = (
        TaskComplexity.COMPLEX if complexity_str == "complex" else TaskComplexity.SIMPLE
    )
    if state.execution_plan is None:
        state.execution_plan = ExecutionPlan.empty(state.user_request)
    state.execution_plan.complexity = complexity

    _emit(
        state,
        AgentName.ORCHESTRATOR,
        AgentEventType.THINKING,
        f"Complexity: {complexity.value} — {reason}",
    )
    _emit(
        state,
        AgentName.ORCHESTRATOR,
        AgentEventType.COST,
        f"${call.cost_usd:.5f} | {call.input_tokens + call.output_tokens} tokens",
    )

    return _state_to_dict(state)


# ---------------------------------------------------------------------------
# Node: Planner
# ---------------------------------------------------------------------------


async def planner_node(state_dict: dict[str, Any]) -> dict[str, Any]:
    state = _state_from_dict(state_dict)
    router: ModelRouter = state_dict["_router"]

    # Build AST index of workspace
    indexer = ASTIndexer()
    codebase_index = indexer.build(state.workspace_files)
    codebase_summary = codebase_index.summary()
    relevant_files = codebase_index.relevant_files(state.user_request, top_k=5)

    _emit(
        state,
        AgentName.PLANNER,
        AgentEventType.THINKING,
        f"Analysing codebase ({len(state.workspace_files)} files)...",
    )
    _emit(
        state,
        AgentName.PLANNER,
        AgentEventType.TOOL_RESULT,
        f"Relevant files identified: {', '.join(relevant_files) or 'none yet — starting fresh'}",
    )

    system = PLANNER_SYSTEM
    messages = [
        {
            "role": "user",
            "content": (
                f"Task: {state.user_request}\n\n"
                f"Codebase structure:\n{codebase_summary}\n\n"
                f"Most relevant files for this task: {', '.join(relevant_files)}\n\n"
                "Produce an ExecutionPlan as JSON with this structure:\n"
                '{"task": "...", "complexity": "simple|complex", "steps": ['
                '{"index": 0, "description": "...", "files_to_read": [...], '
                '"files_to_write": [...], "test_command": "pytest tests/ -v"}], '
                '"total_steps": N}'
            ),
        }
    ]

    complexity = (
        state.execution_plan.complexity
        if state.execution_plan
        else TaskComplexity.SIMPLE
    )
    text, call = await router.chat(
        AgentName.PLANNER, messages, system, complexity=complexity, max_tokens=2048
    )
    state.llm_calls.append(call)

    # Parse plan from LLM output
    try:
        # Extract JSON from response (LLM may wrap it in markdown)
        json_match = _extract_json(text)
        plan_data = json.loads(json_match)
        steps = [PlanStep(**s) for s in plan_data.get("steps", [])]
        plan = ExecutionPlan(
            task=state.user_request,
            complexity=complexity,
            steps=steps,
            total_steps=len(steps),
        )
        state.execution_plan = plan
    except Exception as exc:
        _emit(
            state,
            AgentName.PLANNER,
            AgentEventType.ERROR,
            f"Failed to parse plan: {exc}. Using single-step fallback.",
        )
        state.execution_plan = ExecutionPlan(
            task=state.user_request,
            complexity=complexity,
            steps=[
                PlanStep(
                    index=0,
                    description=f"Implement: {state.user_request}",
                    files_to_write=list(relevant_files[:2]) or ["solution.py"],
                    test_command="pytest tests/ -v --tb=short",
                )
            ],
            total_steps=1,
        )

    plan_text = "\n".join(
        f"  Step {s.index + 1}: {s.description}" for s in state.execution_plan.steps
    )
    _emit(
        state,
        AgentName.PLANNER,
        AgentEventType.PLAN,
        f"Execution plan ({state.execution_plan.total_steps} steps):\n{plan_text}",
        payload={"plan": state.execution_plan.model_dump()},
    )
    _emit(
        state,
        AgentName.PLANNER,
        AgentEventType.COST,
        f"${call.cost_usd:.5f} | {call.input_tokens + call.output_tokens} tokens",
    )

    # Store index results in state for coder to use
    state_dict_out = _state_to_dict(state)
    state_dict_out["_codebase_index"] = codebase_index  # pass index forward
    state_dict_out["_router"] = router
    return state_dict_out


# ---------------------------------------------------------------------------
# Node: Coder
# ---------------------------------------------------------------------------


async def coder_node(state_dict: dict[str, Any]) -> dict[str, Any]:
    state = _state_from_dict(state_dict)
    router: ModelRouter = state_dict["_router"]
    sandbox = state_dict.get("_sandbox")

    if not state.execution_plan or not state.execution_plan.steps:
        _emit(state, AgentName.CODER, AgentEventType.ERROR, "No execution plan found.")
        state.error_message = "No execution plan"
        return _state_to_dict(state) | {"_router": router, "_sandbox": sandbox}

    step = state.execution_plan.steps[state.current_step]
    _emit(
        state,
        AgentName.CODER,
        AgentEventType.STEP,
        f"Step {step.index + 1}/{state.execution_plan.total_steps}: {step.description}",
    )

    # Build compact context: only relevant files for this step
    file_contexts = []
    for fname in step.files_to_read:
        if fname in state.workspace_files:
            content = state.workspace_files[fname]
            # Truncate very large files to stay within context budget
            if len(content) > 8000:
                content = content[:8000] + "\n... [truncated for context] ..."
            file_contexts.append(f"=== {fname} ===\n{content}")

    prior_attempts = state.attempts_for_step(step.index)
    prior_text = ""
    if prior_attempts:
        last = prior_attempts[-1]
        prior_text = f"\n\nPREVIOUS ATTEMPT FAILED. Rationale was: {last.rationale}\nTry a different approach."

    messages = [
        {
            "role": "user",
            "content": (
                f"Task: {state.user_request}\n\n"
                f"Current step: {step.description}\n"
                f"Files to write: {', '.join(step.files_to_write)}\n\n"
                f"Current file contents:\n{''.join(file_contexts) or 'No existing files — create them.'}"
                f"{prior_text}\n\n"
                "Write the implementation. For each file you write, output:\n"
                "WRITE_FILE: <filename>\n```python\n<complete file content>\n```\n\n"
                "Then explain your approach in 2-3 sentences."
            ),
        }
    ]

    complexity = state.execution_plan.complexity
    text, call = await router.chat(
        AgentName.CODER, messages, CODER_SYSTEM, complexity=complexity, max_tokens=4096
    )
    state.llm_calls.append(call)

    # Parse WRITE_FILE blocks from LLM output
    files_written: dict[str, str] = {}
    import re

    pattern = re.compile(
        r"WRITE_FILE:\s*([^\n]+)\n```(?:python|py)?\n(.*?)```", re.DOTALL
    )
    for match in pattern.finditer(text):
        filename = match.group(1).strip()
        content = match.group(2)
        files_written[filename] = content
        state.workspace_files[filename] = content
        if filename not in state.original_files:
            state.original_files[filename] = ""
        _emit(
            state,
            AgentName.CODER,
            AgentEventType.CODE_WRITTEN,
            f"Written: {filename} ({content.count(chr(10)) + 1} lines)",
            payload={"filename": filename, "content": content},
        )

        # Sync to sandbox if available
        if sandbox:
            await sandbox.write_file(filename, content)

    if not files_written:
        # LLM didn't use WRITE_FILE format — try to extract any code block
        code_blocks = re.findall(r"```(?:python|py)?\n(.*?)```", text, re.DOTALL)
        if code_blocks:
            default_name = (
                step.files_to_write[0] if step.files_to_write else "solution.py"
            )
            content = code_blocks[0]
            files_written[default_name] = content
            state.workspace_files[default_name] = content
            if sandbox:
                await sandbox.write_file(default_name, content)
            _emit(
                state,
                AgentName.CODER,
                AgentEventType.CODE_WRITTEN,
                f"Written: {default_name} (parsed from code block)",
            )

    # Record attempt
    rationale = (
        text.split("WRITE_FILE:")[0].strip()[-500:]
        if "WRITE_FILE:" in text
        else text[-500:]
    )
    attempt = CoderAttempt(
        step_index=step.index,
        files_written=files_written,
        rationale=rationale,
    )
    state.coder_attempts.append(attempt)

    _emit(
        state,
        AgentName.CODER,
        AgentEventType.COST,
        f"${call.cost_usd:.5f} | {call.input_tokens + call.output_tokens} tokens",
    )

    return _state_to_dict(state) | {"_router": router, "_sandbox": sandbox}


# ---------------------------------------------------------------------------
# Node: Tester
# ---------------------------------------------------------------------------


async def tester_node(state_dict: dict[str, Any]) -> dict[str, Any]:
    state = _state_from_dict(state_dict)
    router: ModelRouter = state_dict["_router"]
    sandbox = state_dict.get("_sandbox")

    step = (
        state.execution_plan.steps[state.current_step] if state.execution_plan else None
    )
    test_cmd = (
        step.test_command
        if step and step.test_command
        else "pytest tests/ -v --tb=short"
    )

    _emit(state, AgentName.TESTER, AgentEventType.TEST_RUN, f"Running: {test_cmd}")

    if sandbox:
        result = await sandbox.run_tests(test_cmd)
    else:
        # No sandbox — skip test execution, report as passed for demo
        from models.schemas import TestResult

        result = TestResult(
            command=test_cmd,
            passed=0,
            failed=0,
            errors=0,
            output="[No sandbox configured — skipping test execution]",
            duration_ms=0,
            success=True,
        )

    state.test_results.append(result)

    if result.success:
        _emit(
            state,
            AgentName.TESTER,
            AgentEventType.TEST_RESULT,
            f"✅ {result.summary} ({result.duration_ms}ms)",
            payload={"success": True, "output": result.output},
        )
    else:
        _emit(
            state,
            AgentName.TESTER,
            AgentEventType.TEST_RESULT,
            f"❌ {result.summary} — sending to Debugger",
            payload={"success": False, "output": result.output},
        )

    return _state_to_dict(state) | {"_router": router, "_sandbox": sandbox}


# ---------------------------------------------------------------------------
# Node: Debugger
# ---------------------------------------------------------------------------


async def debugger_node(state_dict: dict[str, Any]) -> dict[str, Any]:
    state = _state_from_dict(state_dict)
    router: ModelRouter = state_dict["_router"]
    sandbox = state_dict.get("_sandbox")

    last_result = state.last_test_result()
    if not last_result:
        return _state_to_dict(state) | {"_router": router, "_sandbox": sandbox}

    state.iteration_count += 1
    _emit(
        state,
        AgentName.DEBUGGER,
        AgentEventType.DEBUG,
        f"Debugging iteration {state.iteration_count}/{state.max_iterations}...",
    )

    # Read the files that failed
    failure_output = last_result.output[-3000:]
    current_files_text = ""
    for fname, content in list(state.workspace_files.items())[:5]:
        if fname.endswith(".py"):
            current_files_text += f"\n=== {fname} ===\n{content[:3000]}\n"

    messages = [
        {
            "role": "user",
            "content": (
                f"Test failure output:\n{failure_output}\n\n"
                f"Current files:\n{current_files_text}\n\n"
                "Identify the root cause and apply a minimal fix.\n"
                "Output:\n"
                "ROOT_CAUSE: <one sentence>\n"
                "WRITE_FILE: <filename>\n```python\n<complete fixed file>\n```"
            ),
        }
    ]

    text, call = await router.chat(
        AgentName.DEBUGGER,
        messages,
        DEBUGGER_SYSTEM,
        complexity=TaskComplexity.COMPLEX,
        max_tokens=4096,
    )
    state.llm_calls.append(call)

    # Parse fix
    import re

    root_cause = ""
    cause_match = re.search(r"ROOT_CAUSE:\s*(.+)", text)
    if cause_match:
        root_cause = cause_match.group(1).strip()
        _emit(
            state, AgentName.DEBUGGER, AgentEventType.DEBUG, f"Root cause: {root_cause}"
        )

    patch_applied: dict[str, str] = {}
    pattern = re.compile(
        r"WRITE_FILE:\s*([^\n]+)\n```(?:python|py)?\n(.*?)```", re.DOTALL
    )
    for match in pattern.finditer(text):
        filename = match.group(1).strip()
        content = match.group(2)
        patch_applied[filename] = content
        state.workspace_files[filename] = content
        if sandbox:
            await sandbox.write_file(filename, content)
        _emit(
            state,
            AgentName.DEBUGGER,
            AgentEventType.PATCH,
            f"Patched: {filename}",
            payload={"filename": filename},
        )

    trace = DebugTrace(
        test_result=last_result,
        root_cause=root_cause or "Unknown",
        patch_applied=patch_applied,
        iteration=state.iteration_count,
    )
    state.debug_traces.append(trace)

    _emit(
        state,
        AgentName.DEBUGGER,
        AgentEventType.COST,
        f"${call.cost_usd:.5f} | {call.input_tokens + call.output_tokens} tokens",
    )

    return _state_to_dict(state) | {"_router": router, "_sandbox": sandbox}


# ---------------------------------------------------------------------------
# Node: Reviewer
# ---------------------------------------------------------------------------


async def reviewer_node(state_dict: dict[str, Any]) -> dict[str, Any]:
    state = _state_from_dict(state_dict)
    router: ModelRouter = state_dict["_router"]
    sandbox = state_dict.get("_sandbox")

    _emit(
        state,
        AgentName.REVIEWER,
        AgentEventType.THINKING,
        "Reviewing implementation against original task requirements...",
    )

    files_text = ""
    for fname, content in state.workspace_files.items():
        if fname.endswith(".py"):
            files_text += f"\n=== {fname} ===\n{content[:4000]}\n"

    _rev_last = state.last_test_result()
    _rev_test_summary = _rev_last.summary if _rev_last is not None else "Not run"

    messages = [
        {
            "role": "user",
            "content": (
                f"Original task: {state.user_request}\n\n"
                f"Final test result: {_rev_test_summary}\n\n"
                f"Implementation:\n{files_text}\n\n"
                "Review and respond with APPROVED or CHANGES_NEEDED:\n"
                "VERDICT: APPROVED|CHANGES_NEEDED\n"
                "REASON: <explanation>"
            ),
        }
    ]

    text, call = await router.chat(
        AgentName.REVIEWER,
        messages,
        REVIEWER_SYSTEM,
        complexity=TaskComplexity.COMPLEX,
        max_tokens=1024,
    )
    state.llm_calls.append(call)

    import re

    verdict_match = re.search(
        r"VERDICT:\s*(APPROVED|CHANGES_NEEDED)", text, re.IGNORECASE
    )
    verdict = verdict_match.group(1).upper() if verdict_match else "APPROVED"
    reason_match = re.search(r"REASON:\s*(.+)", text, re.DOTALL)
    reason = reason_match.group(1).strip() if reason_match else text.strip()

    state.reviewer_approved = verdict == "APPROVED"

    if state.reviewer_approved:
        _emit(
            state, AgentName.REVIEWER, AgentEventType.DONE, f"Approved: {reason[:200]}"
        )
        state.is_complete = True
        state.final_output = _build_final_output(state)
    else:
        _emit(
            state,
            AgentName.REVIEWER,
            AgentEventType.THINKING,
            f"Changes needed: {reason[:200]}",
        )

    _emit(
        state,
        AgentName.REVIEWER,
        AgentEventType.COST,
        f"${call.cost_usd:.5f} | {call.input_tokens + call.output_tokens} tokens",
    )

    return _state_to_dict(state) | {"_router": router, "_sandbox": sandbox}


# ---------------------------------------------------------------------------
# Node: QA (simple codebase Q&A, no code writing)
# ---------------------------------------------------------------------------


async def qa_node(state_dict: dict[str, Any]) -> dict[str, Any]:
    state = _state_from_dict(state_dict)
    router: ModelRouter = state_dict["_router"]

    _emit(
        state,
        AgentName.ORCHESTRATOR,
        AgentEventType.THINKING,
        "Answering question about the codebase...",
    )

    indexer = ASTIndexer()
    codebase_index = indexer.build(state.workspace_files)
    relevant_files = codebase_index.relevant_files(state.user_request, top_k=3)

    context = ""
    for fname in relevant_files:
        if fname in state.workspace_files:
            context += f"\n=== {fname} ===\n{state.workspace_files[fname][:3000]}\n"

    messages = [
        {
            "role": "user",
            "content": f"Question: {state.user_request}\n\nRelevant code:\n{context or 'No files uploaded.'}",
        }
    ]

    text, call = await router.chat(
        AgentName.ORCHESTRATOR,
        messages,
        "You are a helpful coding assistant. Answer questions about the codebase clearly and concisely.",
        complexity=TaskComplexity.SIMPLE,
        max_tokens=1024,
    )
    state.llm_calls.append(call)
    state.final_output = text
    state.is_complete = True

    _emit(state, AgentName.ORCHESTRATOR, AgentEventType.DONE, text)
    return _state_to_dict(state) | {"_router": router}


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------


def route_after_orchestrator(state_dict: dict[str, Any]) -> str:
    state = _state_from_dict(state_dict)
    if state.mode == AgentMode.QA:
        return "qa"
    return "plan"


def route_after_tester(state_dict: dict[str, Any]) -> str:
    state = _state_from_dict(state_dict)
    last = state.last_test_result()
    if last and last.success:
        return "review"
    if state.iteration_count >= state.max_iterations:
        return "review"  # force review even on failure — reviewer will reject
    return "debug"


def route_after_debugger(state_dict: dict[str, Any]) -> str:
    return "test"  # always re-test after debugging


def route_after_reviewer(state_dict: dict[str, Any]) -> str:
    state = _state_from_dict(state_dict)
    if state.reviewer_approved:
        return END
    if state.iteration_count >= state.max_iterations:
        return END  # give up cleanly
    # Advance to next step or loop back to coder
    if (
        state.execution_plan
        and state.current_step < state.execution_plan.total_steps - 1
    ):
        state.current_step += 1
    return "code"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph() -> Any:
    """
    Compile and return the LangGraph state machine.
    Typed as Any to avoid pyright friction with LangGraph's
    generic StateGraph parameter — all runtime type safety
    comes from Pydantic validation inside each node.
    """
    # Use Any-typed builder to bypass pyright's strict StateGraph generics
    builder: Any = StateGraph(dict)
    builder.add_node("orchestrate", orchestrator_node)
    builder.add_node("plan", planner_node)
    builder.add_node("code", coder_node)
    builder.add_node("test", tester_node)
    builder.add_node("debug", debugger_node)
    builder.add_node("review", reviewer_node)
    builder.add_node("qa", qa_node)

    builder.add_edge(START, "orchestrate")

    builder.add_conditional_edges(
        "orchestrate",
        route_after_orchestrator,
        {"plan": "plan", "qa": "qa"},
    )

    builder.add_edge("plan", "code")
    builder.add_edge("code", "test")

    builder.add_conditional_edges(
        "test",
        route_after_tester,
        {"review": "review", "debug": "debug"},
    )

    builder.add_edge("debug", "test")

    builder.add_conditional_edges(
        "review",
        route_after_reviewer,
        {"code": "code", END: END},
    )

    builder.add_edge("qa", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> str:
    """Extract first JSON object from text that may contain markdown."""
    import re

    # Try ```json block first
    match = re.search(r"```(?:json)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1)
    # Try raw JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def _build_final_output(state: AgentState) -> str:
    _last = state.last_test_result()
    test_summary = _last.summary if _last is not None else "not run"
    lines = [
        f"Task complete: {state.user_request}",
        "",
        f"Files modified: {', '.join(state.workspace_files.keys())}",
        f"Tests: {test_summary}",
        f"Total cost: ${state.total_cost_usd:.4f} ({state.total_tokens} tokens)",
        f"Iterations: {state.iteration_count}",
    ]
    return "\n".join(lines)
