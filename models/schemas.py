"""
Core Pydantic schemas for CodePilot-Agent.

Every agent reads from and writes to AgentState.
Every user-visible update is an AgentEvent.
Every inter-agent contract is a typed Pydantic model.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AgentMode(str, Enum):
    TASK = "task"  # natural language → working code
    QA = "qa"  # chat with codebase
    ISSUE = "issue"  # GitHub issue URL → PR


class AgentEventType(str, Enum):
    THINKING = "thinking"  # agent internal reasoning (shown dimmed)
    TOOL_CALL = "tool_call"  # "Reading file main.py..."
    TOOL_RESULT = "tool_result"  # "Found 247 lines"
    CODE_WRITTEN = "code_written"  # new/edited file content
    TEST_RUN = "test_run"  # "Running pytest tests/"
    TEST_RESULT = "test_result"  # pass/fail summary
    DEBUG = "debug"  # "NameError on line 42"
    PATCH = "patch"  # fix being applied (unified diff)
    PLAN = "plan"  # execution plan created
    STEP = "step"  # entering a new plan step
    DONE = "done"  # task complete
    ERROR = "error"  # agent gave up
    COST = "cost"  # token/cost update


class AgentName(str, Enum):
    ORCHESTRATOR = "orchestrator"
    PLANNER = "planner"
    CODER = "coder"
    TESTER = "tester"
    DEBUGGER = "debugger"
    REVIEWER = "reviewer"


class TaskComplexity(str, Enum):
    SIMPLE = "simple"  # single file, clear intent → Haiku
    COMPLEX = "complex"  # multi-file, ambiguous, architectural → Sonnet


class StepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Execution Plan
# ---------------------------------------------------------------------------


class PlanStep(BaseModel):
    index: int
    description: str  # what this step does
    files_to_read: list[str] = Field(default_factory=list)
    files_to_write: list[str] = Field(default_factory=list)
    test_command: str | None = None  # pytest command for this step
    status: StepStatus = StepStatus.PENDING


class ExecutionPlan(BaseModel):
    task: str
    complexity: TaskComplexity
    steps: list[PlanStep]
    total_steps: int

    @classmethod
    def empty(cls, task: str) -> "ExecutionPlan":
        return cls(task=task, complexity=TaskComplexity.SIMPLE, steps=[], total_steps=0)


# ---------------------------------------------------------------------------
# Sandbox / Execution Results
# ---------------------------------------------------------------------------


class ExecutionResult(BaseModel):
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False
    duration_ms: int = 0

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class TestResult(BaseModel):
    command: str
    passed: int
    failed: int
    errors: int
    output: str  # full pytest output
    duration_ms: int
    success: bool

    @property
    def summary(self) -> str:
        return f"{self.passed} passed, {self.failed} failed, {self.errors} errors"


# ---------------------------------------------------------------------------
# Agent Attempt Records (for shared state history)
# ---------------------------------------------------------------------------


class CoderAttempt(BaseModel):
    step_index: int
    files_written: dict[str, str]  # filename → content
    rationale: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DebugTrace(BaseModel):
    test_result: TestResult
    root_cause: str
    patch_applied: dict[str, str]  # filename → patched content
    iteration: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Agent Events (streamed to UI in real time)
# ---------------------------------------------------------------------------


class AgentEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    session_id: str
    agent: AgentName
    event_type: AgentEventType
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def to_ui_line(self) -> str:
        """Human-readable line shown in the Gradio chat."""
        icons = {
            AgentEventType.THINKING: "🤔",
            AgentEventType.TOOL_CALL: "🔧",
            AgentEventType.TOOL_RESULT: "📋",
            AgentEventType.CODE_WRITTEN: "✍️ ",
            AgentEventType.TEST_RUN: "🧪",
            AgentEventType.TEST_RESULT: "📊",
            AgentEventType.DEBUG: "🔍",
            AgentEventType.PATCH: "🩹",
            AgentEventType.PLAN: "📝",
            AgentEventType.STEP: "➡️ ",
            AgentEventType.DONE: "✅",
            AgentEventType.ERROR: "❌",
            AgentEventType.COST: "💰",
        }
        icon = icons.get(self.event_type, "•")
        return f"{icon} [{self.agent.value}] {self.message}"


# ---------------------------------------------------------------------------
# LLM Call Tracking (cost/token monitoring)
# ---------------------------------------------------------------------------


class LLMCall(BaseModel):
    agent: AgentName
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def cost_usd(self) -> float:
        """USD per token, Anthropic list price.

        An unknown model returns 0.0 rather than guessing at another model's rate:
        a cost ticker that is confidently wrong is worse than one that is blank.
        """
        pricing = {
            "claude-opus-5": (5.00e-6, 25.00e-6),
            "claude-sonnet-5": (2.00e-6, 10.00e-6),
            "claude-haiku-4-5": (1.00e-6, 5.00e-6),
        }
        if self.model not in pricing:
            return 0.0
        in_price, out_price = pricing[self.model]
        return (self.input_tokens * in_price) + (self.output_tokens * out_price)


# ---------------------------------------------------------------------------
# LangGraph Shared State
# ---------------------------------------------------------------------------


class AgentState(BaseModel):
    """
    The single shared state object that flows through the entire LangGraph.
    Every agent reads from and writes to this. The Orchestrator owns it.
    """

    # Session identity
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # User input
    user_request: str = ""
    mode: AgentMode = AgentMode.TASK
    github_url: str | None = None  # for ISSUE mode

    # Planning
    execution_plan: ExecutionPlan | None = None
    current_step: int = 0

    # Workspace: all files the agent is working with
    workspace_files: dict[str, str] = Field(default_factory=dict)  # path → content
    original_files: dict[str, str] = Field(default_factory=dict)  # for diffing

    # Sandbox
    sandbox_id: str | None = None
    sandbox_type: Literal["docker", "subprocess", "e2b"] = "subprocess"

    # History (what each agent has tried)
    coder_attempts: list[CoderAttempt] = Field(default_factory=list)
    test_results: list[TestResult] = Field(default_factory=list)
    debug_traces: list[DebugTrace] = Field(default_factory=list)
    llm_calls: list[LLMCall] = Field(default_factory=list)

    # Streaming events (ordered log of everything that happened)
    events: list[AgentEvent] = Field(default_factory=list)

    # Control flow
    iteration_count: int = 0        # debugger passes
    max_iterations: int = 5
    review_count: int = 0           # reviewer passes — bounds the coder<->reviewer loop
    max_reviews: int = 3
    max_cost_usd: float = 1.00      # hard ceiling for one session
    reviewer_approved: bool = False
    needs_user_input: bool = False
    user_correction: str | None = None

    # Final output
    final_output: str | None = None
    pr_url: str | None = None
    error_message: str | None = None
    is_complete: bool = False

    # Aggregated metrics
    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.llm_calls)

    @property
    def total_tokens(self) -> int:
        return sum(c.input_tokens + c.output_tokens for c in self.llm_calls)

    def add_event(self, event: AgentEvent) -> None:
        self.events.append(event)

    def last_test_result(self) -> TestResult | None:
        return self.test_results[-1] if self.test_results else None

    def attempts_for_step(self, step_index: int) -> list[CoderAttempt]:
        return [a for a in self.coder_attempts if a.step_index == step_index]


# ---------------------------------------------------------------------------
# API Request / Response models
# ---------------------------------------------------------------------------


class TaskRequest(BaseModel):
    user_request: str
    mode: AgentMode = AgentMode.TASK
    github_url: str | None = None
    sandbox_type: Literal["docker", "subprocess", "e2b"] = "subprocess"
    workspace_files: dict[str, str] = Field(default_factory=dict)
    anthropic_api_key: str | None = None  # user-provided key


class SessionResponse(BaseModel):
    session_id: str
    status: str = "started"
    ws_url: str  # WebSocket URL for streaming


class FinalResponse(BaseModel):
    session_id: str
    is_complete: bool
    final_output: str | None
    pr_url: str | None
    error_message: str | None
    total_cost_usd: float
    total_tokens: int
    workspace_files: dict[str, str]
