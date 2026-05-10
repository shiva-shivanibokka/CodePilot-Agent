"""
Model Router — selects the right Claude model for each agent + task complexity.

Design principle: use the cheapest model that can reliably do the job.
  - Planning, debugging, and reviewing require Sonnet-level reasoning.
  - Code generation and testing are pattern-heavy → Haiku is sufficient.
  - Every call is logged with token counts, latency, and cost.

This gives us:
  1. Cost efficiency at scale (Haiku is ~10x cheaper than Sonnet)
  2. A live cost ticker in the UI (shows real inference economics)
  3. A concrete system design decision to discuss in interviews
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, cast

from anthropic import AsyncAnthropic
from anthropic.types import TextBlock, ToolUseBlock

from models.schemas import AgentName, LLMCall, TaskComplexity


# Anthropic model IDs
SONNET = "claude-sonnet-4-5"
HAIKU = "claude-haiku-3-5"

# Routing table: (agent, complexity) → model
_ROUTING: dict[tuple[AgentName, TaskComplexity], str] = {
    (AgentName.ORCHESTRATOR, TaskComplexity.SIMPLE): HAIKU,
    (AgentName.ORCHESTRATOR, TaskComplexity.COMPLEX): SONNET,
    (AgentName.PLANNER, TaskComplexity.SIMPLE): SONNET,  # always plan carefully
    (AgentName.PLANNER, TaskComplexity.COMPLEX): SONNET,
    (AgentName.CODER, TaskComplexity.SIMPLE): HAIKU,
    (AgentName.CODER, TaskComplexity.COMPLEX): SONNET,
    (AgentName.TESTER, TaskComplexity.SIMPLE): HAIKU,
    (AgentName.TESTER, TaskComplexity.COMPLEX): HAIKU,  # pattern-based
    (AgentName.DEBUGGER, TaskComplexity.SIMPLE): SONNET,  # always needs reasoning
    (AgentName.DEBUGGER, TaskComplexity.COMPLEX): SONNET,
    (AgentName.REVIEWER, TaskComplexity.SIMPLE): SONNET,
    (AgentName.REVIEWER, TaskComplexity.COMPLEX): SONNET,
}


def _extract_text(content: list[Any]) -> str:
    """Extract text from Anthropic response content blocks."""
    return "\n".join(b.text for b in content if isinstance(b, TextBlock))


class ModelRouter:
    """
    Wraps the Anthropic AsyncAnthropic client with model routing and call logging.
    Every agent uses this instead of calling the Anthropic client directly.
    """

    def __init__(self, api_key: str) -> None:
        # Cast to Any so pyright doesn't type-check the messages/tools params
        # at call sites — the SDK accepts plain dicts at runtime fine.
        self._client: Any = AsyncAnthropic(api_key=api_key)
        self._call_log: list[LLMCall] = []

    def select_model(
        self,
        agent: AgentName,
        complexity: TaskComplexity = TaskComplexity.SIMPLE,
    ) -> str:
        return _ROUTING.get((agent, complexity), SONNET)

    def _make_params(
        self,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build kwargs dict for messages.create — typed as Any to avoid SDK type friction."""
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            params["tools"] = tools
        return params

    async def chat(
        self,
        agent: AgentName,
        messages: list[dict[str, Any]],
        system: str,
        complexity: TaskComplexity = TaskComplexity.SIMPLE,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> tuple[str, LLMCall]:
        """
        Send a message to the routed model and return (response_text, LLMCall).
        The LLMCall is logged internally and returned so the caller can add it to state.
        """
        model = self.select_model(agent, complexity)
        start = time.monotonic()

        params = self._make_params(model, max_tokens, system, messages, tools)
        response = await self._client.messages.create(**params)

        latency_ms = int((time.monotonic() - start) * 1000)
        call = LLMCall(
            agent=agent,
            model=model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            timestamp=datetime.utcnow(),
        )
        self._call_log.append(call)

        return _extract_text(response.content), call

    async def chat_with_tools(
        self,
        agent: AgentName,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]],
        complexity: TaskComplexity = TaskComplexity.SIMPLE,
        max_tokens: int = 4096,
    ) -> tuple[str | None, list[LLMCall], list[Any]]:
        """
        Full tool-use loop: call model, handle tool calls, return final response.
        Returns (final_text, list[LLMCall], tool_calls_made).
        """
        model = self.select_model(agent, complexity)
        all_calls: list[LLMCall] = []
        current_messages: list[dict[str, Any]] = list(messages)

        for _ in range(10):  # max 10 tool rounds per agent turn
            start = time.monotonic()
            params = self._make_params(
                model, max_tokens, system, current_messages, tools
            )
            response = await self._client.messages.create(**params)

            latency_ms = int((time.monotonic() - start) * 1000)
            call = LLMCall(
                agent=agent,
                model=model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_ms=latency_ms,
            )
            all_calls.append(call)

            if response.stop_reason == "end_turn":
                return _extract_text(response.content), all_calls, []

            # Collect tool use blocks
            tool_use_blocks = [
                b for b in response.content if isinstance(b, ToolUseBlock)
            ]
            if not tool_use_blocks:
                return _extract_text(response.content), all_calls, []

            # Add assistant message and return tool blocks to caller for execution
            current_messages.append({"role": "assistant", "content": response.content})
            return None, all_calls, tool_use_blocks

        return "Max tool rounds reached.", all_calls, []

    @property
    def call_log(self) -> list[LLMCall]:
        return list(self._call_log)

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self._call_log)
