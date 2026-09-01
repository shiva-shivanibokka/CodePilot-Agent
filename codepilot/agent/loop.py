"""
Arm A — the agent loop.

One loop: ask the model, run whatever tools it asked for, feed the results
back, repeat until it calls `finish` or a ceiling stops it. The model chooses
what to do next; there is no fixed plan → code → test → debug pipeline.

That is the whole difference from arm B, and it is deliberately the *only*
difference: both arms share this file's tools, sandbox, workspace, permissions,
context and event stream. If they differed in any of those, comparing them
would measure the wrong variable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from codepilot.agent.interrupt import InterruptChannel
from codepilot.agent.prompts import LOOP_SYSTEM
from codepilot.context import Conversation
from codepilot.events import EventType
from codepilot.llm import LLMClient
from codepilot.permissions import Budget, BudgetExceeded
from codepilot.tools import ToolContext, execute, schemas

#: Hard stop on model calls within one user turn. The budget is the real
#: ceiling; this catches a loop that is somehow cheap and still not finishing.
MAX_STEPS = 60


@dataclass
class TurnResult:
    finished: bool
    summary: str
    steps: int
    stopped_by: str
    edited: list[str] = field(default_factory=list)


class AgentLoop:
    """Runs one user turn to completion."""

    def __init__(
        self,
        client: LLMClient,
        ctx: ToolContext,
        conversation: Conversation,
        budget: Budget,
        *,
        interrupts: InterruptChannel | None = None,
        tool_names: list[str] | None = None,
        effort: str | None = "high",
        model: str | None = None,
    ) -> None:
        self.client = client
        self.ctx = ctx
        self.convo = conversation
        self.budget = budget
        self.interrupts = interrupts or InterruptChannel()
        self.tools = schemas(tool_names)
        self.effort = effort
        self.model = model

    async def run(self, task: str) -> TurnResult:
        events = self.ctx.events
        events.turn += 1
        events.emit(EventType.TURN_START, task, turn=events.turn)
        events.emit(EventType.USER_MESSAGE, task)

        self.convo.user(task)
        self.ctx.finished = False
        self.ctx.final_message = ""
        edited: list[str] = []
        steps = 0
        stopped_by = "finished"

        while steps < MAX_STEPS:
            # --- checkpoint: budget, stop request, queued steers -------------
            try:
                self.budget.check()
            except BudgetExceeded as exc:
                events.emit(EventType.BUDGET, str(exc), summary=self.budget.summary())
                stopped_by = "budget"
                break

            if self.interrupts.stop_requested:
                self.interrupts.clear_stop()
                events.emit(EventType.INTERRUPT, "stopped at your request")
                stopped_by = "interrupted"
                break

            for steer in self.interrupts.drain():
                events.emit(EventType.INTERRUPT, steer)
                self.convo.user(f"[interruption from the user] {steer}")

            # --- compact if the conversation has outgrown the window ---------
            if self.convo.needs_compaction():
                if await self.convo.compact(self.client, self.tools):
                    events.emit(
                        EventType.THINKING,
                        "compacted the earlier conversation to stay within the window",
                    )

            # --- ask the model ----------------------------------------------
            reply = await self.client.chat(
                self.convo.messages,
                system=self.convo.system_blocks(),
                tools=self.tools,
                model=self.model,
                effort=self.effort,
            )
            steps += 1
            self.budget.record(
                reply.cost_usd, reply.usage.input_tokens + reply.usage.output_tokens
            )
            self.convo.usage = self.convo.usage + reply.usage
            await self.convo.token_count(self.client, self.tools)

            cost = f"${reply.cost_usd:.5f}" if reply.cost_usd is not None else "unpriced"
            events.emit(
                EventType.COST,
                f"{cost} · {reply.usage.input_tokens:,} in / "
                f"{reply.usage.output_tokens:,} out · {reply.latency_ms}ms",
                cost_usd=reply.cost_usd,
                input_tokens=reply.usage.input_tokens,
                output_tokens=reply.usage.output_tokens,
                cache_read=reply.usage.cache_read_tokens,
                model=reply.model,
            )

            if reply.refused:
                events.emit(EventType.ERROR, "the model declined this request")
                stopped_by = "refused"
                self.convo.assistant(reply.content)
                break

            if reply.text.strip():
                events.emit(EventType.ASSISTANT_TEXT, reply.text.strip())
            self.convo.assistant(reply.content)

            # --- no tools means the model is done talking --------------------
            if not reply.wants_tools:
                if not self.ctx.finished:
                    # It answered without calling finish. Accept the text as the
                    # result rather than prodding it again: re-prompting here is
                    # a reliable way to burn a turn restating what was just said.
                    self.ctx.final_message = reply.text.strip() or "(no summary given)"
                    stopped_by = "ended without finish"
                break

            # --- run the tools it asked for ---------------------------------
            results = []
            for call in reply.tool_calls:
                results.append(await execute(self.ctx, call))
                if call.name in ("edit_file", "write_file"):
                    path = call.arguments.get("path")
                    if path and path not in edited:
                        edited.append(path)
                # Between tool calls is the only safe place to notice a stop.
                if self.interrupts.stop_requested:
                    break
            self.convo.tool_results(results)

            if self.ctx.finished:
                break
        else:
            stopped_by = "step limit"
            events.emit(
                EventType.BUDGET, f"stopped after {MAX_STEPS} steps without finishing"
            )

        summary = self.ctx.final_message or "(the agent stopped without a summary)"
        events.emit(
            EventType.DONE if self.ctx.finished else EventType.ERROR,
            summary,
            stopped_by=stopped_by,
            steps=steps,
            edited=edited,
            budget=self.budget.summary(),
        )
        events.emit(EventType.TURN_END, self.budget.summary())
        return TurnResult(
            finished=self.ctx.finished,
            summary=summary,
            steps=steps,
            stopped_by=stopped_by,
            edited=edited,
        )


def new_conversation(project_instructions: str | None = None) -> Conversation:
    return Conversation(
        system_prompt=LOOP_SYSTEM, project_instructions=project_instructions
    )
