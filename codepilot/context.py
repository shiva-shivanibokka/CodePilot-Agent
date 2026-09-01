"""
The conversation.

The thing the old six-agent graph never had: every node there made a stateless
one-shot call, which is why the Debugger could not see what the Coder had just
tried. An agent that cannot remember its own last action cannot iterate.

Two jobs beyond holding a list of messages:

**Cache breakpoints.** A coding agent re-sends its system prompt, its project
instructions and its tool schemas on every single turn. Cached, that prefix
costs a tenth; uncached it is usually the majority of the bill. Caching is a
prefix match, so the order is fixed — tools, then system, then CODEPILOT.md —
and nothing volatile is allowed in front of the breakpoint.

A prefix shorter than the model's minimum **silently does not cache** — no
error, no warning, `cache_creation_input_tokens` just stays 0. Measured here:
a 2,119-token prefix cached on neither model, while 7,239 tokens (Haiku 4.5)
and 9,327 (Opus 5) both cached and were fully read back on the next turn. So
zero cache hits is not proof of a bug, and it is not proof of correctness
either — check the prefix size before concluding anything, and use
`cache_report()` rather than eyeballing one response.

**Compaction.** Tool results are large and go stale fast. Past a threshold the
oldest ones are summarised while the system prefix and the most recent turns
survive verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from codepilot.llm import LLMClient, Usage

#: Compact when the conversation passes this many tokens.
DEFAULT_COMPACT_AT = 120_000
#: Always keep at least this many of the most recent messages verbatim.
KEEP_RECENT = 12


@dataclass
class Conversation:
    """Message history plus the prefix that gets cached in front of it."""

    system_prompt: str
    #: Contents of CODEPILOT.md, if the project has one. Cached with the system
    #: prompt because it changes about as often — which is to say, rarely.
    project_instructions: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    compact_at: int = DEFAULT_COMPACT_AT
    _last_token_count: int = 0
    _compactions: int = 0

    # ------------------------------------------------------------------
    # Building the request
    # ------------------------------------------------------------------

    def system_blocks(self) -> list[dict[str, Any]]:
        """The stable prefix, with the cache breakpoint on its last block.

        Everything volatile — the task, timestamps, running cost — belongs in
        `messages`, after this. A single changing byte in here invalidates the
        cache for every turn that follows.
        """
        blocks: list[dict[str, Any]] = [{"type": "text", "text": self.system_prompt}]
        if self.project_instructions:
            blocks.append(
                {
                    "type": "text",
                    "text": "# Project instructions (CODEPILOT.md)\n\n"
                    + self.project_instructions,
                }
            )
        blocks[-1]["cache_control"] = {"type": "ephemeral"}
        return blocks

    def cache_report(self) -> str:
        """Whether the prefix is actually being cached, and what it saved.

        Worth surfacing: an invalidated prefix costs 10x and reports nothing.
        """
        read = self.usage.cache_read_tokens
        written = self.usage.cache_write_tokens
        if not read and not written:
            return (
                "prompt cache: never engaged — the prefix is likely below the model's "
                "minimum cacheable size"
            )
        billed = self.usage.input_tokens + read
        pct = (read / billed * 100) if billed else 0.0
        return (
            f"prompt cache: {read:,} tokens read, {written:,} written "
            f"({pct:.0f}% of input served from cache)"
        )

    # ------------------------------------------------------------------
    # Appending turns
    # ------------------------------------------------------------------

    def user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def assistant(self, content: list[Any]) -> None:
        """Append the reply's raw content blocks, not just its text.

        Thinking blocks have to round-trip unmodified, and tool_use blocks are
        what the following tool_result blocks refer to by id. Flattening to a
        string loses both.
        """
        self.messages.append({"role": "assistant", "content": content})

    def tool_results(self, results: list[dict[str, Any]]) -> None:
        """All results for one assistant turn, in a single user message.

        Splitting them across several messages teaches the model to stop making
        parallel tool calls, so they go together even when one failed.
        """
        self.messages.append({"role": "user", "content": results})

    @staticmethod
    def tool_result(
        tool_use_id: str, content: str, is_error: bool = False
    ) -> dict[str, Any]:
        """A failed tool is a result, never an exception.

        The model can read an error and recover; it cannot recover from a
        traceback that ended the turn.
        """
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
        }
        if is_error:
            block["is_error"] = True
        return block

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    async def token_count(
        self, client: LLMClient, tools: list[dict[str, Any]] | None = None
    ) -> int:
        if not self.messages:
            return 0
        self._last_token_count = await client.count_tokens(
            self.messages, system=self.system_blocks(), tools=tools
        )
        return self._last_token_count

    def needs_compaction(self) -> bool:
        return self._last_token_count > self.compact_at

    async def compact(
        self, client: LLMClient, tools: list[dict[str, Any]] | None = None
    ) -> bool:
        """Summarise the older half, keep the recent turns verbatim.

        Returns whether anything was compacted. The system prefix is untouched,
        so the cache survives.
        """
        if len(self.messages) <= KEEP_RECENT + 2:
            return False

        cut = self._safe_cut(len(self.messages) - KEEP_RECENT)
        if cut <= 0:
            return False

        old, recent = self.messages[:cut], self.messages[cut:]
        transcript = _flatten(old)
        reply = await client.chat(
            [
                {
                    "role": "user",
                    "content": (
                        "Summarise this portion of a coding session for an agent that "
                        "will continue the work. Keep: files created or edited and what "
                        "changed in each, decisions made and why, what was tried and "
                        "failed, and anything still outstanding. Drop: full file "
                        "contents, full test output, and repeated reasoning.\n\n"
                        f"{transcript}"
                    ),
                }
            ],
            system="You compact agent transcripts without losing decisions or state.",
            max_tokens=2048,
        )
        self.usage = self.usage + reply.usage
        self.messages = [
            {
                "role": "user",
                "content": f"[Earlier in this session, summarised]\n{reply.text}",
            },
            {"role": "assistant", "content": "Understood — continuing from there."},
            *recent,
        ]
        self._compactions += 1
        await self.token_count(client, tools)
        return True

    def _safe_cut(self, index: int) -> int:
        """Move a cut point so it never separates a tool_use from its result.

        Orphaning either half is a 400 from the API: a tool_result whose
        tool_use is gone, or a tool_use with no result following it.
        """
        index = max(0, min(index, len(self.messages)))
        while index < len(self.messages) and _is_tool_result(self.messages[index]):
            index += 1
        return index


def _is_tool_result(message: dict[str, Any]) -> bool:
    content = message.get("content")
    return (
        message.get("role") == "user"
        and isinstance(content, list)
        and any(
            (b.get("type") if isinstance(b, dict) else getattr(b, "type", None))
            == "tool_result"
            for b in content
        )
    )


def _flatten(messages: list[dict[str, Any]], limit: int = 40_000) -> str:
    parts: list[str] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            parts.append(f"{m['role']}: {content}")
            continue
        for block in content or []:
            btype = (
                block.get("type") if isinstance(block, dict) else getattr(block, "type", "")
            )
            if btype == "text":
                text = (
                    block.get("text") if isinstance(block, dict) else getattr(block, "text", "")
                )
                parts.append(f"{m['role']}: {text}")
            elif btype == "tool_use":
                name = (
                    block.get("name") if isinstance(block, dict) else getattr(block, "name", "")
                )
                parts.append(f"{m['role']}: [called {name}]")
            elif btype == "tool_result":
                body = block.get("content", "") if isinstance(block, dict) else ""
                parts.append(f"tool: {str(body)[:500]}")
    return "\n".join(parts)[-limit:]
