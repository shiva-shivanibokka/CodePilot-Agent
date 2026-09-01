"""The conversation: cache prefix placement and compaction."""

from __future__ import annotations

import pytest

from codepilot.context import KEEP_RECENT, Conversation
from codepilot.llm import Reply, Usage


class StubClient:
    """Counts tokens by length and returns a fixed summary. No network."""

    def __init__(self, token_count: int = 10) -> None:
        self.token_count = token_count
        self.summarised: list[str] = []

    async def count_tokens(self, messages, system=None, tools=None) -> int:
        return self.token_count

    async def chat(self, messages, system=None, max_tokens=2048, **kw) -> Reply:
        self.summarised.append(messages[0]["content"])
        return Reply(
            text="SUMMARY: wrote fib.py, tests pass, conftest.py still missing.",
            content=[],
            tool_calls=[],
            stop_reason="end_turn",
            model="stub",
            usage=Usage(input_tokens=100, output_tokens=20),
            latency_ms=1,
            cost_usd=0.0,
        )


# ------------------------------------------------------------- cache prefix


def test_the_cache_breakpoint_sits_on_the_last_stable_block():
    convo = Conversation(system_prompt="SYS", project_instructions="PROJ")
    blocks = convo.system_blocks()
    assert len(blocks) == 2
    assert "cache_control" not in blocks[0], "breakpoint must be on the LAST block only"
    assert blocks[-1]["cache_control"] == {"type": "ephemeral"}


def test_the_prefix_is_byte_identical_across_turns():
    """Any drift here silently invalidates the cache for every following turn."""
    convo = Conversation(system_prompt="SYS", project_instructions="PROJ")
    first = convo.system_blocks()
    convo.user("do a thing")
    convo.user("do another thing")
    assert convo.system_blocks() == first


def test_project_instructions_are_optional():
    blocks = Conversation(system_prompt="SYS").system_blocks()
    assert len(blocks) == 1
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}


# ------------------------------------------------------------- tool results


def test_a_failed_tool_is_a_result_not_an_exception():
    block = Conversation.tool_result("tu_1", "No such file: x.py", is_error=True)
    assert block["is_error"] is True
    assert block["tool_use_id"] == "tu_1"


def test_all_results_for_one_turn_go_in_a_single_message():
    """Splitting them teaches the model to stop calling tools in parallel."""
    convo = Conversation(system_prompt="SYS")
    convo.tool_results(
        [
            Conversation.tool_result("tu_1", "ok"),
            Conversation.tool_result("tu_2", "boom", is_error=True),
        ]
    )
    assert len(convo.messages) == 1
    assert len(convo.messages[0]["content"]) == 2


# --------------------------------------------------------------- compaction


@pytest.mark.asyncio
async def test_compaction_keeps_the_recent_turns_and_summarises_the_rest():
    convo = Conversation(system_prompt="SYS")
    for i in range(40):
        convo.user(f"message {i}")
    client = StubClient()

    assert await convo.compact(client) is True

    assert len(convo.messages) == KEEP_RECENT + 2, "summary + ack + the recent window"
    assert "summarised" in convo.messages[0]["content"]
    assert convo.messages[-1]["content"] == "message 39", "newest turn must survive"


def _tool_conversation(pairs: int = 30) -> Conversation:
    convo = Conversation(system_prompt="SYS")
    for i in range(pairs):
        convo.assistant(
            [{"type": "tool_use", "id": f"tu_{i}", "name": "read_file", "input": {}}]
        )
        convo.tool_results([Conversation.tool_result(f"tu_{i}", f"contents {i}")])
    return convo


def test_a_cut_landing_on_a_tool_result_is_moved_past_it():
    """A tool_result whose tool_use was compacted away is a 400 from the API.

    Asserted on _safe_cut directly: cutting at the natural KEEP_RECENT offset
    happens to land on an assistant message, so a whole-compaction test would
    pass without ever exercising this guard.
    """
    convo = _tool_conversation()
    on_a_result = 4  # 0=tool_use, 1=tool_result, 2=tool_use, 3=tool_result...
    assert convo.messages[on_a_result + 1]["role"] == "user"
    assert convo._safe_cut(on_a_result + 1) == on_a_result + 2, (
        "cut point was left on a tool_result, orphaning it from its tool_use"
    )


def test_a_cut_on_an_assistant_message_is_left_where_it_is():
    convo = _tool_conversation()
    assert convo._safe_cut(4) == 4


@pytest.mark.asyncio
async def test_compaction_of_a_tool_heavy_session_leaves_no_dangling_result():
    convo = _tool_conversation()
    await convo.compact(StubClient())

    window = convo.messages[2:]  # skip the summary + acknowledgement pair
    assert window, "compaction removed everything"
    first = window[0]
    is_orphan = (
        first["role"] == "user"
        and isinstance(first["content"], list)
        and any(b.get("type") == "tool_result" for b in first["content"])
    )
    assert not is_orphan, "window starts with a tool_result whose tool_use is gone"


@pytest.mark.asyncio
async def test_a_short_conversation_is_left_alone():
    convo = Conversation(system_prompt="SYS")
    convo.user("one")
    convo.user("two")
    assert await convo.compact(StubClient()) is False
    assert len(convo.messages) == 2
