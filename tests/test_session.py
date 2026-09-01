"""Session persistence and resume."""

from __future__ import annotations

import json
import subprocess

import pytest
from pydantic import BaseModel

from codepilot.context import Conversation
from codepilot.events import EventStream, EventType
from codepilot.session import SessionStore, list_sessions
from codepilot.workspace import Workspace


class FakeTextBlock(BaseModel):
    """Stands in for an SDK content block: a pydantic model, not a dict."""

    type: str = "text"
    text: str = ""
    citations: None = None


def test_events_are_persisted_as_they_are_emitted(tmp_path):
    store = SessionStore.create(tmp_path)
    stream = EventStream(session_id=store.session_id)
    store.attach(stream)
    stream.emit(EventType.TOOL_CALL, "read_file(path=a.py)")
    stream.emit(EventType.DONE, "finished")

    # Written during the run, not at the end: a crash must still leave a record.
    assert [e.message for e in store.events()] == [
        "read_file(path=a.py)",
        "finished",
    ]


def test_sdk_content_blocks_survive_the_round_trip(tmp_path):
    """json.dumps(default=str) turns a block object into its repr, so a resumed
    conversation would carry "TextBlock(...)" where a message belongs."""
    convo = Conversation(system_prompt="S")
    convo.user("do a thing")
    convo.assistant([FakeTextBlock(text="I did the thing")])

    store = SessionStore.create(tmp_path)
    store.record_messages(convo.messages)
    restored = store.messages()

    assert restored[1]["role"] == "assistant"
    block = restored[1]["content"][0]
    assert isinstance(block, dict), f"block came back as {type(block).__name__}"
    assert block["text"] == "I did the thing"
    assert "TextBlock(" not in json.dumps(restored)


def test_tool_use_blocks_keep_their_ids(tmp_path):
    """A tool_result refers to its tool_use by id; losing it is a 400."""

    class FakeToolUse(BaseModel):
        type: str = "tool_use"
        id: str = "tu_7"
        name: str = "read_file"
        input: dict = {"path": "a.py"}

    convo = Conversation(system_prompt="S")
    convo.assistant([FakeToolUse()])
    convo.tool_results([Conversation.tool_result("tu_7", "contents")])

    store = SessionStore.create(tmp_path)
    store.record_messages(convo.messages)
    restored = store.messages()
    assert restored[0]["content"][0]["id"] == "tu_7"
    assert restored[1]["content"][0]["tool_use_id"] == "tu_7"


def test_resume_reloads_the_latest_snapshot(tmp_path):
    store = SessionStore.create(tmp_path)
    store.record_messages([{"role": "user", "content": "first turn"}])
    store.record_messages(
        [{"role": "user", "content": "first turn"}, {"role": "user", "content": "second"}]
    )
    reopened = SessionStore.create(tmp_path, store.session_id)
    assert len(reopened.messages()) == 2, "resume took an older snapshot"


def test_a_truncated_final_line_does_not_lose_the_session(tmp_path):
    """A crash mid-write leaves half a line. The rest is still good."""
    store = SessionStore.create(tmp_path)
    store.record_turn("task", "summary", "$0.01")
    with store.path.open("a", encoding="utf-8") as fh:
        fh.write('{"kind": "event", "eve')
    assert len(store.turns()) == 1


def test_sessions_are_listed_newest_first(tmp_path):
    first = SessionStore.create(tmp_path, "aaa")
    first.record_turn("t1", "s", "$0")
    second = SessionStore.create(tmp_path, "bbb")
    second.record_turn("t2", "s", "$0")
    rows = list_sessions(tmp_path)
    assert [r[0] for r in rows][0] == "bbb"
    assert dict((r[0], r[2]) for r in rows) == {"aaa": 1, "bbb": 1}


@pytest.mark.asyncio
async def test_undo_does_not_truncate_the_session_log(tmp_path):
    """The checkpoint predates the log's contents, so restoring the worktree
    wholesale rewinds the record of the turn being undone."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@e.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")

    store = SessionStore.create(tmp_path)  # writes the meta line only
    ws = Workspace(root=tmp_path, session_id=store.session_id)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)
    ws.checkpoint("before")

    store.record_turn("the task", "what happened", "$0.05")
    ws.read("a.py")
    ws.write("a.py", "x = 2\n")

    ws.undo()

    assert (tmp_path / "a.py").read_text() == "x = 1\n", "the edit was not reverted"
    assert len(store.turns()) == 1, "undo erased the log of the turn it undid"
