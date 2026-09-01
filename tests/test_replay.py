"""Exporting a recorded session for the replay page."""

from __future__ import annotations

import json

import pytest

from codepilot.events import EventStream, EventType
from codepilot.replay import build, export
from codepilot.session import SessionStore


def _recorded(tmp_path) -> SessionStore:
    store = SessionStore.create(tmp_path, "sess1")
    stream = EventStream(session_id="sess1")
    store.attach(stream)
    stream.emit(EventType.USER_MESSAGE, "make mean() safe")
    stream.emit(EventType.TOOL_CALL, "read_file(path=stats.py)", tool="read_file")
    stream.emit(EventType.DIFF, "stats.py  +2 -1", diff="+guard\n-old", added=2, removed=1)
    stream.emit(EventType.COST, "$0.01", cost_usd=0.01, input_tokens=100)
    stream.emit(EventType.TURN_START, "internal bookkeeping")
    stream.emit(EventType.DONE, "done")
    store.record_turn("make mean() safe", "added a guard", "$0.01")
    return store


def test_export_keeps_the_events_a_viewer_can_follow(tmp_path):
    replay = build(_recorded(tmp_path))
    types = [e["type"] for e in replay.events]
    assert "user_message" in types and "diff" in types and "done" in types
    assert "turn_start" not in types, "internal bookkeeping should not be shown"


def test_costs_and_calls_are_totalled_from_the_events(tmp_path):
    replay = build(_recorded(tmp_path))
    assert replay.model_calls == 1
    assert replay.total_cost_usd == pytest.approx(0.01)


def test_timestamps_become_offsets_not_wall_clock(tmp_path):
    """A published recording should not date itself on every page load."""
    replay = build(_recorded(tmp_path))
    assert replay.events[0]["at_ms"] == 0
    assert all(e["at_ms"] >= 0 for e in replay.events)
    assert "at" not in replay.events[0]


def test_an_api_key_that_reached_the_log_is_redacted(tmp_path):
    """A recording gets published. Anything in it gets published too."""
    store = SessionStore.create(tmp_path, "leaky")
    stream = EventStream(session_id="leaky")
    store.attach(stream)
    leaked = "sk-ant-api03-" + "A" * 40
    stream.emit(EventType.TOOL_CALL, f"run_command(cmd=curl -H 'x: {leaked}')")
    stream.emit(EventType.TOOL_RESULT, "ok", output=f"echo {leaked}")
    store.record_turn(f"use {leaked}", "done", "$0")

    text = json.dumps(build(store).to_dict())
    assert leaked not in text
    assert "REDACTED" in text


def test_export_writes_a_manifest_the_page_can_read(tmp_path):
    store = _recorded(tmp_path)
    out = tmp_path / "web" / "replays"
    path = export(tmp_path, store.session_id, out, name="demo", title="A demo")

    assert path.name == "demo.json"
    manifest = json.loads((out / "index.json").read_text(encoding="utf-8"))
    assert manifest["replays"][0]["file"] == "demo.json"
    assert manifest["replays"][0]["title"] == "A demo"
    assert manifest["replays"][0]["events"] > 0


def test_exporting_an_empty_session_fails_loudly(tmp_path):
    SessionStore.create(tmp_path, "empty")
    with pytest.raises(ValueError, match="no events"):
        export(tmp_path, "empty", tmp_path / "out")


def test_the_page_states_that_there_is_no_live_agent(tmp_path):
    """The dead-demo rule: never imply a hosted thing that does not exist."""
    replay = build(_recorded(tmp_path)).to_dict()
    assert "not hosted" in replay["note"]
    assert replay["recorded"] is True
