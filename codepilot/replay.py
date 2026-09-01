"""
Turning a real session into a demo.

The agent cannot be hosted: it executes code, and exposing that to the internet
means running arbitrary commands for strangers. So the public artefact is a
recording — the same event stream the CLI renders, replayed at its original
pacing from a committed JSON file.

Nothing is reconstructed or embellished. The exporter reads a session's JSONL,
keeps the events a viewer can follow, and drops the API key if one ever reached
the log. What the page shows is what happened.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from codepilot.events import Event, EventType
from codepilot.session import SessionStore

#: Events worth showing. Bookkeeping the viewer cannot act on is dropped.
VISIBLE = {
    EventType.USER_MESSAGE,
    EventType.ASSISTANT_TEXT,
    EventType.PLAN,
    EventType.TOOL_CALL,
    EventType.TOOL_RESULT,
    EventType.DIFF,
    EventType.TEST_RESULT,
    EventType.CHECKPOINT,
    EventType.PERMISSION_DECISION,
    EventType.INTERRUPT,
    EventType.COST,
    EventType.BUDGET,
    EventType.ERROR,
    EventType.DONE,
}

#: Anything shaped like a key, wherever it appears. A recording is published;
#: a key that reaches it is published too.
_SECRET = re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}")


def _scrub(value):
    if isinstance(value, str):
        return _SECRET.sub("sk-ant-***REDACTED***", value)
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    return value


@dataclass
class Replay:
    title: str
    events: list[dict]
    turns: list[dict]
    total_cost_usd: float
    model_calls: int
    duration_seconds: float

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "recorded": True,
            "note": (
                "A recording of a real local session. Nothing here is simulated. "
                "The agent is not hosted: it runs commands and edits files, so it "
                "runs on your machine, not on a server."
            ),
            "total_cost_usd": round(self.total_cost_usd, 4),
            "model_calls": self.model_calls,
            "duration_seconds": round(self.duration_seconds, 1),
            "turns": self.turns,
            "events": self.events,
        }


def build(store: SessionStore, title: str = "") -> Replay:
    events: list[Event] = store.events()
    if not events:
        raise ValueError(f"session {store.session_id} has no events to replay")

    kept: list[dict] = []
    first = events[0].at
    cost = 0.0
    calls = 0
    for event in events:
        if event.type not in VISIBLE:
            continue
        if event.type is EventType.COST:
            calls += 1
            cost += float(event.data.get("cost_usd") or 0.0)
        kept.append(
            {
                "seq": event.seq,
                "turn": event.turn,
                "type": event.type.value,
                "message": _scrub(event.message),
                "data": _scrub(event.data),
                # Offset rather than wall-clock: the viewer replays pacing, and
                # a timestamp would date the recording on every page load.
                "at_ms": int((event.at - first).total_seconds() * 1000),
            }
        )

    turns = [
        {"task": _scrub(t.get("task", "")), "summary": _scrub(t.get("summary", ""))}
        for t in store.turns()
    ]
    duration = (events[-1].at - first).total_seconds()
    return Replay(
        title=title or (turns[0]["task"][:80] if turns else store.session_id),
        events=kept,
        turns=turns,
        total_cost_usd=cost,
        model_calls=calls,
        duration_seconds=duration,
    )


def export(root: Path, session_id: str, out_dir: Path, name: str = "", title: str = "") -> Path:
    store = SessionStore(root=Path(root), session_id=session_id)
    replay = build(store, title=title)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name or session_id}.json"
    path.write_text(json.dumps(replay.to_dict(), indent=2), encoding="utf-8")
    _write_index(out_dir)
    return path


def _write_index(out_dir: Path) -> None:
    """A manifest of what is available, so the page needs no server."""
    entries = []
    for path in sorted(out_dir.glob("*.json")):
        if path.name == "index.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        entries.append(
            {
                "file": path.name,
                "title": data.get("title", path.stem),
                "cost_usd": data.get("total_cost_usd"),
                "model_calls": data.get("model_calls"),
                "duration_seconds": data.get("duration_seconds"),
                "events": len(data.get("events", [])),
            }
        )
    (out_dir / "index.json").write_text(
        json.dumps({"replays": entries}, indent=2), encoding="utf-8"
    )
