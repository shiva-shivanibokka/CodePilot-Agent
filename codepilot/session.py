"""
Session persistence.

Append-only JSONL, one record per line, written as the session runs rather
than at the end — so a crash, a Ctrl-C, or a closed laptop lid leaves a
readable session instead of nothing.

The same file is three things: what `--resume` reads to continue a
conversation, what the replay exporter turns into a demo, and what you read
afterwards to find out what the agent actually did.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from codepilot.events import Event, EventStream

SESSIONS_DIR = ".codepilot/sessions"


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class SessionStore:
    """One session's file. Records append; nothing is ever rewritten."""

    root: Path
    session_id: str

    @classmethod
    def create(cls, root: Path, session_id: str | None = None) -> SessionStore:
        store = cls(root=Path(root), session_id=session_id or uuid.uuid4().hex[:12])
        store.path.parent.mkdir(parents=True, exist_ok=True)
        if not store.path.exists():
            store._append({"kind": "meta", "session_id": store.session_id, "at": _now()})
        return store

    @property
    def path(self) -> Path:
        return self.root / SESSIONS_DIR / f"{self.session_id}.jsonl"

    def _append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    # -- writing -------------------------------------------------------------

    def record_event(self, event: Event) -> None:
        self._append({"kind": "event", "event": json.loads(event.model_dump_json())})

    def record_messages(self, messages: list[dict[str, Any]]) -> None:
        """Snapshot the conversation after a turn.

        A snapshot rather than a delta: the conversation is rewritten wholesale
        by compaction, so replaying deltas would rebuild a history that no
        longer matches what the model was actually sent.
        """
        self._append({"kind": "messages", "at": _now(), "messages": messages})

    def record_turn(self, task: str, summary: str, budget: str) -> None:
        self._append(
            {"kind": "turn", "at": _now(), "task": task, "summary": summary, "budget": budget}
        )

    def attach(self, stream: EventStream) -> None:
        """Persist every event as it is emitted."""
        stream.subscribe(self.record_event)

    # -- reading -------------------------------------------------------------

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A half-written final line is what a crash looks like. The rest
                # of the session is still perfectly good.
                continue
        return records

    def events(self) -> list[Event]:
        return [
            Event.model_validate(r["event"]) for r in self.read() if r.get("kind") == "event"
        ]

    def messages(self) -> list[dict[str, Any]]:
        """The most recent conversation snapshot, or [] for a fresh session."""
        snapshots = [r for r in self.read() if r.get("kind") == "messages"]
        return snapshots[-1]["messages"] if snapshots else []

    def turns(self) -> list[dict[str, Any]]:
        return [r for r in self.read() if r.get("kind") == "turn"]


def list_sessions(root: Path) -> list[tuple[str, datetime, int]]:
    """(id, modified, turn count), newest first."""
    directory = Path(root) / SESSIONS_DIR
    if not directory.is_dir():
        return []
    out = []
    for path in directory.glob("*.jsonl"):
        store = SessionStore(root=Path(root), session_id=path.stem)
        out.append(
            (
                path.stem,
                datetime.fromtimestamp(path.stat().st_mtime, UTC),
                len(store.turns()),
            )
        )
    return sorted(out, key=lambda row: row[1], reverse=True)
