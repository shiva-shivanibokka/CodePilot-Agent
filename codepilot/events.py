"""
The event stream.

Both agent arms emit the same sequence of events, and three things consume it:
the CLI renders it live, the recorder writes it to JSONL, and the replay site
plays that JSONL back. Keeping one stream is what makes recording a demo free
and what makes the two arms comparable — if they reported differently, any
comparison between them would be measuring the reporting.

Nothing here imports the agent, the sandbox, or a provider SDK.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    # lifecycle
    SESSION_START = "session_start"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    # conversation
    USER_MESSAGE = "user_message"
    ASSISTANT_TEXT = "assistant_text"
    THINKING = "thinking"
    INTERRUPT = "interrupt"
    # work
    PLAN = "plan"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    DIFF = "diff"
    TEST_RESULT = "test_result"
    # control
    PERMISSION_REQUEST = "permission_request"
    PERMISSION_DECISION = "permission_decision"
    CHECKPOINT = "checkpoint"
    COST = "cost"
    BUDGET = "budget"
    ERROR = "error"
    DONE = "done"


#: Rendered before the message in a terminal. ASCII fallbacks live in the CLI —
#: a Windows console in cp1252 raises UnicodeEncodeError on these.
ICONS: dict[EventType, str] = {
    EventType.SESSION_START: "◆",
    EventType.TURN_START: "▸",
    EventType.TURN_END: "◂",
    EventType.USER_MESSAGE: "❯",
    EventType.ASSISTANT_TEXT: "·",
    EventType.THINKING: "🤔",
    EventType.INTERRUPT: "✋",
    EventType.PLAN: "📝",
    EventType.TOOL_CALL: "🔧",
    EventType.TOOL_RESULT: "📋",
    EventType.DIFF: "±",
    EventType.TEST_RESULT: "🧪",
    EventType.PERMISSION_REQUEST: "🔐",
    EventType.PERMISSION_DECISION: "🔓",
    EventType.CHECKPOINT: "⎘",
    EventType.COST: "💰",
    EventType.BUDGET: "⛔",
    EventType.ERROR: "❌",
    EventType.DONE: "✅",
}

ASCII_ICONS: dict[EventType, str] = {
    EventType.THINKING: "*",
    EventType.INTERRUPT: "!",
    EventType.PLAN: "#",
    EventType.TOOL_CALL: ">",
    EventType.TOOL_RESULT: "<",
    EventType.TEST_RESULT: "T",
    EventType.PERMISSION_REQUEST: "?",
    EventType.PERMISSION_DECISION: "=",
    EventType.CHECKPOINT: "^",
    EventType.COST: "$",
    EventType.BUDGET: "X",
    EventType.ERROR: "X",
    EventType.DONE: "v",
}


def supports_unicode(stream: Any = None) -> bool:
    """Whether this stream can encode the icon set.

    A Windows console defaults to cp1252, where writing an emoji raises
    UnicodeEncodeError mid-render — which kills a run that had otherwise
    succeeded. Checked once, not per line.
    """
    stream = stream or sys.stdout
    encoding = getattr(stream, "encoding", None) or ""
    try:
        "🤔".encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True


class Event(BaseModel):
    """One thing that happened. Serialisable, replayable, provider-agnostic."""

    seq: int = 0
    session_id: str = ""
    turn: int = 0
    type: EventType
    message: str = ""
    #: Structured detail: tool arguments, diffs, token counts, test output.
    data: dict[str, Any] = Field(default_factory=dict)
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def render(self, unicode_ok: bool = True) -> str:
        icon = (
            ICONS.get(self.type, "•")
            if unicode_ok
            else ASCII_ICONS.get(self.type, "-")
        )
        return f"{icon} {self.message}"


class EventStream:
    """Ordered, append-only log with live subscribers.

    Subscribers are called synchronously as events arrive, which is what makes
    streaming real rather than a replay of a finished run. A subscriber that
    raises must not take the agent down with it — rendering is not the work.
    """

    def __init__(self, session_id: str = "") -> None:
        self.session_id = session_id
        self.events: list[Event] = []
        self.turn = 0
        self._subscribers: list[Callable[[Event], None]] = []

    def subscribe(self, fn: Callable[[Event], None]) -> Callable[[], None]:
        """Register a live consumer. Returns a function that unsubscribes it."""
        self._subscribers.append(fn)
        return lambda: self._subscribers.remove(fn)

    def emit(
        self,
        type: EventType,
        message: str = "",
        **data: Any,
    ) -> Event:
        event = Event(
            seq=len(self.events),
            session_id=self.session_id,
            turn=self.turn,
            type=type,
            message=message,
            data=data,
        )
        self.events.append(event)
        for fn in list(self._subscribers):
            try:
                fn(event)
            except Exception:  # noqa: BLE001 - a broken renderer must not stop work
                pass
        return event

    def to_jsonl(self) -> str:
        """The recorder's format, and the replay site's input."""
        return "\n".join(e.model_dump_json() for e in self.events)

    @classmethod
    def from_jsonl(cls, text: str, session_id: str = "") -> EventStream:
        stream = cls(session_id=session_id)
        stream.events = [
            Event.model_validate_json(line) for line in text.splitlines() if line.strip()
        ]
        if stream.events:
            stream.turn = stream.events[-1].turn
        return stream
