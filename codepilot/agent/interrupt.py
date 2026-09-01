"""
Mid-run steering.

The one genuinely concurrent thing in this codebase, so it is deliberately the
smallest possible surface: a flag and a queue, set from anywhere, read by the
loop **between tool calls only**.

Never mid-tool. A tool is in the middle of writing a file, running a test, or
talking to git; interrupting there is how you get half-written state. The cost
of waiting is that an interrupt can land one tool call late, which is a far
better failure than a truncated write.
"""

from __future__ import annotations

import threading
from collections import deque


class InterruptChannel:
    """Thread-safe hand-off from a reader (or a UI) into the running loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: deque[str] = deque()
        self._stop = threading.Event()

    def send(self, message: str) -> None:
        """Queue a steer. Delivered before the loop's next model call."""
        with self._lock:
            self._pending.append(message)

    def stop(self) -> None:
        """Ask the loop to wind up after the current tool call."""
        self._stop.set()

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def clear_stop(self) -> None:
        self._stop.clear()

    def drain(self) -> list[str]:
        """Take everything queued. Called at the loop's checkpoint."""
        with self._lock:
            messages = list(self._pending)
            self._pending.clear()
        return messages

    @property
    def has_pending(self) -> bool:
        with self._lock:
            return bool(self._pending)
