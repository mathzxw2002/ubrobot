"""Bounded event stream for Robot Edge."""

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

from ubrobot_contracts.edge_api import CommandEvent, CommandState


@dataclass
class EventRecord:
    """A single event in the stream with an ID."""

    event_id: int
    event: CommandEvent


class EventStream:
    """Bounded event stream with monotonic IDs and cursor replay.

    The monotonic ``event_id`` is also stamped onto ``CommandEvent.sequence``
    so clients can track a cursor from the serialized event payload alone.
    """

    def __init__(self, max_history: int = 1000) -> None:
        if max_history <= 0:
            raise ValueError("max_history must be positive")
        self._max_history = max_history
        self._events: deque[EventRecord] = deque(maxlen=max_history)
        self._next_id = 1
        self._lock = threading.Lock()

    def append(
        self,
        command_id: str,
        state: CommandState,
        message: str = "",
        payload: dict[str, Any] | None = None,
    ) -> int:
        """Append an event to the stream and return its ID."""
        with self._lock:
            event_id = self._next_id
            event = CommandEvent(
                command_id=command_id,
                state=state,
                message=message,
                payload=payload or {},
                sequence=event_id,
            )
            self._events.append(EventRecord(event_id=event_id, event=event))
            self._next_id += 1
            return event_id

    def get_since(self, event_id: int) -> list[EventRecord]:
        """Get all events with ID > event_id."""
        with self._lock:
            return [record for record in self._events if record.event_id > event_id]

    def latest_id(self) -> int:
        """Return the latest event ID, or 0 if empty."""
        with self._lock:
            return self._events[-1].event_id if self._events else 0

    def get_latest(self) -> EventRecord | None:
        """Get the latest event, if any."""
        with self._lock:
            return self._events[-1] if self._events else None

    def __len__(self) -> int:
        """Return number of events in history."""
        with self._lock:
            return len(self._events)
