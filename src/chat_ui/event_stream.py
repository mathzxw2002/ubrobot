"""Bounded, transport-neutral event stream for the Operator Console."""

from __future__ import annotations

import logging
import queue
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("ubrobot.operator_events")


@dataclass(frozen=True)
class EventEnvelope:
    event_id: int
    timestamp: datetime
    kind: str
    source: str
    correlation_id: str | None = None
    task_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["timestamp"] = self.timestamp.isoformat()
        return value


class EventSubscription:
    def __init__(
        self,
        stream: "EventStream",
        *,
        replay: list[EventEnvelope],
        replay_truncated: bool,
        queue_size: int,
    ):
        self._stream = stream
        self.replay = tuple(replay)
        self.replay_truncated = replay_truncated
        self._queue: queue.Queue[EventEnvelope] = queue.Queue(maxsize=queue_size)
        self._lock = threading.Lock()
        self._dropped = 0
        self._closed = False

    def get(self, timeout: float | None = None) -> EventEnvelope:
        return self._queue.get(timeout=timeout)

    def dropped_count(self, *, reset: bool = False) -> int:
        with self._lock:
            value = self._dropped
            if reset:
                self._dropped = 0
            return value

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._stream._unsubscribe(self)

    def _offer(self, event: EventEnvelope) -> None:
        with self._lock:
            if self._closed:
                return
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                self._dropped += 1
            self._queue.put_nowait(event)


class EventStream:
    """Thread-safe append-only history plus bounded subscriber mailboxes."""

    def __init__(self, *, max_history: int = 1000):
        if max_history <= 0:
            raise ValueError("max_history must be positive")
        self._history: deque[EventEnvelope] = deque(maxlen=max_history)
        self._subscribers: set[EventSubscription] = set()
        self._lock = threading.RLock()
        self._next_event_id = 1

    def publish(
        self,
        *,
        kind: str,
        source: str,
        payload: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        task_id: str | None = None,
    ) -> EventEnvelope:
        if not kind or not source:
            raise ValueError("event kind and source must be non-empty")
        with self._lock:
            event = EventEnvelope(
                event_id=self._next_event_id,
                timestamp=datetime.now(timezone.utc),
                kind=kind,
                source=source,
                correlation_id=correlation_id,
                task_id=task_id,
                payload=dict(payload or {}),
            )
            self._next_event_id += 1
            self._history.append(event)
            for subscriber in tuple(self._subscribers):
                subscriber._offer(event)
            log = (
                logger.debug
                if kind
                in {
                    "telemetry.updated",
                    "voice.microphone_level",
                    "voice.transcript.partial",
                    "task.feedback",
                }
                else logger.info
            )
            log(
                "event_id=%s kind=%s source=%s correlation_id=%s task_id=%s",
                event.event_id,
                event.kind,
                event.source,
                event.correlation_id or "none",
                event.task_id or "none",
            )
            return event

    def history(self, *, after_event_id: int = 0) -> list[EventEnvelope]:
        with self._lock:
            return [event for event in self._history if event.event_id > after_event_id]

    def latest_event_id(self) -> int:
        with self._lock:
            return self._next_event_id - 1

    def subscribe(
        self,
        *,
        after_event_id: int = 0,
        queue_size: int = 64,
    ) -> EventSubscription:
        if after_event_id < 0:
            raise ValueError("after_event_id must not be negative")
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        with self._lock:
            oldest = self._history[0].event_id if self._history else self._next_event_id
            replay_truncated = after_event_id > 0 and after_event_id < oldest - 1
            replay = [
                event for event in self._history if event.event_id > after_event_id
            ]
            subscription = EventSubscription(
                self,
                replay=replay,
                replay_truncated=replay_truncated,
                queue_size=queue_size,
            )
            self._subscribers.add(subscription)
            return subscription

    def _unsubscribe(self, subscription: EventSubscription) -> None:
        with self._lock:
            self._subscribers.discard(subscription)
