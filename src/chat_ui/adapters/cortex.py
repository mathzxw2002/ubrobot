"""Cortex transport contracts plus a fixture-only TaskBackend adapter.

Real ROS Action and Robot Edge clients belong in deployment-specific packages;
this workstation module deliberately contains no hardware or ROS imports.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable, Mapping
from uuid import uuid4


@dataclass(frozen=True)
class CortexCommand:
    text: str
    correlation_id: str = field(default_factory=lambda: uuid4().hex)
    requested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        value = asdict(self)
        value["requested_at"] = self.requested_at.isoformat()
        return value


@dataclass(frozen=True)
class CortexFeedback:
    correlation_id: str
    message: str
    sequence: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        value = asdict(self)
        value["timestamp"] = self.timestamp.isoformat()
        return value


@dataclass(frozen=True)
class CortexResult:
    correlation_id: str
    status: str
    reply: str
    finished_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        value = asdict(self)
        value["finished_at"] = self.finished_at.isoformat()
        return value


class FixtureCortexAdapter:
    """Deterministic, cancellable fixture implementing TaskRuntime's backend API."""

    hardware_authority = False

    def __init__(
        self,
        fixtures: Mapping[str, tuple[tuple[str, ...], str]] | None = None,
    ):
        self._fixtures = dict(fixtures or {})
        self._lock = threading.Lock()
        self._cancelled = threading.Event()
        self.requests: list[str] = []

    def execute(self, task: str, *, on_feedback: Callable[[str], None]) -> str:
        if not task.strip():
            raise ValueError("task must be non-empty")
        with self._lock:
            self._cancelled.clear()
            self.requests.append(task)
        feedback, reply = self._fixtures.get(
            task,
            (
                ("fixture planning", "fixture execution", "fixture complete"),
                "fixture done",
            ),
        )
        for message in feedback:
            if self._cancelled.is_set():
                raise RuntimeError("fixture execution cancelled")
            on_feedback(message)
        return reply

    def cancel_active(self) -> bool:
        self._cancelled.set()
        return True

    def emergency_stop(self) -> bool:
        self._cancelled.set()
        return True

    def close(self) -> None:
        self._cancelled.set()
