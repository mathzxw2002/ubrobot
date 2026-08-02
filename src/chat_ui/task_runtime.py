"""Task lifecycle and serialization for semantic robot capabilities.

The runtime deliberately knows nothing about Gradio, ROS, or robot hardware.
It serializes commands with physical side effects through one backend and
records an event stream that can later be transported to a remote console.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
import threading
from typing import Any, Callable, Protocol
from uuid import uuid4


class TaskStatus(str, Enum):
    QUEUED = "queued"
    PLANNING = "planning"
    RUNNING = "running"
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"


TERMINAL_STATUSES = {
    TaskStatus.SUCCEEDED,
    TaskStatus.FAILED,
    TaskStatus.CANCELLED,
    TaskStatus.SUPERSEDED,
}


class TaskBackend(Protocol):
    def execute(self, task: str, *, on_feedback: Callable[[str], None]) -> str: ...

    def cancel_active(self) -> bool: ...


@dataclass
class TaskRecord:
    task_id: str
    intent: str
    correlation_id: str = ""
    capability: str = "cortex"
    parameters: dict[str, Any] = field(default_factory=dict)
    parent_task_id: str | None = None
    root_task_id: str = ""
    sequence_no: int = 0
    dependencies: list[str] = field(default_factory=list)
    priority: int = 0
    status: TaskStatus = TaskStatus.QUEUED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    cancel_requested: bool = False

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        for key in ("created_at", "started_at", "finished_at"):
            value = result[key]
            result[key] = value.isoformat() if value is not None else None
        return result


@dataclass(frozen=True)
class TaskEvent:
    event_id: str
    task_id: str
    event_type: str
    message: str
    timestamp: datetime
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result


@dataclass(frozen=True)
class TaskExecution:
    task: TaskRecord
    reply: str
    dispatched: bool


class TaskRuntime:
    """Own one active Cortex task while retaining queued/task-tree metadata."""

    def __init__(
        self,
        backend: TaskBackend,
        *,
        max_events: int = 500,
        event_publisher: Callable[..., Any] | None = None,
    ):
        if max_events <= 0:
            raise ValueError("max_events must be positive")
        self._backend = backend
        self._lock = threading.RLock()
        self._tasks: dict[str, TaskRecord] = {}
        self._pending: deque[str] = deque()
        self._events: deque[TaskEvent] = deque(maxlen=max_events)
        self._active_task_id: str | None = None
        self._event_publisher = event_publisher

    def submit(
        self,
        intent: str,
        *,
        capability: str = "cortex",
        parameters: dict[str, Any] | None = None,
        parent_task_id: str | None = None,
        sequence_no: int = 0,
        dependencies: list[str] | None = None,
        priority: int = 0,
        correlation_id: str | None = None,
    ) -> TaskRecord:
        if not isinstance(intent, str) or not intent.strip():
            raise ValueError("task intent must be non-empty text")
        with self._lock:
            if parent_task_id is not None and parent_task_id not in self._tasks:
                raise KeyError(f"unknown parent task: {parent_task_id}")
            task_id = uuid4().hex
            root_id = (
                self._tasks[parent_task_id].root_task_id
                if parent_task_id is not None
                else task_id
            )
            task = TaskRecord(
                task_id=task_id,
                root_task_id=root_id,
                intent=intent.strip(),
                correlation_id=correlation_id or uuid4().hex,
                capability=capability,
                parameters=dict(parameters or {}),
                parent_task_id=parent_task_id,
                sequence_no=sequence_no,
                dependencies=list(dependencies or []),
                priority=priority,
            )
            self._tasks[task_id] = task
            self._pending.append(task_id)
            self._append_event(task, "task.queued", "Task queued")
            return replace(task)

    def execute(
        self,
        intent: str,
        *,
        on_feedback: Callable[[str], None] | None = None,
        **task_fields: Any,
    ) -> TaskExecution:
        task = self.submit(intent, **task_fields)
        with self._lock:
            if self._active_task_id is not None:
                self._append_event(
                    self._tasks[task.task_id],
                    "task.waiting",
                    "Waiting for the active motion task",
                )
                return TaskExecution(
                    replace(self._tasks[task.task_id]),
                    "已有任务正在执行；新任务已进入待处理队列。",
                    False,
                )
            self._active_task_id = task.task_id
            self._pending.remove(task.task_id)
            current = self._tasks[task.task_id]
            current.status = TaskStatus.PLANNING
            current.started_at = datetime.now(timezone.utc)
            self._append_event(current, "task.started", "Task planning started")

        def feedback(text: str) -> None:
            message = str(text)
            with self._lock:
                current = self._tasks[task.task_id]
                if current.status == TaskStatus.PLANNING:
                    current.status = TaskStatus.RUNNING
                self._append_event(current, "task.feedback", message)
            if on_feedback is not None:
                on_feedback(message)

        try:
            reply = self._backend.execute(current.intent, on_feedback=feedback)
            with self._lock:
                current = self._tasks[task.task_id]
                current.finished_at = datetime.now(timezone.utc)
                if current.cancel_requested:
                    current.status = TaskStatus.CANCELLED
                    current.result = None
                    self._append_event(current, "task.cancelled", "Task cancelled")
                else:
                    current.status = TaskStatus.SUCCEEDED
                    current.result = {"text": reply}
                    self._append_event(current, "task.succeeded", reply or "Task succeeded")
            return TaskExecution(replace(current), reply or "", True)
        except Exception as exc:
            with self._lock:
                current = self._tasks[task.task_id]
                current.finished_at = datetime.now(timezone.utc)
                current.status = (
                    TaskStatus.CANCELLED
                    if current.cancel_requested
                    else TaskStatus.FAILED
                )
                current.error = {"type": type(exc).__name__, "message": str(exc)}
                event_type = (
                    "task.cancelled"
                    if current.status == TaskStatus.CANCELLED
                    else "task.failed"
                )
                self._append_event(current, event_type, str(exc))
            raise
        finally:
            with self._lock:
                if self._active_task_id == task.task_id:
                    self._active_task_id = None

    def cancel_active(self) -> bool:
        with self._lock:
            task = (
                self._tasks.get(self._active_task_id)
                if self._active_task_id is not None
                else None
            )
            if task is not None and task.status not in TERMINAL_STATUSES:
                task.cancel_requested = True
                task.status = TaskStatus.CANCELLING
                self._append_event(task, "task.cancelling", "Cancellation requested")
        # Keep the backend call outside the lock: ROS cancellation may block.
        acknowledged = bool(self._backend.cancel_active())
        return acknowledged

    def emergency_stop(
        self,
        *,
        source: str,
        reason: str = "Emergency stop",
        correlation_id: str | None = None,
    ) -> bool:
        """Request an immediate backend stop independently of task planning.

        The backend may expose a stronger ``emergency_stop`` primitive. Older
        backends remain compatible and fall back to ``cancel_active``.
        """
        with self._lock:
            safety_correlation_id = correlation_id or uuid4().hex
            task = (
                self._tasks.get(self._active_task_id)
                if self._active_task_id is not None
                else None
            )
            event_task_id = task.task_id if task is not None else "safety"
            if task is not None and task.status not in TERMINAL_STATUSES:
                task.cancel_requested = True
                task.status = TaskStatus.CANCELLING
            superseded_task_ids = []
            while self._pending:
                pending_id = self._pending.popleft()
                pending = self._tasks[pending_id]
                pending.status = TaskStatus.SUPERSEDED
                pending.finished_at = datetime.now(timezone.utc)
                superseded_task_ids.append(pending_id)
                self._append_event(
                    pending,
                    "task.superseded",
                    "Task superseded by emergency stop",
                    {"safety_correlation_id": safety_correlation_id},
                )
            event = TaskEvent(
                event_id=uuid4().hex,
                task_id=event_task_id,
                event_type="safety.emergency_stop",
                message=reason,
                timestamp=datetime.now(timezone.utc),
                data={
                    "source": source,
                    "priority": "critical",
                    "bypass_queue": True,
                    "superseded_task_ids": superseded_task_ids,
                },
            )
            self._events.append(event)
            self._publish_event(
                kind=event.event_type,
                source="task_runtime",
                task_id=event_task_id,
                correlation_id=safety_correlation_id,
                payload={
                    "message": reason,
                    "source": source,
                    "priority": "critical",
                    "bypass_queue": True,
                    "superseded_task_ids": superseded_task_ids,
                },
            )

        stop = getattr(self._backend, "emergency_stop", None)
        if stop is not None:
            return bool(stop())
        return bool(self._backend.cancel_active())

    def active_task(self) -> TaskRecord | None:
        with self._lock:
            task = self._tasks.get(self._active_task_id or "")
            return replace(task) if task is not None else None

    def task(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return replace(task) if task is not None else None

    def tasks(self) -> list[TaskRecord]:
        with self._lock:
            return [replace(task) for task in self._tasks.values()]

    def pending_tasks(self) -> list[TaskRecord]:
        with self._lock:
            return [replace(self._tasks[task_id]) for task_id in self._pending]

    def events(self, *, task_id: str | None = None) -> list[TaskEvent]:
        with self._lock:
            events = list(self._events)
        if task_id is not None:
            events = [event for event in events if event.task_id == task_id]
        return events

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            active = self._tasks.get(self._active_task_id or "")
            return {
                "active_task": active.to_dict() if active is not None else None,
                "pending_tasks": [
                    self._tasks[task_id].to_dict() for task_id in self._pending
                ],
                "tasks": [task.to_dict() for task in self._tasks.values()],
                "events": [event.to_dict() for event in self._events],
            }

    def _append_event(
        self,
        task: TaskRecord,
        event_type: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        event = TaskEvent(
            event_id=uuid4().hex,
            task_id=task.task_id,
            event_type=event_type,
            message=message,
            timestamp=datetime.now(timezone.utc),
            data=dict(data or {}),
        )
        self._events.append(event)
        self._publish_event(
            kind=event_type,
            source="task_runtime",
            correlation_id=task.correlation_id,
            task_id=task.task_id,
            payload={
                "message": message,
                "status": task.status.value,
                "intent": task.intent,
                "data": dict(data or {}),
            },
        )

    def _publish_event(self, **event: Any) -> None:
        if self._event_publisher is not None:
            self._event_publisher(**event)
