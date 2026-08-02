"""Independent text/voice interaction channel above :mod:`task_runtime`."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import re
import threading
from typing import Any, Callable
from uuid import uuid4

try:
    from .task_runtime import TaskRuntime
except ImportError:  # Direct-script compatibility.
    from task_runtime import TaskRuntime


class InteractionCategory(str, Enum):
    QUERY = "query"
    CONTROL = "control"
    EMERGENCY_STOP = "emergency_stop"
    NEW_TASK = "new_task"


@dataclass(frozen=True)
class InteractionTurn:
    turn_id: str
    session_id: str
    text: str
    source: str
    category: InteractionCategory
    related_task_id: str | None
    confidence: float
    created_at: datetime

    def to_dict(self):
        value = asdict(self)
        value["category"] = self.category.value
        value["created_at"] = self.created_at.isoformat()
        return value


@dataclass(frozen=True)
class InteractionResult:
    turn: InteractionTurn
    reply: str
    dispatched: bool
    task_id: str | None = None


_CANCEL_RE = re.compile(
    r"^\s*(停止|停下|停一下|取消(?:任务)?|急停|stop|cancel|emergency stop)[!！。.]?\s*$",
    re.IGNORECASE,
)
_STATUS_RE = re.compile(
    r"(现在到哪|任务状态|进度|执行到哪|状态怎么样|what.*status|status|progress)",
    re.IGNORECASE,
)
_EMERGENCY_STOP_RE = re.compile(
    r"^\s*(?:\u7d27\u6025(?:\u505c\u6b62|\u53eb\u505c)(?:\u673a\u5668\u4eba)?|"
    r"\u6025\u505c(?:\u673a\u5668\u4eba)?|emergency\s+stop)"
    r"[!\uff01\u3002\uff1f?]?\s*$",
    re.IGNORECASE,
)


class InteractionRuntime:
    """Classify interaction turns without coupling ASR to task execution."""

    def __init__(
        self,
        task_runtime: TaskRuntime,
        *,
        max_turns: int = 200,
        event_publisher: Callable[..., Any] | None = None,
    ):
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        self._task_runtime = task_runtime
        self._max_turns = max_turns
        self._turns: list[InteractionTurn] = []
        self._lock = threading.Lock()
        self._event_publisher = event_publisher

    def handle(
        self,
        text: str,
        *,
        source: str = "text",
        session_id: str = "operator-console",
        on_feedback: Callable[[str], None] | None = None,
        correlation_id: str | None = None,
    ) -> InteractionResult:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("interaction text must be non-empty")
        normalized = text.strip()
        category = self.classify(normalized)
        active = self._task_runtime.active_task()
        turn = InteractionTurn(
            turn_id=correlation_id or uuid4().hex,
            session_id=session_id,
            text=normalized,
            source=source,
            category=category,
            related_task_id=active.task_id if active is not None else None,
            confidence=1.0,
            created_at=datetime.now(timezone.utc),
        )
        self._remember(turn)
        self._publish(
            kind="interaction.received",
            source=source,
            correlation_id=turn.turn_id,
            task_id=turn.related_task_id,
            payload={"text": normalized, "category": category.value},
        )

        if category == InteractionCategory.QUERY:
            if active is None:
                return self._finish(turn, "当前没有正在执行的主任务。", False)
            return self._finish(
                turn,
                f"当前任务：{active.intent}；状态：{active.status.value}。",
                False,
                active.task_id,
            )
        if category == InteractionCategory.CONTROL:
            if active is None:
                return self._finish(turn, "当前没有可取消的主任务。", False)
            acknowledged = self._task_runtime.cancel_active()
            reply = "已请求取消当前任务。" if acknowledged else "取消请求已发送，等待执行端确认。"
            return self._finish(turn, reply, False, active.task_id)
        if category == InteractionCategory.EMERGENCY_STOP:
            acknowledged = self._task_runtime.emergency_stop(
                source=source,
                reason=normalized,
                correlation_id=turn.turn_id,
            )
            reply = (
                "紧急停止已确认。"
                if acknowledged
                else "紧急停止命令已发出，执行端尚未确认。"
            )
            return self._finish(
                turn,
                reply,
                False,
                active.task_id if active is not None else None,
            )

        try:
            execution = self._task_runtime.execute(
                normalized,
                on_feedback=on_feedback,
                correlation_id=turn.turn_id,
            )
            return self._finish(
                turn,
                execution.reply,
                execution.dispatched,
                execution.task.task_id,
            )
        except Exception as exc:
            self._publish(
                kind="interaction.failed",
                source=source,
                correlation_id=turn.turn_id,
                task_id=turn.related_task_id,
                payload={"error_type": type(exc).__name__, "message": str(exc)},
            )
            raise

    @staticmethod
    def classify(text: str) -> InteractionCategory:
        if _EMERGENCY_STOP_RE.search(text):
            return InteractionCategory.EMERGENCY_STOP
        if _CANCEL_RE.search(text):
            return InteractionCategory.CONTROL
        if _STATUS_RE.search(text):
            return InteractionCategory.QUERY
        return InteractionCategory.NEW_TASK

    def turns(self) -> list[InteractionTurn]:
        with self._lock:
            return list(self._turns)

    def _remember(self, turn: InteractionTurn) -> None:
        with self._lock:
            self._turns.append(turn)
            if len(self._turns) > self._max_turns:
                del self._turns[: len(self._turns) - self._max_turns]

    def _finish(
        self,
        turn: InteractionTurn,
        reply: str,
        dispatched: bool,
        task_id: str | None = None,
    ) -> InteractionResult:
        result = InteractionResult(turn, reply, dispatched, task_id)
        self._publish(
            kind="interaction.completed",
            source=turn.source,
            correlation_id=turn.turn_id,
            task_id=task_id or turn.related_task_id,
            payload={
                "category": turn.category.value,
                "reply": reply,
                "dispatched": dispatched,
            },
        )
        return result

    def _publish(self, **event: Any) -> None:
        if self._event_publisher is not None:
            self._event_publisher(**event)
