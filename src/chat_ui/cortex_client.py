"""UI-facing client for the EMOS Cortex command Action."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from typing import Callable, Protocol


FeedbackCallback = Callable[[str], None]


@dataclass(frozen=True)
class CortexResult:
    success: bool
    text: str


class CortexRequestError(RuntimeError):
    """The Cortex request completed without a successful result."""


class CortexBusyError(CortexRequestError):
    """A second request was attempted while one goal was active."""


class CortexGoal(Protocol):
    def wait(self, timeout_sec: float) -> CortexResult: ...

    def cancel(self, timeout_sec: float) -> bool: ...


class CortexTransport(Protocol):
    def send(self, task: str, on_feedback: FeedbackCallback) -> CortexGoal: ...


_PENDING_GOAL = object()


class CortexClient:
    """Serialize UI requests and expose bounded cancellation to the Stop path."""

    def __init__(
        self,
        transport: CortexTransport,
        *,
        result_timeout_sec: float = 180.0,
        cancel_timeout_sec: float = 2.0,
    ):
        self._transport = transport
        self._result_timeout_sec = _positive_finite(
            "result_timeout_sec", result_timeout_sec
        )
        self._cancel_timeout_sec = _positive_finite(
            "cancel_timeout_sec", cancel_timeout_sec
        )
        self._lock = threading.Lock()
        self._active_goal: CortexGoal | object | None = None

    def execute(
        self,
        task: str,
        *,
        on_feedback: FeedbackCallback | None = None,
    ) -> str:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Cortex task must be non-empty text")

        with self._lock:
            if self._active_goal is not None:
                raise CortexBusyError("another Cortex request is already active")
            self._active_goal = _PENDING_GOAL

        goal = None
        last_feedback = ""

        def forward_feedback(text: str) -> None:
            nonlocal last_feedback
            last_feedback = text
            if on_feedback is not None:
                on_feedback(text)

        try:
            goal = self._transport.send(task, forward_feedback)
            with self._lock:
                self._active_goal = goal
            result = goal.wait(self._result_timeout_sec)
            final_text = result.text or last_feedback
            if not result.success:
                raise CortexRequestError(final_text or "Cortex request failed")
            return final_text
        finally:
            with self._lock:
                if self._active_goal is goal or self._active_goal is _PENDING_GOAL:
                    self._active_goal = None

    def cancel_active(self) -> bool:
        with self._lock:
            goal = self._active_goal
        if goal is None or goal is _PENDING_GOAL:
            return False
        return bool(goal.cancel(self._cancel_timeout_sec))


def _positive_finite(name: str, value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return number
