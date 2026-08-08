"""UI-facing client for the EMOS Cortex command Action."""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
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


class CortexUnavailableError(CortexRequestError):
    """The Cortex ROS Action server was not available before its deadline."""


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

    def close(self) -> None:
        self.cancel_active()
        close = getattr(self._transport, "close", None)
        if close is not None:
            close()


class _FeedbackState:
    def __init__(self):
        self._lock = threading.Lock()
        self._last = ""
        self._completed = ""

    def update(self, text: str, completed: bool) -> None:
        with self._lock:
            self._last = text
            if completed:
                self._completed = text

    def final_text(self) -> str:
        with self._lock:
            return self._completed or self._last


class _RosCortexGoal:
    def __init__(self, goal_handle, feedback: _FeedbackState):
        self._goal_handle = goal_handle
        self._feedback = feedback

    def wait(self, timeout_sec: float) -> CortexResult:
        wrapped = _wait_future(
            self._goal_handle.get_result_async(),
            timeout_sec,
            "Cortex result",
        )
        return CortexResult(
            success=bool(wrapped.result.success),
            text=self._feedback.final_text(),
        )

    def cancel(self, timeout_sec: float) -> bool:
        response = _wait_future(
            self._goal_handle.cancel_goal_async(),
            timeout_sec,
            "Cortex cancellation",
        )
        return bool(response.goals_canceling)


class RosCortexTransport:
    """Own a private ROS context and Action client for the desktop UI."""

    def __init__(
        self,
        *,
        action_name: str = "/cortex_input_command",
        server_timeout_sec: float = 5.0,
        shutdown_timeout_sec: float = 2.0,
        bindings=None,
    ):
        self._server_timeout_sec = _positive_finite(
            "server_timeout_sec", server_timeout_sec
        )
        self._shutdown_timeout_sec = _positive_finite(
            "shutdown_timeout_sec", shutdown_timeout_sec
        )
        self._bindings = bindings or _load_ros_bindings()
        self._context = self._bindings.Context()
        self._bindings.init(context=self._context)
        self._node = self._bindings.Node(
            "ubrobot_chat_cortex_client",
            context=self._context,
        )
        self._executor = self._bindings.Executor(context=self._context)
        self._executor.add_node(self._node)
        self._action_client = self._bindings.ActionClient(
            self._node,
            self._bindings.ActionType,
            action_name,
        )
        self._closed = False
        self._close_lock = threading.Lock()
        self._spin_thread = threading.Thread(
            target=self._executor.spin,
            name="cortex-ros-executor",
            daemon=True,
        )
        self._spin_thread.start()

    def send(self, task: str, on_feedback: FeedbackCallback) -> CortexGoal:
        if self._closed:
            raise CortexUnavailableError("Cortex ROS transport is closed")
        if not self._action_client.wait_for_server(
            timeout_sec=self._server_timeout_sec
        ):
            raise CortexUnavailableError(
                f"Cortex Action server unavailable after "
                f"{self._server_timeout_sec:.3f}s"
            )

        feedback_state = _FeedbackState()

        def feedback_callback(message) -> None:
            feedback = message.feedback
            text = str(feedback.feedback)
            feedback_state.update(text, bool(feedback.completed))
            on_feedback(text)

        request = self._bindings.ActionType.Goal()
        request.task = task
        goal_handle = _wait_future(
            self._action_client.send_goal_async(
                request,
                feedback_callback=feedback_callback,
            ),
            self._server_timeout_sec,
            "Cortex goal response",
        )
        if not goal_handle.accepted:
            raise CortexRequestError("Cortex Action goal was rejected")
        return _RosCortexGoal(goal_handle, feedback_state)

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(timeout_sec=self._shutdown_timeout_sec)
        self._spin_thread.join(self._shutdown_timeout_sec)
        self._action_client.destroy()
        self._node.destroy_node()
        self._bindings.shutdown(context=self._context)


def create_ros_cortex_client(*, bindings=None) -> CortexClient:
    """Build the production client from non-secret runtime configuration."""
    transport = RosCortexTransport(
        action_name=os.environ.get("CORTEX_ACTION_NAME", "/cortex_input_command"),
        server_timeout_sec=float(os.environ.get("CORTEX_SERVER_TIMEOUT_SEC", "5")),
        bindings=bindings,
    )
    return CortexClient(
        transport,
        result_timeout_sec=float(os.environ.get("CORTEX_RESULT_TIMEOUT_SEC", "180")),
        cancel_timeout_sec=float(os.environ.get("CORTEX_CANCEL_TIMEOUT_SEC", "2")),
    )


def _wait_future(future, timeout_sec: float, description: str):
    deadline = time.monotonic() + _positive_finite("timeout_sec", timeout_sec)
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not future.done():
        raise TimeoutError(f"timed out waiting for {description}")
    exception = future.exception()
    if exception is not None:
        raise exception
    return future.result()


def _load_ros_bindings():
    # Imports remain lazy so UI contract tests work on non-ROS workstations.
    import rclpy
    from automatika_embodied_agents.action import VisionLanguageAction
    from rclpy.action import ActionClient
    from rclpy.context import Context
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node

    return SimpleNamespace(
        ActionType=VisionLanguageAction,
        ActionClient=ActionClient,
        Context=Context,
        Node=Node,
        Executor=MultiThreadedExecutor,
        init=rclpy.init,
        shutdown=rclpy.shutdown,
    )


def _positive_finite(name: str, value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return number
