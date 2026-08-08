"""In-process mock of the Cortex backend for Windows UI development.

Simulates the production Cortex Action semantics without ROS or a robot, so
the real Gradio UI can be exercised offline:

- feedback streams through ``on_feedback`` in the same shapes the real
  client produces (planning echo, step dispatch, waiting, completion);
- navigation-like prompts produce a fixed-count deterministic feedback
  sequence that can be accelerated for tests and cancelled mid-flight;
- ``cancel_active()`` aborts the simulated execution within ~0.1 s and the
  running ``execute()`` raises :class:`CortexRequestError`, mirroring the
  real "Plan aborted while waiting for async actions." outcome;
- only one request may be active, like the real client.

Select with ``UBROBOT_CHAT_BACKEND=cortex-mock`` or inject directly.
"""

from __future__ import annotations

import re
import threading

try:  # Package import for tests and `python -m chat_ui.app`.
    from .cortex_client import CortexBusyError, CortexRequestError
except ImportError:  # Script compatibility: direct module import.
    from cortex_client import CortexBusyError, CortexRequestError

NAV_PATTERN = re.compile(
    r"(走到|走向|导航|navigate|go to|move to|follow)", re.IGNORECASE
)
GRASP_PATTERN = re.compile(r"(抓取|抓住|拿起|grasp|pick up|pick)", re.IGNORECASE)


class MockCortexBackend:
    """Deterministic offline stand-in for the Cortex client."""

    def __init__(
        self,
        *,
        nav_duration_sec: float = 4.0,
        reply_delay_sec: float = 0.3,
        navigation_feedback_steps: int = 4,
    ):
        if nav_duration_sec <= 0 or reply_delay_sec < 0:
            raise ValueError("mock timings must be positive")
        if navigation_feedback_steps <= 0:
            raise ValueError("navigation_feedback_steps must be positive")
        self._nav_duration_sec = float(nav_duration_sec)
        self._reply_delay_sec = float(reply_delay_sec)
        self._navigation_feedback_steps = int(navigation_feedback_steps)
        self._lock = threading.Lock()
        self._active_cancel: threading.Event | None = None
        self.completed_actions = []
        self.requests = []

    # ------------------------------------------------------------------ API

    def execute(self, task: str, *, on_feedback) -> str:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be non-empty text")

        with self._lock:
            if self._active_cancel is not None:
                raise CortexBusyError("another Cortex request is already active")
            cancel_event = threading.Event()
            self._active_cancel = cancel_event
            self.requests.append(task.strip())

        try:
            on_feedback(f"Received task. Creating a plan for: {task}")
            if NAV_PATTERN.search(task) and GRASP_PATTERN.search(task):
                return self._run_sequence(task, on_feedback, cancel_event)
            if NAV_PATTERN.search(task):
                return self._run_navigation(task, on_feedback, cancel_event)
            if GRASP_PATTERN.search(task):
                return self._run_grasp(task, on_feedback, cancel_event)
            return self._run_text_only(task, on_feedback, cancel_event)
        finally:
            with self._lock:
                if self._active_cancel is cancel_event:
                    self._active_cancel = None

    def cancel_active(self) -> bool:
        with self._lock:
            event = self._active_cancel
        if event is None:
            return False
        event.set()
        return True

    def close(self) -> None:
        self.cancel_active()

    # -------------------------------------------------------------- phases

    def _run_navigation(self, task, on_feedback, cancel_event) -> str:
        on_feedback(
            "[Step 1/1 (send_goal_to__ubrobot_navigation_navigate_to_object)]"
            " -> EXECUTE"
        )
        step_delay = self._nav_duration_sec / self._navigation_feedback_steps
        for index in range(1, self._navigation_feedback_steps + 1):
            if cancel_event.wait(timeout=step_delay):
                on_feedback("Plan aborted while waiting for async actions.")
                raise CortexRequestError(
                    "Plan aborted while waiting for async actions."
                )
            on_feedback(
                "Step 1/1: waiting for async actions to complete... "
                f"({index}/{self._navigation_feedback_steps})"
            )
        on_feedback("All 1 steps completed.")
        self.completed_actions.append("navigation")
        return "All 1 steps completed."

    def _run_grasp(self, task, on_feedback, cancel_event):
        on_feedback(
            "[Step 1/1 (send_goal_to__ubrobot_manipulation_grasp_object)] -> EXECUTE"
        )
        for phase in ("approach", "align", "grasp", "retreat"):
            if cancel_event.wait(timeout=self._nav_duration_sec / 8.0):
                on_feedback("Plan aborted while waiting for async actions.")
                raise CortexRequestError(
                    "Plan aborted while waiting for async actions."
                )
            on_feedback(f"grasp phase: {phase}")
        self.completed_actions.append("grasp")
        on_feedback("All 1 steps completed.")
        return "All 1 steps completed."

    def _run_sequence(self, task, on_feedback, cancel_event):
        on_feedback(
            "[Step 1/2 (send_goal_to__ubrobot_navigation_navigate_to_object)]"
            " -> EXECUTE"
        )
        step_delay = self._nav_duration_sec / self._navigation_feedback_steps
        for index in range(1, self._navigation_feedback_steps + 1):
            if cancel_event.wait(timeout=step_delay):
                on_feedback("Plan aborted while waiting for async actions.")
                raise CortexRequestError(
                    "Plan aborted while waiting for async actions."
                )
            on_feedback(
                "Step 1/2: waiting for async actions to complete... "
                f"({index}/{self._navigation_feedback_steps})"
            )
        self.completed_actions.append("navigation")
        on_feedback(
            "[Step 2/2 (send_goal_to__ubrobot_manipulation_grasp_object)] -> EXECUTE"
        )
        for phase in ("approach", "align", "grasp", "retreat"):
            if cancel_event.wait(timeout=self._nav_duration_sec / 8.0):
                on_feedback("Plan aborted while waiting for async actions.")
                raise CortexRequestError(
                    "Plan aborted while waiting for async actions."
                )
            on_feedback(f"Step 2/2 grasp phase: {phase}")
        self.completed_actions.append("grasp")
        on_feedback("All 2 steps completed.")
        return "All 2 steps completed."

    def _run_text_only(self, task, on_feedback, cancel_event) -> str:
        if cancel_event.wait(timeout=self._reply_delay_sec):
            on_feedback("Plan aborted while waiting for async actions.")
            raise CortexRequestError("Plan aborted while waiting for async actions.")
        reply = f"[No actions needed]. 收到：{task}（离线开发模式，无真实规划）"
        on_feedback(reply)
        return reply
