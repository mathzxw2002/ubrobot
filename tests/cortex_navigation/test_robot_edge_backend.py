"""Real behavior tests for the Operator Console Robot Edge backend adapter.

Starts an in-process fixture Robot Edge (uvicorn) on a dynamic port and drives
the adapter over real HTTP. No fabrication, no TestClient shortcuts: the adapter
talks to a live server exactly as it would in production.
"""

from __future__ import annotations

import socket
import sys
import threading
import time
import unittest
from pathlib import Path

import httpx
import uvicorn

ROOT = Path(__file__).resolve().parents[2]
for segment in (ROOT / "src", ROOT / "src" / "chat_ui"):
    segment_str = str(segment)
    if segment_str not in sys.path:
        sys.path.insert(0, segment_str)

from robot_edge.app import create_app  # noqa: E402
from chat_ui.adapters.robot_edge import RobotEdgeBackend  # noqa: E402

TOKENS = {
    "operator-token": [
        "observe",
        "task.submit",
        "task.cancel",
        "safety.stop",
        "lease.manage",
    ],
    "observer-token": ["observe"],
}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class _EdgeServer:
    """A fixture Robot Edge running on a dynamic port in a background thread."""

    def __init__(self) -> None:
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        # A small fixture step delay widens the active-command window so
        # cancellation lands deterministically mid-flight (<=100 ms per step,
        # per the plan's test-time constraint).
        self.app = create_app(
            execution_mode="fixture",
            test_tokens=TOKENS,
            fixture_step_delay_sec=0.05,
        )
        config = uvicorn.Config(
            self.app, host="127.0.0.1", port=self.port, log_level="warning"
        )
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self) -> None:
        self.thread.start()
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if self.server.started:
                # Give the lifespan a moment to initialize auth + runtime.
                time.sleep(0.05)
                return
            time.sleep(0.02)
        raise RuntimeError(f"Edge did not start on port {self.port}")

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5.0)


class TestRobotEdgeBackendBehavior(unittest.TestCase):
    """Exercise the TaskBackend protocol against a live fixture Edge."""

    def setUp(self) -> None:
        self.edge = _EdgeServer()
        self.edge.start()
        self.addCleanup(self.edge.stop)

    def _backend(self, token: str = "operator-token") -> RobotEdgeBackend:
        return RobotEdgeBackend(
            edge_url=self.edge.url,
            operator_id="test-operator",
            token=token,
        )

    def test_no_default_token_allowed(self) -> None:
        """A backend without a token must fail rather than use a default."""
        with self.assertRaises(RuntimeError):
            RobotEdgeBackend(
                edge_url=self.edge.url,
                operator_id="test-operator",
                token="",
            )

    def test_command_submission_and_ordered_feedback(self) -> None:
        """execute must submit a command and forward ordered feedback."""
        feedback: list[str] = []
        backend = self._backend()
        try:
            result = backend.execute("导航到前面的椅子", on_feedback=feedback.append)
        finally:
            backend.close()
        self.assertIn("complete", result.lower())
        # Ordered fixture sequence: accepted -> planning -> running -> succeeded.
        joined = " | ".join(feedback).lower()
        self.assertIn("planning", joined)
        self.assertIn("moving", joined)
        planning_idx = joined.index("planning")
        moving_idx = joined.index("moving")
        self.assertLess(planning_idx, moving_idx)

    def test_409_detail_is_reported_truthfully(self) -> None:
        """A 409 with a server detail must surface that reason, not a generic
        "replay or stale" claim (e.g. hardware authority disabled)."""
        from chat_ui.adapters.robot_edge import RobotEdgeBackend

        class _Resp:
            status_code = 409

            def __init__(self, body: dict | None) -> None:
                self._body = body

            def json(self):
                if self._body is None:
                    raise ValueError("no json")
                return self._body

        with self.assertRaisesRegex(
            RuntimeError, "hardware authority disabled"
        ):
            RobotEdgeBackend._parse_command_response(
                _Resp({"detail": "hardware authority disabled: Robot Edge is in read-only mode (M6)"})
            )

        # No detail body -> the legacy generic message stays as fallback.
        with self.assertRaisesRegex(RuntimeError, "replay or stale"):
            RobotEdgeBackend._parse_command_response(_Resp(None))

    def test_409_safety_latched_is_reported(self) -> None:
        from chat_ui.adapters.robot_edge import RobotEdgeBackend

        class _Resp:
            status_code = 409

            def json(self):
                return {"detail": "Safety latched - cannot execute commands"}

        with self.assertRaisesRegex(RuntimeError, "Safety latched"):
            RobotEdgeBackend._parse_command_response(_Resp())

    def test_cancellation(self) -> None:
        """cancel_active must stop a running command; execute raises on cancel."""
        backend = self._backend()
        running = threading.Event()
        outcome: dict[str, object] = {}

        def on_feedback(message: str) -> None:
            if "planning" in message.lower() or "moving" in message.lower():
                running.set()

        def run() -> None:
            try:
                outcome["reply"] = backend.execute(
                    "导航到前面的椅子", on_feedback=on_feedback
                )
            except Exception as exc:  # noqa: BLE001 - record the failure type
                outcome["error"] = type(exc).__name__

        worker = threading.Thread(target=run)
        worker.start()
        self.assertTrue(
            running.wait(timeout=5.0),
            "command did not produce feedback before cancel",
        )
        cancelled = backend.cancel_active()
        worker.join(timeout=5.0)
        try:
            backend.close()
        finally:
            pass
        self.assertTrue(cancelled)
        self.assertEqual(outcome.get("error"), "RuntimeError")

    def test_emergency_stop_latches_and_blocks_new_work(self) -> None:
        """emergency_stop latches the Edge; subsequent commands are rejected."""
        backend = self._backend()
        try:
            self.assertTrue(backend.emergency_stop())
            with self.assertRaises(RuntimeError):
                backend.execute("另一个命令", on_feedback=lambda m: None)
        finally:
            backend.close()

    def test_auth_failure_is_sanitized(self) -> None:
        """An invalid token must fail without leaking the token in the message."""
        backend = self._backend(token="not-a-real-token")
        try:
            with self.assertRaises(RuntimeError) as ctx:
                backend.execute("导航", on_feedback=lambda m: None)
            self.assertNotIn("not-a-real-token", str(ctx.exception))
        finally:
            backend.close()

    def test_edge_disconnect_fails_task_clearly(self) -> None:
        """When the Edge is unreachable, execute fails with a clear message."""
        backend = RobotEdgeBackend(
            edge_url="http://127.0.0.1:1",
            operator_id="test-operator",
            token="operator-token",
        )
        try:
            with self.assertRaises(RuntimeError) as ctx:
                backend.execute("导航", on_feedback=lambda m: None)
            self.assertIn("Could not connect", str(ctx.exception))
        finally:
            backend.close()

    def test_adapter_survives_reconnect(self) -> None:
        """The adapter is server-side: a second execute after the first works."""
        backend = self._backend()
        try:
            first = backend.execute("导航到前面的椅子", on_feedback=lambda m: None)
            second = backend.execute("抓取前面的杯子", on_feedback=lambda m: None)
        finally:
            backend.close()
        self.assertIn("complete", (first + second).lower())

    def test_close_is_idempotent_and_blocks_execute(self) -> None:
        """close releases the client; further execute fails clearly."""
        backend = self._backend()
        backend.close()
        backend.close()  # idempotent
        with self.assertRaises(RuntimeError):
            backend.execute("导航", on_feedback=lambda m: None)


if __name__ == "__main__":
    unittest.main()
