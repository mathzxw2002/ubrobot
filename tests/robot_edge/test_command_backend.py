"""Tests for the Robot Edge Cortex command backend (M8).

Workstation-safe: the rclpy-backed client is never constructed; tests inject
a fake client factory that simulates goal acceptance, feedback streaming,
terminal results, and cancellation.
"""

from __future__ import annotations

import queue
import threading
import unittest

from ubrobot_contracts.edge_api import CommandState

from robot_edge.ros.backend import RosCortexCommandBackend, _classify_feedback


class _FakeGraph:
    """Minimal RosGraph stand-in (telemetry/capabilities not exercised)."""

    def __init__(self) -> None:
        self.shutdown_called = False

    def has_topic(self, topic: str) -> bool:
        return False

    def read_topic(self, topic: str):
        return None

    def has_action_server(self, action_name: str) -> bool:
        return False

    def shutdown(self) -> None:
        self.shutdown_called = True


class _FakeCortexClient:
    """Simulates the Cortex action client contract used by the backend."""

    def __init__(
        self,
        *,
        feedback: list[str] | None = None,
        terminal: dict | None = None,
        raise_on_send: Exception | None = None,
        hold: bool = False,
    ) -> None:
        self.feedback = list(feedback or [])
        self.terminal = terminal or {"status": "succeeded", "message": "All 1 steps completed."}
        self.raise_on_send = raise_on_send
        self.hold = hold
        self.release_event = threading.Event()
        self.sent_task: str | None = None
        self.cancelled = False
        self.shutdown_called = False
        self.goal_handle = None

    def send_goal(
        self,
        task: str,
        *,
        feedback_callback,
        terminal_callback,
    ) -> None:
        self.sent_task = task
        if self.raise_on_send is not None:
            raise self.raise_on_send
        for message in self.feedback:
            feedback_callback(message)
        if self.hold:
            # Wait until released (or cancelled) before finishing.
            self.release_event.wait(timeout=10.0)
            if self.cancelled:
                terminal_callback(status="cancelled", message="Command cancelled")
                return
        terminal_callback(**self.terminal)

    def cancel_goal_async(self) -> None:
        self.cancelled = True

    def shutdown(self) -> None:
        self.shutdown_called = True


class TestClassifyFeedback(unittest.TestCase):
    def test_planning_hints(self) -> None:
        self.assertEqual(
            _classify_feedback("Received task. Creating a plan for: x"),
            CommandState.PLANNING,
        )
        self.assertEqual(
            _classify_feedback("Plan: send_goal_to_..."),
            CommandState.PLANNING,
        )

    def test_running_hints_and_default(self) -> None:
        self.assertEqual(
            _classify_feedback("Executing Step 1/1 ..."),
            CommandState.RUNNING,
        )
        self.assertEqual(_classify_feedback("未知反馈"), CommandState.RUNNING)


class TestCortexCommandBackend(unittest.TestCase):
    def _backend(self, client: _FakeCortexClient) -> RosCortexCommandBackend:
        return RosCortexCommandBackend(
            _FakeGraph(), client_factory=lambda: client
        )

    def _collect(self, backend, text: str) -> list[tuple[CommandState, str, dict]]:
        return list(backend.get_command_sequence(text))

    def test_accepted_then_succeeded_with_feedback(self) -> None:
        client = _FakeCortexClient(
            feedback=[
                "Received task. Creating a plan for: 请走到椅子旁边",
                "Plan: send_goal_to__ubrobot_navigation_navigate_to_object",
                "Executing Step 1/1 ... dispatched",
            ]
        )
        backend = self._backend(client)
        events = self._collect(backend, "请走到椅子旁边")
        self.assertEqual(client.sent_task, "请走到椅子旁边")
        states = [state for state, _, _ in events]
        self.assertEqual(states[0], CommandState.ACCEPTED)
        self.assertIn(CommandState.PLANNING, states)
        self.assertIn(CommandState.RUNNING, states)
        self.assertEqual(states[-1], CommandState.SUCCEEDED)
        backend.close()

    def test_failed_terminal(self) -> None:
        client = _FakeCortexClient(
            terminal={"status": "failed", "message": "Cortex status 5"}
        )
        backend = self._backend(client)
        events = self._collect(backend, "x")
        self.assertEqual(events[-1][0], CommandState.FAILED)
        self.assertIn("Cortex status 5", events[-1][1])
        backend.close()

    def test_cancelled_terminal(self) -> None:
        client = _FakeCortexClient(
            terminal={"status": "cancelled", "message": "Command cancelled"}
        )
        backend = self._backend(client)
        events = self._collect(backend, "x")
        self.assertEqual(events[-1][0], CommandState.CANCELLED)
        backend.close()

    def test_client_error_reports_failed(self) -> None:
        client = _FakeCortexClient(raise_on_send=RuntimeError("boom"))
        backend = self._backend(client)
        events = self._collect(backend, "x")
        self.assertEqual(events[-1][0], CommandState.FAILED)
        self.assertIn("boom", events[-1][1])
        backend.close()

    def test_cancel_active_unblocks_generator(self) -> None:
        """cancel_active from another thread terminates the sequence."""
        client = _FakeCortexClient(
            feedback=["Executing Step 1/1 ..."], hold=True
        )
        backend = self._backend(client)

        generator = backend.get_command_sequence("请走到椅子旁边")
        self.assertEqual(next(generator)[0], CommandState.ACCEPTED)
        self.assertEqual(next(generator)[0], CommandState.RUNNING)

        # The generator now blocks on the queue; cancel from another thread.
        def run_cancel() -> None:
            backend.cancel_active()

        thread = threading.Thread(target=run_cancel)
        thread.start()
        state, _, _ = next(generator)
        thread.join(timeout=5.0)
        self.assertEqual(state, CommandState.CANCELLED)
        with self.assertRaises(StopIteration):
            next(generator)
        backend.close()

    def test_authority_and_mode(self) -> None:
        backend = self._backend(_FakeCortexClient())
        self.assertTrue(backend.hardware_authority)
        self.assertEqual(backend.execution_mode, "hardware")
        backend.close()

    def test_close_shuts_down_client_and_graph(self) -> None:
        client = _FakeCortexClient()
        graph = _FakeGraph()
        backend = RosCortexCommandBackend(graph, client_factory=lambda: client)
        self._collect(backend, "x")  # trigger the client factory
        backend.close()
        self.assertTrue(client.shutdown_called)
        self.assertTrue(graph.shutdown_called)


if __name__ == "__main__":
    unittest.main()
