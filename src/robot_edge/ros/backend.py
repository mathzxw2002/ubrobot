"""ROS backends for Robot Edge (M6 read-only, M8 command).

The read-only backend (M6) serves capability/telemetry snapshots from the
real ROS graph and rejects every command. The command backend (M8) forwards
operator commands to the Cortex action (``/cortex_input_command``) so the
full chain frontend -> Robot Edge -> Cortex/ARK -> Kompass -> guard -> wheels
works through the Edge's auth/lease/safety layers.
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Callable, Iterator, Optional

from ubrobot_contracts.capabilities import CapabilityName, CapabilitySnapshot
from ubrobot_contracts.edge_api import CommandState
from ubrobot_contracts.telemetry import TelemetryChannel, TelemetrySnapshot

from robot_edge.ros.actions import RosActionInventory
from robot_edge.ros.context import RosGraph, create_ros_context
from robot_edge.ros.telemetry import RosTelemetryReader

# Cortex action the Edge forwards operator commands to.
CORTEX_ACTION_NAME = "/cortex_input_command"

# Feedback text -> command-state classification (best effort; unknown text
# stays RUNNING).
_FEEDBACK_STATE_HINTS: tuple[tuple[tuple[str, ...], CommandState], ...] = (
    (("creating a plan", "plan:"), CommandState.PLANNING),
    (("executing step", "dispatched"), CommandState.RUNNING),
    (("no actions needed",), CommandState.RUNNING),
)


class RosReadonlyBackend:
    """Robot Edge backend backed by a read-only ROS graph."""

    execution_mode = "hardware"
    hardware_authority = False

    def __init__(self, graph: RosGraph) -> None:
        self._graph = graph
        self._telemetry = RosTelemetryReader(graph)
        self._actions = RosActionInventory(graph)

    def get_capabilities(self) -> dict[CapabilityName, CapabilitySnapshot]:
        return self._actions.capabilities()

    def get_telemetry_snapshot(self) -> dict[TelemetryChannel, TelemetrySnapshot]:
        return self._telemetry.snapshot()

    def get_command_sequence(
        self, command_text: str
    ) -> Iterator[tuple[CommandState, str, dict[str, Any]]]:
        # M6: read-only. No command may start; the runtime maps this to a
        # rejected submit (409) without touching ROS execution.
        raise RuntimeError(
            "hardware authority disabled: Robot Edge is in read-only mode (M6)"
        )

    def close(self) -> None:
        self._graph.shutdown()


def _classify_feedback(message: str) -> CommandState:
    lowered = message.lower()
    for hints, state in _FEEDBACK_STATE_HINTS:
        if any(hint in lowered for hint in hints):
            return state
    return CommandState.RUNNING


class RosCortexCommandBackend:
    """Hardware backend that forwards commands to the Cortex action (M8).

    ``hardware_authority=True``: submitting a command starts a
    ``/cortex_input_command`` goal in a background thread and streams the
    Cortex feedback back as a command sequence. Cancellation and emergency
    stop cancel the downstream goal.

    The rclpy-dependent parts are created lazily by ``_make_client_factory``
    so workstation tests can inject a fake factory; the real factory imports
    rclpy only when hardware mode with authority is requested.
    """

    execution_mode = "hardware"
    hardware_authority = True

    def __init__(
        self,
        graph: RosGraph,
        *,
        action_name: str = CORTEX_ACTION_NAME,
        client_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._graph = graph
        self._telemetry = RosTelemetryReader(graph)
        self._actions = RosActionInventory(graph)
        self._action_name = action_name
        self._client_factory = client_factory or self._default_client_factory
        self._events: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._goal_thread: Optional[threading.Thread] = None
        self._active_client: Any = None
        self._closed = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ API

    def get_capabilities(self) -> dict[CapabilityName, CapabilitySnapshot]:
        return self._actions.capabilities()

    def get_telemetry_snapshot(self) -> dict[TelemetryChannel, TelemetrySnapshot]:
        return self._telemetry.snapshot()

    def get_command_sequence(
        self, command_text: str
    ) -> Iterator[tuple[CommandState, str, dict[str, Any]]]:
        if self._closed:
            raise RuntimeError("Robot Edge command backend is closed")
        # Fresh queue per command so stale events (e.g. a "cancelled" from
        # the previous command) cannot leak into the new command's stream.
        self._events = queue.Queue()
        yield CommandState.ACCEPTED, "Command accepted", {"source": "cortex"}
        self._start_goal(command_text)
        while True:
            item = self._events.get()
            kind = item.get("kind")
            if kind == "feedback":
                message = str(item.get("message", ""))
                yield _classify_feedback(message), message, {"source": "cortex"}
            elif kind == "terminal":
                status = item.get("status", "failed")
                message = str(item.get("message", "") or "")
                payload: dict[str, Any] = {"source": "cortex"}
                raw_status = item.get("raw_status")
                if raw_status is not None:
                    payload["raw_ros_status"] = raw_status
                if status == "succeeded":
                    yield CommandState.SUCCEEDED, message or "Task complete!", payload
                elif status == "cancelled":
                    yield CommandState.CANCELLED, message or "Command cancelled", payload
                else:
                    yield CommandState.FAILED, message or "Command failed", payload
                return

    def cancel_active(self) -> bool:
        """Cancel the in-flight Cortex goal; unblocks the command generator."""
        client = self._active_client
        handle = getattr(client, "goal_handle", None)
        if handle is not None:
            try:
                handle.cancel_goal_async()
            except Exception:
                pass
        self._events.put({"kind": "terminal", "status": "cancelled",
                          "message": "Command cancelled by operator"})
        return True

    def close(self) -> None:
        self._closed = True
        self.cancel_active()
        client = self._active_client
        if client is not None and hasattr(client, "shutdown"):
            try:
                client.shutdown()
            except Exception:
                pass
        thread = self._goal_thread
        if thread is not None:
            thread.join(timeout=5.0)
        self._graph.shutdown()

    # ------------------------------------------------------------- internal

    def _start_goal(self, command_text: str) -> None:
        thread = threading.Thread(
            target=self._run_goal,
            args=(command_text,),
            name="robot-edge-cortex-goal",
            daemon=True,
        )
        self._goal_thread = thread
        thread.start()

    def _run_goal(self, command_text: str) -> None:
        # Capture the per-command queue so late callbacks from a previous
        # command cannot pollute the current command's event stream.
        events = self._events
        try:
            client = self._client_factory()
            with self._lock:
                self._active_client = client
            client.send_goal(
                command_text,
                feedback_callback=lambda msg: events.put(
                    {"kind": "feedback", "message": msg}
                ),
                terminal_callback=lambda **kw: events.put(
                    {"kind": "terminal", **kw}
                ),
            )
        except Exception as exc:  # noqa: BLE001 - report any goal failure
            events.put(
                {"kind": "terminal", "status": "failed", "message": str(exc)}
            )

    def _on_feedback(self, message: str) -> None:
        self._events.put({"kind": "feedback", "message": message})

    def _on_terminal(self, *, status: str, message: str, raw_status: Optional[int] = None) -> None:
        self._events.put(
            {"kind": "terminal", "status": status, "message": message,
             "raw_status": raw_status}
        )

    def _default_client_factory(self) -> Any:
        """Build the real rclpy Cortex action client (hardware side only).

        The client spins on its OWN SingleThreadedExecutor in a dedicated
        thread, so goal waiting never contends with the telemetry graph's
        global-spin reads.
        """
        import rclpy  # noqa: PLC0415 - hardware-only import
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node

        from automatika_embodied_agents.action import VisionLanguageAction

        if not rclpy.ok():
            rclpy.init(args=[])
        node = Node("robot_edge_command")
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        class _CortexClient:
            """Small rclpy bridge: send goal, stream feedback, report result."""

            def __init__(self) -> None:
                from rclpy.action import ActionClient

                self._node = node
                self._executor = executor
                self._spin_thread = spin_thread
                self._action_type = VisionLanguageAction
                self._action_client = ActionClient(
                    node, VisionLanguageAction, CORTEX_ACTION_NAME
                )
                self.goal_handle = None

            def _wait(self, future: Any, *, timeout_sec: float) -> bool:
                """Wait for a future on the dedicated executor thread."""
                import time

                deadline = time.monotonic() + timeout_sec
                while not future.done() and time.monotonic() < deadline:
                    time.sleep(0.05)
                return future.done()

            def send_goal(
                self,
                task: str,
                *,
                feedback_callback: Callable[[str], None],
                terminal_callback: Callable[..., None],
            ) -> None:
                if not self._action_client.wait_for_server(timeout_sec=10.0):
                    terminal_callback(status="failed", message="Cortex action server unavailable")
                    return
                goal = VisionLanguageAction.Goal()
                goal.task = task

                def on_feedback(message) -> None:
                    text = str(message.feedback.feedback)
                    feedback_callback(text)

                send_future = self._action_client.send_goal_async(
                    goal, feedback_callback=on_feedback
                )
                if not self._wait(send_future, timeout_sec=15.0):
                    terminal_callback(status="failed", message="Cortex goal send timed out")
                    return
                handle = send_future.result()
                if handle is None or not handle.accepted:
                    terminal_callback(status="failed", message="Cortex goal rejected")
                    return
                self.goal_handle = handle
                result_future = handle.get_result_async()
                if not self._wait(result_future, timeout_sec=300.0):
                    terminal_callback(status="failed", message="Cortex result timed out")
                    return
                result = result_future.result()
                # rclpy action GoalStatus codes:
                #   4 = STATUS_SUCCEEDED, 5 = STATUS_CANCELED, 6 = STATUS_ABORTED
                # (These were previously mis-mapped, turning real ABORTED
                # results into fake successes.)
                if result.status == 4:
                    terminal_callback(
                        status="succeeded",
                        message="Task complete!",
                        raw_status=result.status,
                    )
                elif result.status == 5:
                    terminal_callback(
                        status="cancelled",
                        message="Command cancelled",
                        raw_status=result.status,
                    )
                else:
                    terminal_callback(
                        status="failed",
                        message=(
                            f"Cortex action ended with status {result.status} "
                            "(ABORTED)"
                        ),
                        raw_status=result.status,
                    )

            def shutdown(self) -> None:
                try:
                    self._action_client.destroy()
                    self._executor.shutdown(timeout_sec=2.0)
                    self._node.destroy_node()
                except Exception:
                    pass

        return _CortexClient()


def create_readonly_ros_backend(*, execution_mode: str) -> RosReadonlyBackend | None:
    """Build the read-only ROS backend, or None outside hardware mode.

    ``rclpy`` is imported only here, and only when ``execution_mode`` is
    ``hardware``. Fixture/mock modes never touch the ROS stack.
    """
    graph = create_ros_context(execution_mode=execution_mode)
    if graph is None:
        return None
    return RosReadonlyBackend(graph)


def create_cortex_command_backend(
    *, execution_mode: str, client_factory: Optional[Callable[[], Any]] = None
) -> RosCortexCommandBackend | None:
    """Build the command backend (M8) or None outside hardware mode."""
    graph = create_ros_context(execution_mode=execution_mode)
    if graph is None:
        return None
    return RosCortexCommandBackend(graph, client_factory=client_factory)
