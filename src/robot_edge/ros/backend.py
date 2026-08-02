"""Read-only ROS backend for Robot Edge (M6).

Serves capability/telemetry snapshots from the real ROS graph while every
command endpoint rejects with ``hardware authority disabled``. No motion,
no torque, no action clients, no control topics.
"""

from __future__ import annotations

from typing import Any, Iterator

from ubrobot_contracts.capabilities import CapabilityName, CapabilitySnapshot
from ubrobot_contracts.edge_api import CommandState
from ubrobot_contracts.telemetry import TelemetryChannel, TelemetrySnapshot

from robot_edge.ros.actions import RosActionInventory
from robot_edge.ros.context import RosGraph, create_ros_context
from robot_edge.ros.telemetry import RosTelemetryReader


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


def create_readonly_ros_backend(*, execution_mode: str) -> RosReadonlyBackend | None:
    """Build the read-only ROS backend, or None outside hardware mode.

    ``rclpy`` is imported only here, and only when ``execution_mode`` is
    ``hardware``. Fixture/mock modes never touch the ROS stack.
    """
    graph = create_ros_context(execution_mode=execution_mode)
    if graph is None:
        return None
    return RosReadonlyBackend(graph)
