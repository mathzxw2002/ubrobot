"""Read-only ROS Action server inventory (M6)."""

from __future__ import annotations

from datetime import datetime, timezone

from ubrobot_contracts.capabilities import (
    CapabilityAvailability,
    CapabilityHealth,
    CapabilityName,
    CapabilitySnapshot,
    ExecutionMode,
)

from robot_edge.ros.context import RosGraph

# Expected ROS 2 Action servers, per the robot contracts:
#   ros_depends_ws/src/ubrobot_interfaces/action/NavigateToObject.action
#   ros_depends_ws/src/ubrobot_interfaces/action/GraspObject.action
EXPECTED_ACTIONS: dict[CapabilityName, str] = {
    CapabilityName.NAVIGATION: "/ubrobot/navigation/navigate_to_object",
    CapabilityName.GRASP: "/ubrobot/manipulation/grasp_object",
}

# Capabilities without an Action server backing (observation, follow, stop)
# are derived from topic presence / local state. Topic names are the
# measured live paths (2026-08-03): RealSense under /camera/camera/...,
# lekiwi odometry under /lekiwi_base_controller/odom. Legacy paths are kept
# as fallbacks for graph compatibility.
_TOPIC_CAPABILITIES: dict[CapabilityName, tuple[str, ...]] = {
    CapabilityName.OBSERVATION: (
        "/camera/camera/color/camera_info",
        "/camera/camera/depth/camera_info",
        "/camera/camera_info",
        "/camera/depth/camera_info",
    ),
    CapabilityName.FOLLOW: ("/lekiwi_base_controller/odom", "/odom/wheel", "/odom"),
    CapabilityName.STOP: (),
}


class RosActionInventory:
    """Read-only inventory of ROS Action servers for capability health.

    Read-only mode never constructs Action clients for command execution and
    never publishes control topics; it only reports what exists in the graph.
    """

    def __init__(self, graph: RosGraph) -> None:
        self._graph = graph

    def capabilities(self) -> dict[CapabilityName, CapabilitySnapshot]:
        now = datetime.now(timezone.utc)
        result: dict[CapabilityName, CapabilitySnapshot] = {}

        for capability, action_name in EXPECTED_ACTIONS.items():
            present = self._graph.has_action_server(action_name)
            result[capability] = CapabilitySnapshot(
                name=capability,
                availability=(
                    CapabilityAvailability.AVAILABLE
                    if present
                    else CapabilityAvailability.UNAVAILABLE
                ),
                health=CapabilityHealth.HEALTHY if present else CapabilityHealth.UNKNOWN,
                execution_mode=ExecutionMode.HARDWARE,
                required_resources=["camera", "depth", "odometry", "navigation_lease"],
                hardware_authority=False,  # M6: read-only, no command authority
                detail=(
                    f"ROS action server present: {action_name}"
                    if present
                    else f"ROS action server missing: {action_name}"
                ),
                last_updated=now,
            )

        for capability, topics in _TOPIC_CAPABILITIES.items():
            present = any(self._graph.has_topic(t) for t in topics) if topics else True
            result[capability] = CapabilitySnapshot(
                name=capability,
                availability=(
                    CapabilityAvailability.AVAILABLE
                    if present
                    else CapabilityAvailability.UNAVAILABLE
                ),
                health=CapabilityHealth.HEALTHY if present else CapabilityHealth.UNKNOWN,
                execution_mode=ExecutionMode.HARDWARE,
                required_resources=(
                    list(topics) if capability != CapabilityName.STOP else ["safety_control"]
                ),
                hardware_authority=False,
                detail=(
                    "derived from ROS topics"
                    if present
                    else "no backing ROS topic present"
                ),
                last_updated=now,
            )
        return result
