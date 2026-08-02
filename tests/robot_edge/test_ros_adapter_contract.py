"""Contract tests for Robot Edge ROS-side read-only adapters (M6).

Uses fake ROS graph clients only: these tests never import rclpy and never
touch a real ROS installation (plan constraint).
"""

import json
import sys
import unittest
from typing import Any
from unittest.mock import patch

from ubrobot_contracts.capabilities import (
    CapabilityAvailability,
    CapabilityName,
    ExecutionMode,
)
from ubrobot_contracts.telemetry import TelemetryChannel, TelemetryState


class FakeRosGraph:
    """Deterministic fake of the read-only ROS graph contract."""

    def __init__(self, topics=None, action_servers=None, reads=None) -> None:
        self._topics = set(topics or [])
        self._actions = set(action_servers or [])
        self._reads = dict(reads or {})
        self.shutdown_called = False

    def has_topic(self, topic: str) -> bool:
        return topic in self._topics

    def read_topic(self, topic: str):
        return self._reads.get(topic)

    def has_action_server(self, action_name: str) -> bool:
        return action_name in self._actions

    def shutdown(self) -> None:
        self.shutdown_called = True


from robot_edge.ros.backend import RosReadonlyBackend  # noqa: E402
from robot_edge.ros.context import create_ros_context  # noqa: E402


class TestRosImportBoundary(unittest.TestCase):
    """rclpy must never be imported by the package or outside hardware mode."""

    def tearDown(self) -> None:
        sys.modules.pop("rclpy", None)
        sys.modules.pop("rclpy.node", None)

    def test_package_import_does_not_import_rclpy(self) -> None:
        import robot_edge.ros  # noqa: F401

        self.assertNotIn("rclpy", sys.modules)

    def test_factory_returns_none_outside_hardware_mode(self) -> None:
        for mode in ("fixture", "mock", "remote", ""):
            self.assertIsNone(
                create_ros_context(execution_mode=mode),
                f"mode {mode!r} must not construct a ROS context",
            )
        self.assertNotIn("rclpy", sys.modules)

    def test_factory_imports_rclpy_only_for_hardware_mode(self) -> None:
        created: list[str] = []

        class _FakeNode:
            def __init__(self, name: str) -> None:
                created.append(name)

        class _FakeRclpy:
            @staticmethod
            def ok() -> bool:
                return True

            @staticmethod
            def init(args: Any = None) -> None:
                pass

            class Node(_FakeNode):
                pass

        class _FakeRclpyNodeModule:
            Node = _FakeNode

        with patch.dict(
            sys.modules,
            {"rclpy": _FakeRclpy(), "rclpy.node": _FakeRclpyNodeModule()},
        ):
            graph = create_ros_context(execution_mode="hardware")
        self.assertIsNotNone(graph)
        self.assertEqual(created, ["robot_edge_readonly"])


class TestRosTelemetryMapping(unittest.TestCase):
    """ROS topics map onto shared telemetry DTOs; missing means unavailable."""

    def _backend(self, topics=None, reads=None) -> RosReadonlyBackend:
        graph = FakeRosGraph(topics=topics, reads=reads)
        return RosReadonlyBackend(graph)

    def test_odometry_mapping(self) -> None:
        backend = self._backend(
            topics={"/odom/wheel", "/joint_states", "/camera/camera_info"},
            reads={
                "/odom/wheel": {
                    "pose": {"position": {"x": 1.25, "y": -0.5}},
                    "twist": {"linear": {"x": 0.0}},
                }
            },
        )
        snapshot = backend.get_telemetry_snapshot()
        odom = snapshot[TelemetryChannel.ODOMETRY]
        self.assertEqual(odom.latest.state, TelemetryState.AVAILABLE)
        self.assertEqual(odom.latest.value["x"], 1.25)
        self.assertEqual(odom.latest.value["y"], -0.5)
        json.dumps(odom.model_dump(mode="json"))

    def test_all_six_channels_present_with_explicit_state(self) -> None:
        backend = self._backend(topics={"/odom/wheel", "/joint_states"})
        snapshot = backend.get_telemetry_snapshot()
        self.assertEqual(set(snapshot), set(TelemetryChannel))
        for channel in (
            TelemetryChannel.CAMERA,
            TelemetryChannel.DEPTH,
            TelemetryChannel.NAVIGATION_LEASE,
            TelemetryChannel.CAPABILITY_HEALTH,
        ):
            self.assertEqual(
                snapshot[channel].latest.state, TelemetryState.UNAVAILABLE
            )

    def test_topic_present_but_no_message_is_disconnected(self) -> None:
        backend = self._backend(topics={"/odom/wheel"}, reads={"/odom/wheel": None})
        snapshot = backend.get_telemetry_snapshot()
        self.assertEqual(
            snapshot[TelemetryChannel.ODOMETRY].latest.state,
            TelemetryState.DISCONNECTED,
        )

    def test_sdk_like_objects_never_reach_the_dto(self) -> None:
        class _SdkObject:
            pass

        backend = self._backend(
            topics={"/joint_states"},
            reads={
                "/joint_states": {
                    "name": ["a", "b"],
                    "position": [1.0, 2.0],
                    "sdk": _SdkObject(),
                }
            },
        )
        snapshot = backend.get_telemetry_snapshot()
        value = snapshot[TelemetryChannel.JOINT_STATES].latest.value
        self.assertNotIn("sdk", value)
        self.assertEqual(value["names"], ["a", "b"])
        json.dumps(snapshot[TelemetryChannel.JOINT_STATES].model_dump(mode="json"))

    def test_timestamps_are_timezone_aware(self) -> None:
        backend = self._backend(topics=set())
        snapshot = backend.get_telemetry_snapshot()
        for channel, sample in snapshot.items():
            self.assertIsNotNone(sample.latest.timestamp.tzinfo, channel)


class TestRosActionInventory(unittest.TestCase):
    """Action servers are reported read-only; missing is never healthy."""

    def test_present_actions_are_available_but_authority_false(self) -> None:
        graph = FakeRosGraph(
            topics={
                "/ubrobot/navigation/navigate_to_object/_action/status",
                "/ubrobot/manipulation/grasp_object/_action/send_goal",
            },
            action_servers={
                "/ubrobot/navigation/navigate_to_object",
                "/ubrobot/manipulation/grasp_object",
            },
        )
        backend = RosReadonlyBackend(graph)
        caps = backend.get_capabilities()
        self.assertEqual(
            caps[CapabilityName.NAVIGATION].availability,
            CapabilityAvailability.AVAILABLE,
        )
        self.assertEqual(
            caps[CapabilityName.GRASP].availability,
            CapabilityAvailability.AVAILABLE,
        )
        self.assertFalse(caps[CapabilityName.NAVIGATION].hardware_authority)
        self.assertEqual(
            caps[CapabilityName.NAVIGATION].execution_mode, ExecutionMode.HARDWARE
        )

    def test_missing_actions_are_unavailable_not_healthy(self) -> None:
        graph = FakeRosGraph(topics={"/joint_states"})
        backend = RosReadonlyBackend(graph)
        caps = backend.get_capabilities()
        for name in (CapabilityName.NAVIGATION, CapabilityName.GRASP):
            self.assertEqual(
                caps[name].availability, CapabilityAvailability.UNAVAILABLE
            )
            self.assertFalse(caps[name].hardware_authority)


class TestReadonlyBackendRejectsCommands(unittest.TestCase):
    """M6: every command path rejects while read-only."""

    def test_get_command_sequence_raises_hardware_authority_disabled(self) -> None:
        backend = RosReadonlyBackend(FakeRosGraph())
        with self.assertRaises(RuntimeError) as ctx:
            backend.get_command_sequence("导航到前面的椅子")
        self.assertIn("hardware authority disabled", str(ctx.exception))

    def test_backend_is_hardware_mode_with_no_authority(self) -> None:
        backend = RosReadonlyBackend(FakeRosGraph())
        self.assertEqual(backend.execution_mode, "hardware")
        self.assertFalse(backend.hardware_authority)

    def test_close_releases_the_graph(self) -> None:
        graph = FakeRosGraph()
        backend = RosReadonlyBackend(graph)
        backend.close()
        self.assertTrue(graph.shutdown_called)


if __name__ == "__main__":
    unittest.main()
