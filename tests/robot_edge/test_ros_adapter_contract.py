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
from robot_edge.ros.context import (
    _json_safe,
    _message_field_names,
    create_ros_context,
)  # noqa: E402


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
            topics={"/lekiwi_base_controller/odom", "/joint_states", "/camera/camera/color/camera_info"},
            reads={
                "/lekiwi_base_controller/odom": {
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
        backend = self._backend(topics={"/lekiwi_base_controller/odom", "/joint_states"})
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
        backend = self._backend(topics={"/lekiwi_base_controller/odom"}, reads={"/lekiwi_base_controller/odom": None})
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


def _make_msg_class(fields: tuple[str, ...]):
    """Build a class mimicking an rclpy-generated message.

    Real generated messages declare slots with private names (``_frame_id``)
    and expose public properties, with no instance ``__dict__``.
    """

    slots = tuple(f"_{name}" for name in fields)
    attrs: dict[str, Any] = {"__slots__": slots}

    def make_property(name: str):
        def getter(self, _name=name):
            return getattr(self, f"_{_name}")
        return property(getter)

    for name in fields:
        attrs[name] = make_property(name)

    def _init(self, **kw) -> None:
        for name in fields:
            object.__setattr__(self, f"_{name}", kw.get(name))

    attrs["__init__"] = _init
    return type("_Ros2StyleMessage", (), attrs)


class TestJsonSafeRosMessage(unittest.TestCase):
    """_json_safe must extract fields from rclpy-style generated messages."""

    def test_private_slots_map_to_public_names(self) -> None:
        Odometry = _make_msg_class(("pose", "twist", "header"))
        msg = Odometry(
            pose=_make_msg_class(("position",))(
                position=_make_msg_class(("x", "y", "z"))(x=1.25, y=-0.5, z=0.0)
            ),
            twist=_make_msg_class(("linear",))(
                linear=_make_msg_class(("x",))(x=0.05)
            ),
            header=_make_msg_class(("stamp", "frame_id"))(
                stamp=_make_msg_class(("sec", "nanosec"))(sec=10, nanosec=5),
                frame_id="base",
            ),
        )
        self.assertEqual(_message_field_names(msg), ["pose", "twist", "header"])
        safe = _json_safe(msg)
        self.assertEqual(safe["pose"]["position"]["x"], 1.25)
        self.assertEqual(safe["twist"]["linear"]["x"], 0.05)
        self.assertEqual(safe["header"]["stamp"]["sec"], 10)
        self.assertEqual(safe["header"]["frame_id"], "base")

    def test_scalar_arrays_are_kept(self) -> None:
        JointStates = _make_msg_class(("name", "position", "velocity"))
        msg = JointStates(
            name=["back", "left", "right"],
            position=[0.0, 0.1, -0.1],
            velocity=[0.0, 0.0, 0.0],
        )
        safe = _json_safe(msg)
        self.assertEqual(safe["name"], ["back", "left", "right"])
        self.assertEqual(safe["position"], [0.0, 0.1, -0.1])
        self.assertEqual(safe["velocity"], [0.0, 0.0, 0.0])

    def test_array_of_nested_messages(self) -> None:
        WithCov = _make_msg_class(("covariance",))
        CovEntry = _make_msg_class(("a", "b"))
        msg = WithCov(covariance=[CovEntry(a=1.0), CovEntry(b=2.0)])
        safe = _json_safe(msg)
        self.assertEqual(safe["covariance"][0]["a"], 1.0)
        self.assertEqual(safe["covariance"][1]["b"], 2.0)

    def test_bytes_are_reduced_to_size(self) -> None:
        DataMsg = _make_msg_class(("data", "label"))
        msg = DataMsg(data=b"\x00\x01\x02\x03", label="ok")
        safe = _json_safe(msg)
        self.assertEqual(safe["data"], 4)
        self.assertEqual(safe["label"], "ok")

    def test_depth_bound(self) -> None:
        # Six nested levels; _json_safe recurses up to depth 4, so the
        # innermost nodes collapse to {} and only 5 child keys survive.
        Node = _make_msg_class(("value", "child"))
        deep = Node(value=0)
        for _ in range(6):
            deep = Node(child=deep)
        safe = _json_safe(deep)
        self.assertEqual(
            safe,
            {"child": {"child": {"child": {"child": {"child": {}}}}}},
        )
