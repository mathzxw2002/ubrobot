"""Contract tests for hardware health readers (M6, read-only).

Uses fake ROS graphs and fake system probes. These tests never import
pyrealsense2, piper_sdk, unitree_sdk2py, or rclpy.
"""

import sys
import unittest
from datetime import datetime, timedelta, timezone

from ubrobot_contracts.capabilities import (
    CapabilityAvailability,
    CapabilityHealth,
    CapabilityName,
)
from ubrobot_contracts.telemetry import TelemetryChannel, TelemetryState


class FakeRosGraph:
    def __init__(self, topics=None, reads=None) -> None:
        self._topics = set(topics or [])
        self._reads = dict(reads or {})

    def has_topic(self, topic: str) -> bool:
        return topic in self._topics

    def read_topic(self, topic: str):
        return self._reads.get(topic)


def _stamped(**fields):
    """Build a dict message with a recent header stamp."""
    base = {
        "header": {
            "frame_id": "camera_color_optical_frame",
            "stamp": {
                "sec": int(datetime.now(timezone.utc).timestamp()),
                "nanosec": 0,
            },
        }
    }
    base.update(fields)
    return base


def _old_stamp() -> dict:
    old = datetime.now(timezone.utc) - timedelta(seconds=10)
    return {
        "sec": int(old.timestamp()),
        "nanosec": 0,
    }


from robot_edge.hardware.mobile_base_health import MobileBaseHealth  # noqa: E402
from robot_edge.hardware.piper_health import PiperHealth  # noqa: E402
from robot_edge.hardware.realsense_ros import RealsenseHealthReader  # noqa: E402


class TestHardwareImportBoundary(unittest.TestCase):
    def test_no_hardware_sdk_imports(self) -> None:
        import robot_edge.hardware  # noqa: F401
        import robot_edge.hardware.realsense_ros  # noqa: F401
        import robot_edge.hardware.mobile_base_health  # noqa: F401
        import robot_edge.hardware.piper_health  # noqa: F401

        for forbidden in (
            "rclpy",
            "pyrealsense2",
            "piper_sdk",
            "unitree_sdk2py",
        ):
            self.assertNotIn(forbidden, sys.modules, forbidden)


class TestRealsenseHealth(unittest.TestCase):
    def _reader(self, topics=None, reads=None) -> RealsenseHealthReader:
        return RealsenseHealthReader(
            FakeRosGraph(topics=topics, reads=reads), max_age_sec=2.0
        )

    def test_valid_metadata_is_available_and_checked(self) -> None:
        reader = self._reader(
            topics={"/camera/camera/color/camera_info", "/camera/camera/depth/camera_info"},
            reads={
                "/camera/camera/color/camera_info": _stamped(
                    width=1280,
                    height=720,
                    encoding="rgb8",
                    distortion_model="plumb_bob",
                    k=[380.0, 0.0, 640.0, 0.0, 380.0, 360.0, 0.0, 0.0, 1.0],
                ),
                "/camera/camera/depth/camera_info": _stamped(
                    width=640,
                    height=480,
                    encoding="16UC1",
                    k=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                ),
            },
        )
        snap = reader.snapshot()
        cam = snap[TelemetryChannel.CAMERA]
        self.assertEqual(cam.latest.state, TelemetryState.AVAILABLE)
        self.assertEqual(cam.latest.value["width"], 1280)
        self.assertEqual(cam.latest.value["encoding"], "rgb8")
        self.assertTrue(cam.latest.value["calibrated"])
        self.assertTrue(cam.latest.value["frame_matches_expected"])
        depth = snap[TelemetryChannel.DEPTH]
        self.assertEqual(depth.latest.state, TelemetryState.AVAILABLE)
        self.assertEqual(depth.latest.value["encoding"], "16UC1")

    def test_missing_topics_are_disconnected(self) -> None:
        reader = self._reader(topics=set())
        snap = reader.snapshot()
        self.assertEqual(snap[TelemetryChannel.CAMERA].latest.state, TelemetryState.DISCONNECTED)
        self.assertEqual(snap[TelemetryChannel.DEPTH].latest.state, TelemetryState.DISCONNECTED)

    def test_stale_stamp_is_stale_not_available(self) -> None:
        reader = self._reader(
            topics={"/camera/camera/color/camera_info"},
            reads={
                "/camera/camera/color/camera_info": {
                    "header": {"frame_id": "camera_color_optical_frame", "stamp": _old_stamp()},
                    "width": 640,
                    "height": 480,
                    "encoding": "rgb8",
                    "k": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                }
            },
        )
        snap = reader.snapshot()
        self.assertEqual(snap[TelemetryChannel.CAMERA].latest.state, TelemetryState.STALE)

    def test_uncalibrated_camera_is_not_claimed_calibrated(self) -> None:
        reader = self._reader(
            topics={"/camera/camera/color/camera_info"},
            reads={
                "/camera/camera/color/camera_info": _stamped(
                    width=640, height=480, encoding="rgb8", k=None
                )
            },
        )
        snap = reader.snapshot()
        self.assertFalse(snap[TelemetryChannel.CAMERA].latest.value["calibrated"])


class TestMobileBaseHealth(unittest.TestCase):
    def test_lekiwi_odometry_and_joint_mapping(self) -> None:
        reader = MobileBaseHealth(
            FakeRosGraph(
                topics={"/lekiwi_base_controller/odom", "/joint_states"},
                reads={
                    "/lekiwi_base_controller/odom": {
                        "header": {"stamp": {"sec": int(datetime.now(timezone.utc).timestamp()), "nanosec": 0}},
                        "pose": {
                            "position": {"x": 0.1, "y": -0.2},
                            "orientation": {"x": 0.0, "y": 0.0, "z": 0.7071, "w": 0.7071},
                        },
                        "twist": {"linear": {"x": 0.0}},
                    },
                    "/joint_states": {
                        "header": {"stamp": {"sec": int(datetime.now(timezone.utc).timestamp()), "nanosec": 0}},
                        "name": ["back", "right", "left"],
                        "position": [1.0, 2.0, 3.0],
                        "velocity": [0.0, 0.0, 0.0],
                    },
                },
            ),
            profile="lekiwi",
        )
        snap = reader.snapshot()
        odom = snap[TelemetryChannel.ODOMETRY]
        self.assertEqual(odom.latest.state, TelemetryState.AVAILABLE)
        self.assertEqual(odom.latest.value["x"], 0.1)
        # z=0.7071, w=0.7071 is a 90-degree rotation -> yaw ~= pi/2.
        self.assertAlmostEqual(odom.latest.value["yaw"], 1.5708, places=3)
        joints = snap[TelemetryChannel.JOINT_STATES]
        self.assertEqual(joints.latest.value["motor_count"], 3)
        self.assertEqual(joints.latest.value["names"], ["back", "right", "left"])

    def test_missing_topics_are_disconnected(self) -> None:
        reader = MobileBaseHealth(FakeRosGraph(), profile="lekiwi")
        snap = reader.snapshot()
        self.assertEqual(snap[TelemetryChannel.ODOMETRY].latest.state, TelemetryState.DISCONNECTED)
        self.assertEqual(snap[TelemetryChannel.JOINT_STATES].latest.state, TelemetryState.DISCONNECTED)

    def test_stale_odometry_is_stale(self) -> None:
        reader = MobileBaseHealth(
            FakeRosGraph(
                topics={"/lekiwi_base_controller/odom"},
                reads={
                    "/lekiwi_base_controller/odom": {
                        "header": {"stamp": _old_stamp()},
                        "pose": {"position": {"x": 1.0, "y": 0.0}},
                    }
                },
            ),
            profile="lekiwi",
            max_age_sec=2.0,
        )
        snap = reader.snapshot()
        self.assertEqual(snap[TelemetryChannel.ODOMETRY].latest.state, TelemetryState.STALE)

    def test_unsupported_profile_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MobileBaseHealth(FakeRosGraph(), profile="go2")


class _FakePiperProbe:
    def __init__(self, can=False, driver=False, torque_off=False, arm=False) -> None:
        self._can = can
        self._driver = driver
        self._torque_off = torque_off
        self._arm = arm

    def can_interface_present(self) -> bool:
        return self._can

    def driver_process_running(self) -> bool:
        return self._driver

    def torque_confirmed_disabled(self) -> bool:
        return self._torque_off

    def arm_present(self) -> bool:
        return self._arm


class TestPiperHealth(unittest.TestCase):
    def test_absent_piper_is_disconnected(self) -> None:
        health = PiperHealth(_FakePiperProbe())
        cap = health.capability()
        self.assertEqual(cap.availability, CapabilityAvailability.DISCONNECTED)
        self.assertEqual(cap.health, CapabilityHealth.UNKNOWN)
        self.assertFalse(cap.hardware_authority)

    def test_torque_enabled_is_unhealthy_stop_condition(self) -> None:
        health = PiperHealth(
            _FakePiperProbe(can=True, driver=True, torque_off=False, arm=True)
        )
        cap = health.capability()
        self.assertEqual(cap.availability, CapabilityAvailability.UNAVAILABLE)
        self.assertEqual(cap.health, CapabilityHealth.UNHEALTHY)
        self.assertIn("torque", cap.detail)

    def test_ready_piper_is_available_authority_false(self) -> None:
        health = PiperHealth(
            _FakePiperProbe(can=True, driver=True, torque_off=True, arm=True)
        )
        cap = health.capability()
        self.assertEqual(cap.availability, CapabilityAvailability.AVAILABLE)
        self.assertEqual(cap.health, CapabilityHealth.HEALTHY)
        self.assertFalse(cap.hardware_authority)

    def test_telemetry_maps_truthfully(self) -> None:
        health = PiperHealth(_FakePiperProbe(can=True, driver=True, torque_off=True, arm=True))
        snap = health.telemetry()
        self.assertEqual(
            snap[TelemetryChannel.CAPABILITY_HEALTH].latest.state,
            TelemetryState.AVAILABLE,
        )


if __name__ == "__main__":
    unittest.main()
