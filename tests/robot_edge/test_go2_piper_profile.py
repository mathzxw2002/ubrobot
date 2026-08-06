"""Contract tests for the go2_piper platform assembly and profile gating.

``robot_edge.platforms`` is the single source of platform enumeration for
Robot Edge: it defines what base/arm/perception a platform uses. The
``go2_piper`` profile must be conservative (stationary base, low approach
speed) and must never grant hardware authority from read-only state alone.
"""

import sys
import unittest

from robot_edge.platforms import PlatformDefinition, get_platform, supported_platforms
from robot_edge.hardware.mobile_base_health import MobileBaseHealth, SUPPORTED_PROFILES


class FakeRosGraph:
    def __init__(self, topics=None, reads=None) -> None:
        self._topics = set(topics or [])
        self._reads = dict(reads or {})

    def has_topic(self, topic: str) -> bool:
        return topic in self._topics

    def read_topic(self, topic: str):
        return self._reads.get(topic)


class TestGo2PiperPlatformDefinition(unittest.TestCase):
    def test_go2_piper_is_a_defined_platform(self) -> None:
        platform = get_platform("go2_piper")
        self.assertIsInstance(platform, PlatformDefinition)
        self.assertEqual(platform.key, "go2_piper")

    def test_go2_piper_assembly(self) -> None:
        platform = get_platform("go2_piper")
        self.assertEqual(platform.base, "go2")
        self.assertEqual(platform.arm, "piper")
        self.assertEqual(platform.perception, "remote-service")

    def test_go2_piper_requires_stationary_base(self) -> None:
        platform = get_platform("go2_piper")
        self.assertTrue(platform.requires_stationary_base)

    def test_go2_piper_conservative_velocity_limits(self) -> None:
        platform = get_platform("go2_piper")
        self.assertLessEqual(platform.max_base_linear_mps, 0.2)
        self.assertLessEqual(platform.max_base_angular_radps, 0.5)

    def test_supported_platforms_includes_go2_piper(self) -> None:
        self.assertIn("go2_piper", supported_platforms())

    def test_unknown_platform_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            get_platform("not_a_real_robot")


class TestMobileBaseHealthGo2Profile(unittest.TestCase):
    def test_go2_is_supported_profile(self) -> None:
        self.assertIn("go2", SUPPORTED_PROFILES)

    def test_go2_profile_constructs_readonly(self) -> None:
        reader = MobileBaseHealth(FakeRosGraph(), profile="go2")
        self.assertEqual(reader.profile, "go2")

    def test_go2_uses_go2_odometry_topics(self) -> None:
        from ubrobot_contracts.telemetry import TelemetryChannel

        reader = MobileBaseHealth(FakeRosGraph(), profile="go2")
        # go2 odom topic is the unitree bridge output, not lekiwi's wheel odom.
        # Snapshot with no topics must report DISCONNECTED (fail-closed).
        snap = reader.snapshot()
        self.assertEqual(
            snap[TelemetryChannel.ODOMETRY].latest.state.value,
            "disconnected",
        )

    def test_unsupported_profile_still_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MobileBaseHealth(FakeRosGraph(), profile="some_other_robot")


class TestGo2PiperImportBoundary(unittest.TestCase):
    def test_no_hardware_sdk_imports(self) -> None:
        import robot_edge.platforms  # noqa: F401

        for forbidden in ("rclpy", "piper_sdk", "unitree_sdk2py", "unitree_sdk2"):
            self.assertNotIn(forbidden, sys.modules, forbidden)


if __name__ == "__main__":
    unittest.main()
