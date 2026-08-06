"""Contract tests for the Go2 read-only health probe and Go2Health mapper.

Workstation tests: fake probes only; never import unitree_sdk2py, piper_sdk,
or rclpy. The Go2SystemProbe protocol MUST expose no movement method, and any
disconnected/stale/non-stationary/out-of-limit evidence must be fail-closed
(nothing is reported healthy by default).
"""

import sys
import unittest
from datetime import datetime, timezone

from ubrobot_contracts.capabilities import (
    CapabilityAvailability,
    CapabilityHealth,
    CapabilityName,
)
from ubrobot_contracts.telemetry import TelemetryChannel, TelemetryState

from robot_edge.hardware.go2_health import (
    Go2Health,
    Go2PiperHealth,
    Go2SystemProbe,
)


class FakeGo2Probe:
    """Deterministic fake of the Go2 read-only probe."""

    _UNSET = object()

    def __init__(
        self,
        *,
        connected=True,
        standing=True,
        odometry=_UNSET,
        body_velocity=(0.0, 0.0, 0.0),
        imu=_UNSET,
        body_orientation=(0.0, 0.0, 0.0),
        local_stop_ready=True,
    ) -> None:
        self._connected = connected
        self._standing = standing
        # Healthy by default: fresh odometry + IMU unless overridden.
        self._odometry = _fresh_odometry() if odometry is self._UNSET else odometry
        self._body_velocity = body_velocity
        self._imu = _fresh_imu() if imu is self._UNSET else imu
        self._body_orientation = body_orientation
        self._local_stop_ready = local_stop_ready

    def connected(self) -> bool:
        return self._connected

    def standing(self) -> bool:
        return self._standing

    def odometry(self):
        return self._odometry

    def body_velocity(self):
        return self._body_velocity

    def imu(self):
        return self._imu

    def body_orientation(self):
        return self._body_orientation

    def local_stop_ready(self) -> bool:
        return self._local_stop_ready


def _fresh_odometry(**overrides):
    value = {"x": 0.1, "y": -0.2, "yaw": 0.0, "age_sec": 0.01}
    value.update(overrides)
    return value


def _fresh_imu(**overrides):
    value = {"roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0, "age_sec": 0.01}
    value.update(overrides)
    return value


class TestGo2ProbeProtocol(unittest.TestCase):
    def test_probe_exposes_no_movement_method(self) -> None:
        for name in (
            "move",
            "stop",
            "stand_up",
            "stand_down",
            "set_velocity",
            "set_speed_level",
        ):
            self.assertFalse(
                hasattr(Go2SystemProbe, name),
                f"Go2SystemProbe must not expose movement method {name}",
            )

    def test_required_readonly_methods_present(self) -> None:
        for name in (
            "connected",
            "standing",
            "odometry",
            "body_velocity",
            "imu",
            "body_orientation",
            "local_stop_ready",
        ):
            self.assertTrue(hasattr(Go2SystemProbe, name), name)


class TestGo2Health(unittest.TestCase):
    def _health(self, **probe_kwargs) -> Go2Health:
        return Go2Health(FakeGo2Probe(**probe_kwargs))

    def test_disconnected_go2_is_disconnected_not_healthy(self) -> None:
        health = self._health(
            connected=False,
            standing=False,
            odometry=None,
            body_velocity=None,
            imu=None,
            body_orientation=None,
        )
        cap = health.capability()
        self.assertEqual(cap.availability, CapabilityAvailability.DISCONNECTED)
        self.assertEqual(cap.health, CapabilityHealth.UNKNOWN)
        self.assertFalse(cap.hardware_authority)

    def test_non_standing_go2_is_unavailable(self) -> None:
        health = self._health(standing=False)
        cap = health.capability()
        self.assertEqual(cap.availability, CapabilityAvailability.UNAVAILABLE)
        self.assertEqual(cap.health, CapabilityHealth.UNHEALTHY)
        self.assertIn("standing", cap.detail)

    def test_stale_odometry_is_unhealthy(self) -> None:
        health = self._health(odometry=_fresh_odometry(age_sec=10.0))
        cap = health.capability()
        self.assertEqual(cap.health, CapabilityHealth.UNHEALTHY)
        self.assertIn("odometry", cap.detail)

    def test_missing_odometry_is_unhealthy(self) -> None:
        health = self._health(odometry=None, body_velocity=None)
        cap = health.capability()
        self.assertEqual(cap.health, CapabilityHealth.UNHEALTHY)
        self.assertIn("odometry", cap.detail)

    def test_nonzero_body_velocity_is_unhealthy(self) -> None:
        health = self._health(body_velocity=(0.05, 0.0, 0.0))
        cap = health.capability()
        self.assertEqual(cap.health, CapabilityHealth.UNHEALTHY)
        self.assertIn("moving", cap.detail)

    def test_stale_imu_is_unhealthy(self) -> None:
        health = self._health(imu=_fresh_imu(age_sec=10.0))
        cap = health.capability()
        self.assertEqual(cap.health, CapabilityHealth.UNHEALTHY)
        self.assertIn("imu", cap.detail.lower())

    def test_orientation_over_limit_is_unhealthy(self) -> None:
        health = self._health(body_orientation=(0.9, 0.0, 0.0))  # ~51 deg roll
        cap = health.capability()
        self.assertEqual(cap.health, CapabilityHealth.UNHEALTHY)
        self.assertIn("orientation", cap.detail)

    def test_local_stop_not_ready_is_unhealthy(self) -> None:
        health = self._health(local_stop_ready=False)
        cap = health.capability()
        self.assertEqual(cap.health, CapabilityHealth.UNHEALTHY)
        self.assertIn("local stop", cap.detail)

    def test_healthy_go2_is_available_without_authority(self) -> None:
        health = self._health()
        cap = health.capability()
        self.assertEqual(cap.availability, CapabilityAvailability.AVAILABLE)
        self.assertEqual(cap.health, CapabilityHealth.HEALTHY)
        self.assertEqual(cap.name, CapabilityName.NAVIGATION)
        self.assertFalse(cap.hardware_authority)  # M6 read-only

    def test_telemetry_maps_truthfully(self) -> None:
        health = self._health()
        snap = health.telemetry()
        odom = snap[TelemetryChannel.ODOMETRY]
        self.assertEqual(odom.latest.state, TelemetryState.AVAILABLE)
        self.assertIn("x", odom.latest.value)


class TestGo2PiperHealth(unittest.TestCase):
    def _piper(self, **kw):
        from tests.robot_edge.test_hardware_health_mapping import _FakePiperProbe  # noqa: PLC0415

        defaults = {"can": True, "driver": True, "torque_off": True, "arm": True}
        defaults.update(kw)
        return _FakePiperProbe(**defaults)

    def _platform(self, go2=None, piper=None, tf_complete=True, local_stop_bound=True):
        from robot_edge.hardware.piper_health import PiperHealth  # noqa: PLC0415

        g2 = go2 or Go2Health(FakeGo2Probe())
        pp = piper or PiperHealth(self._piper())
        return Go2PiperHealth(
            go2_health=g2,
            piper_health=pp,
            tf_complete=tf_complete,
            local_stop_bound=local_stop_bound,
        )

    def test_go2_failure_blocks_platform_authority(self) -> None:
        platform = self._platform(go2=Go2Health(FakeGo2Probe(standing=False)))
        self.assertFalse(platform.authority().granted)
        self.assertIn("go2", platform.authority().detail)

    def test_piper_failure_blocks_platform_authority(self) -> None:
        from robot_edge.hardware.piper_health import PiperHealth  # noqa: PLC0415

        platform = self._platform(piper=PiperHealth(self._piper(torque_off=False)))
        self.assertFalse(platform.authority().granted)

    def test_incomplete_tf_blocks_platform_authority(self) -> None:
        platform = self._platform(tf_complete=False)
        self.assertFalse(platform.authority().granted)

    def test_unbound_estop_blocks_platform_authority(self) -> None:
        platform = self._platform(local_stop_bound=False)
        self.assertFalse(platform.authority().granted)

    def test_all_healthy_gives_authority_without_hardware_execution(self) -> None:
        platform = self._platform()
        auth = platform.authority()
        self.assertTrue(auth.granted)
        for name in (CapabilityName.NAVIGATION, CapabilityName.GRASP):
            cap = platform.capability(name)
            self.assertFalse(cap.hardware_authority)


class TestGo2Telemetry(unittest.TestCase):
    """Go2 bridge topic mapping onto shared telemetry (read-only)."""

    def _graph(self, topics=None, reads=None):
        from tests.robot_edge.test_hardware_health_mapping import FakeRosGraph  # noqa: PLC0415

        return FakeRosGraph(topics=topics, reads=reads)

    def _reader(self, graph):
        from robot_edge.hardware.go2_telemetry import Go2Telemetry  # noqa: PLC0415

        return Go2Telemetry(graph, max_age_sec=2.0)

    def test_fresh_go2_odometry_is_available(self) -> None:
        from datetime import datetime, timezone  # noqa: PLC0415

        now = datetime.now(timezone.utc)
        stamp = {"sec": int(now.timestamp()), "nanosec": 0}
        reader = self._reader(
            self._graph(
                topics={"/odom"},
                reads={
                    "/odom": {
                        "header": {"stamp": stamp},
                        "pose": {"pose": {"position": {"x": 0.5, "y": 0.0}}},
                        "twist": {"twist": {"linear": {"x": 0.0}}},
                    }
                },
            )
        )
        snap = reader.snapshot()
        odom = snap[TelemetryChannel.ODOMETRY]
        self.assertEqual(odom.latest.state, TelemetryState.AVAILABLE)
        self.assertEqual(odom.latest.value["x"], 0.5)

    def test_missing_topic_is_disconnected(self) -> None:
        reader = self._reader(self._graph(topics=set()))
        snap = reader.snapshot()
        self.assertEqual(
            snap[TelemetryChannel.ODOMETRY].latest.state,
            TelemetryState.DISCONNECTED,
        )

    def test_stale_topic_is_stale(self) -> None:
        from datetime import timedelta  # noqa: PLC0415

        old = datetime.now(timezone.utc) - timedelta(seconds=10)
        stamp = {"sec": int(old.timestamp()), "nanosec": 0}
        reader = self._reader(
            self._graph(
                topics={"/odom"},
                reads={"/odom": {"header": {"stamp": stamp}, "pose": {}}},
            )
        )
        snap = reader.snapshot()
        self.assertEqual(snap[TelemetryChannel.ODOMETRY].latest.state, TelemetryState.STALE)


class TestRosGo2Probe(unittest.TestCase):
    """The real dock probe maps ROS bridge topics onto the probe protocol."""

    def _probe(self, graph, **kw):
        from robot_edge.hardware.go2_health import RosGo2Probe  # noqa: PLC0415

        return RosGo2Probe(graph, **kw)

    def _graph(self, topics=None, reads=None):
        from tests.robot_edge.test_hardware_health_mapping import FakeRosGraph  # noqa: PLC0415

        return FakeRosGraph(topics=topics, reads=reads)

    def test_connected_requires_odom_and_joint_states(self) -> None:
        probe = self._probe(self._graph(topics={"/odom"}))
        self.assertFalse(probe.connected())
        probe = self._probe(self._graph(topics={"/odom", "/joint_states"}))
        self.assertTrue(probe.connected())

    def test_standing_requires_fresh_joint_states_and_odom(self) -> None:
        now = datetime.now(timezone.utc)
        stamp = {"sec": int(now.timestamp()), "nanosec": 0}
        graph = self._graph(
            topics={"/odom", "/joint_states"},
            reads={
                "/joint_states": {"header": {"stamp": stamp}, "name": ["fl_hip", "fr_hip"]},
                "/odom": {"header": {"stamp": stamp}},
            },
        )
        probe = self._probe(graph)
        self.assertTrue(probe.standing())

    def test_local_stop_unbound_is_fail_closed(self) -> None:
        probe = self._probe(self._graph())
        self.assertFalse(probe.local_stop_ready())
        probe = self._probe(self._graph(), local_stop_ready=lambda: True)
        self.assertTrue(probe.local_stop_ready())

    def test_body_velocity_reads_odom_twist(self) -> None:
        now = datetime.now(timezone.utc)
        stamp = {"sec": int(now.timestamp()), "nanosec": 0}
        graph = self._graph(
            topics={"/odom"},
            reads={
                "/odom": {
                    "header": {"stamp": stamp},
                    "twist": {"twist": {"linear": {"x": 0.1}, "angular": {"z": 0.05}}},
                }
            },
        )
        probe = self._probe(graph)
        self.assertAlmostEqual(probe.body_velocity()[0], 0.1)
        self.assertAlmostEqual(probe.body_velocity()[2], 0.05)


class TestGo2ImportBoundary(unittest.TestCase):
    def test_no_hardware_sdk_imports(self) -> None:
        import robot_edge.hardware.go2_health  # noqa: F401
        import robot_edge.hardware.go2_telemetry  # noqa: F401
        import robot_edge.platforms  # noqa: F401

        for forbidden in ("rclpy", "piper_sdk", "unitree_sdk2py", "unitree_sdk2"):
            self.assertNotIn(forbidden, sys.modules, forbidden)


if __name__ == "__main__":
    unittest.main()
