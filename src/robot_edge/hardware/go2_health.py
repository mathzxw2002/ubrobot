"""Read-only Go2 base health via injected system probe (Task 2, read-only).

Go2 connects to the dock over Unitree DDS (CycloneDDS); the dock bridge
exposes odometry/IMU/joint state to ROS 2 topics. This module never calls
SportClient, never enables velocity mode, and never moves the dog: it only
maps read-only evidence (connected / standing / odometry / body velocity /
imu / orientation / local-stop readiness) onto capability and telemetry
snapshots. Disconnected, stale and abnormal evidence is never reported
healthy (fail-closed).

The probe is injected so workstation tests can fake it; the real probe is a
thin read-only system check (ROS topic reads on the dock side). No
unitree_sdk2py / piper_sdk / rclpy import anywhere in this package.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Protocol

from robot_edge.ros.context import RosGraph
from ubrobot_contracts.capabilities import (
    CapabilityAvailability,
    CapabilityHealth,
    CapabilityName,
    CapabilitySnapshot,
    ExecutionMode,
)
from ubrobot_contracts.telemetry import (
    TelemetryChannel,
    TelemetrySnapshot,
    TelemetryState,
    TimestampedSample,
)

# Evidence is only trustworthy while fresh; anything older is fail-closed.
MAX_EVIDENCE_AGE_SEC = 2.0
# Go2 must be essentially stationary for a stationary-base grasp. Any
# measured speed above this (m/s / rad/s) fails closed.
MAX_BODY_SPEED_MPS = 0.01
MAX_BODY_SPEED_RADPS = 0.02
# Orientation (roll/pitch) beyond this (radians) means the dog is not level.
MAX_ORIENTATION_RAD = 0.35  # ~20 deg


class Go2SystemProbe(Protocol):
    """Read-only view of the Go2 system state (no movement methods).

    Implementations must return only JSON-serializable values and must never
    expose a movement/velocity/stand method.
    """

    def connected(self) -> bool:
        """Whether the dock-to-Go2 bridge link is up."""
        ...

    def standing(self) -> bool:
        """Whether the Go2 is in a standing pose (velocity mode valid)."""
        ...

    def odometry(self) -> dict | None:
        """Latest odometry snapshot, or None when unavailable.

        Expected keys: ``x``, ``y``, ``yaw``, ``age_sec``.
        """
        ...

    def body_velocity(self) -> tuple[float, float, float] | None:
        """(vx, vy, vyaw) body velocity in m/s / rad/s, or None."""
        ...

    def imu(self) -> dict | None:
        """Latest IMU snapshot, or None when unavailable.

        Expected keys: ``roll_deg``, ``pitch_deg``, ``yaw_deg``, ``age_sec``.
        """
        ...

    def body_orientation(self) -> tuple[float, float, float] | None:
        """(roll, pitch, yaw) body orientation in radians, or None."""
        ...

    def local_stop_ready(self) -> bool:
        """Whether the bound local stop (E-stop) is ready."""
        ...


class Go2Health:
    """Maps the Go2 read-only probe onto capability/telemetry snapshots."""

    def __init__(
        self,
        probe: Go2SystemProbe,
        *,
        max_age_sec: float = MAX_EVIDENCE_AGE_SEC,
    ) -> None:
        self._probe = probe
        self._max_age_sec = max_age_sec

    def capability(self, *, now: datetime | None = None) -> CapabilitySnapshot:
        now = now or datetime.now(timezone.utc)
        detail, availability, health = self._evidence(now)
        return CapabilitySnapshot(
            name=CapabilityName.NAVIGATION,
            availability=availability,
            health=health,
            execution_mode=ExecutionMode.HARDWARE,
            required_resources=[
                "go2_bridge",
                "odometry",
                "imu",
                "safety_control",
            ],
            hardware_authority=False,  # M6: read-only, no command authority
            detail=detail,
            last_updated=now,
        )

    def telemetry(
        self, *, now: datetime | None = None
    ) -> dict[TelemetryChannel, TelemetrySnapshot]:
        now = now or datetime.now(timezone.utc)
        caps = self.capability(now=now)
        odom = self._probe.odometry()
        if caps.availability == CapabilityAvailability.AVAILABLE and odom:
            state = TelemetryState.AVAILABLE
            value = {
                "source": "robot-edge:probe",
                "x": odom.get("x"),
                "y": odom.get("y"),
                "yaw": odom.get("yaw"),
                "age_sec": odom.get("age_sec"),
            }
        else:
            state = TelemetryState.DISCONNECTED
            value = {"source": "robot-edge:probe", "detail": caps.detail}
        return {
            TelemetryChannel.ODOMETRY: TelemetrySnapshot(
                channel=TelemetryChannel.ODOMETRY,
                latest=TimestampedSample(timestamp=now, state=state, value=value),
                sequence=1,
            )
        }

    # ------------------------------------------------------------------ internal

    def _evidence(
        self, now: datetime
    ) -> tuple[str, CapabilityAvailability, CapabilityHealth]:
        probe = self._probe
        if not probe.connected():
            return (
                "Go2 bridge disconnected",
                CapabilityAvailability.DISCONNECTED,
                CapabilityHealth.UNKNOWN,
            )
        if not probe.standing():
            return (
                "Go2 not standing (velocity mode not valid)",
                CapabilityAvailability.UNAVAILABLE,
                CapabilityHealth.UNHEALTHY,
            )
        odom = probe.odometry()
        if odom is None or not isinstance(odom, dict):
            return (
                "Go2 odometry unavailable",
                CapabilityAvailability.UNAVAILABLE,
                CapabilityHealth.UNHEALTHY,
            )
        if not _age_is_fresh(odom.get("age_sec"), self._max_age_sec):
            return (
                "Go2 odometry stale",
                CapabilityAvailability.UNAVAILABLE,
                CapabilityHealth.UNHEALTHY,
            )
        velocity = probe.body_velocity()
        if velocity is None or not _body_is_stationary(velocity):
            return (
                f"Go2 base moving: velocity={velocity}",
                CapabilityAvailability.UNAVAILABLE,
                CapabilityHealth.UNHEALTHY,
            )
        imu = probe.imu()
        if imu is None or not isinstance(imu, dict):
            return (
                "Go2 IMU unavailable",
                CapabilityAvailability.UNAVAILABLE,
                CapabilityHealth.UNHEALTHY,
            )
        if not _age_is_fresh(imu.get("age_sec"), self._max_age_sec):
            return (
                "Go2 IMU stale",
                CapabilityAvailability.UNAVAILABLE,
                CapabilityHealth.UNHEALTHY,
            )
        orientation = probe.body_orientation()
        if orientation is None or not _orientation_within_limits(orientation):
            return (
                f"Go2 body orientation out of limits: {orientation}",
                CapabilityAvailability.UNAVAILABLE,
                CapabilityHealth.UNHEALTHY,
            )
        if not probe.local_stop_ready():
            return (
                "Go2 local stop not ready",
                CapabilityAvailability.UNAVAILABLE,
                CapabilityHealth.UNHEALTHY,
            )
        return (
            "Go2 connected, standing, stationary, orientation nominal, local stop ready",
            CapabilityAvailability.AVAILABLE,
            CapabilityHealth.HEALTHY,
        )


@dataclass(frozen=True)
class PlatformAuthority:
    """Fail-closed authority verdict for the go2_piper platform."""

    granted: bool
    detail: str


class Go2PiperHealth:
    """Composite read-only health for the go2_piper platform.

    Authority is granted only when the Go2 base, the Piper arm, the TF tree
    and the local stop are all healthy/complete/bound. Any missing evidence
    is fail-closed (no authority). This is a READ-ONLY health check; it
    grants no command execution authority by itself.
    """

    def __init__(
        self,
        *,
        go2_health: Go2Health,
        piper_health,
        tf_complete: bool,
        local_stop_bound: bool,
    ) -> None:
        self._go2 = go2_health
        self._piper = piper_health
        self._tf_complete = tf_complete
        self._local_stop_bound = local_stop_bound

    def authority(self, *, now: datetime | None = None) -> PlatformAuthority:
        now = now or datetime.now(timezone.utc)
        go2 = self._go2.capability(now=now)
        if go2.availability != CapabilityAvailability.AVAILABLE:
            return PlatformAuthority(False, f"go2 not serviceable: {go2.detail}")
        piper = self._piper.capability(now=now)
        if piper.availability != CapabilityAvailability.AVAILABLE:
            return PlatformAuthority(False, f"piper not serviceable: {piper.detail}")
        if not self._tf_complete:
            return PlatformAuthority(False, "TF tree incomplete")
        if not self._local_stop_bound:
            return PlatformAuthority(False, "local stop not bound")
        return PlatformAuthority(
            True,
            "go2 standing + piper torque off + TF complete + local stop bound",
        )

    def capability(
        self, name: CapabilityName, *, now: datetime | None = None
    ) -> CapabilitySnapshot:
        now = now or datetime.now(timezone.utc)
        auth = self.authority(now=now)
        availability = (
            CapabilityAvailability.AVAILABLE
            if auth.granted
            else CapabilityAvailability.UNAVAILABLE
        )
        health = (
            CapabilityHealth.HEALTHY if auth.granted else CapabilityHealth.UNHEALTHY
        )
        return CapabilitySnapshot(
            name=name,
            availability=availability,
            health=health,
            execution_mode=ExecutionMode.HARDWARE,
            required_resources=["go2_bridge", "can", "piper_driver", "safety_control"],
            hardware_authority=False,  # read-only health, no command authority
            detail=auth.detail,
            last_updated=now,
        )


def _age_is_fresh(age: object, max_age_sec: float) -> bool:
    try:
        value = float(age)
    except (TypeError, ValueError):
        return False
    return value >= 0.0 and value <= max_age_sec


def _body_is_stationary(velocity: tuple[float, float, float]) -> bool:
    try:
        vx, vy, vyaw = (float(v) for v in velocity)
    except (TypeError, ValueError):
        return False
    return (
        abs(vx) <= MAX_BODY_SPEED_MPS
        and abs(vy) <= MAX_BODY_SPEED_MPS
        and abs(vyaw) <= MAX_BODY_SPEED_RADPS
    )


def _orientation_within_limits(orientation: tuple[float, float, float]) -> bool:
    try:
        roll, pitch, _ = (float(v) for v in orientation)
    except (TypeError, ValueError):
        return False
    return abs(roll) <= MAX_ORIENTATION_RAD and abs(pitch) <= MAX_ORIENTATION_RAD


class RosGo2Probe:
    """Real Go2 read-only probe backed by the dock ROS 2 graph.

    Subscribes (read-only) the dock bridge topics exposed from the Unitree
    DDS interface (Task 1 inventory): odometry, IMU and joint states. It
    never publishes, never calls SportClient, and has no movement method.

    ``local_stop_ready`` may be injected as a callable (e.g. the Robot Edge
    E-stop binding); when not provided it defaults to False so authority is
    never granted while the stop is unbound (fail-closed).
    """

    def __init__(
        self,
        graph: RosGraph,
        *,
        odom_topic: str = "/odom",
        imu_topic: str = "/imu",
        joint_states_topic: str = "/joint_states",
        local_stop_ready: Callable[[], bool] | None = None,
        max_age_sec: float = MAX_EVIDENCE_AGE_SEC,
    ) -> None:
        self._graph = graph
        self._odom_topic = odom_topic
        self._imu_topic = imu_topic
        self._joint_states_topic = joint_states_topic
        self._local_stop_ready = local_stop_ready
        self._max_age_sec = max_age_sec

    def connected(self) -> bool:
        # The bridge is "connected" when it publishes odometry and joint
        # state; both are required for any base-side evidence.
        return self._graph.has_topic(self._odom_topic) and self._graph.has_topic(
            self._joint_states_topic
        )

    def standing(self) -> bool:
        # Read-only proxy for "standing": joint states are present and the
        # odometry topic is fresh. The bridge may later expose an explicit
        # standing signal; until then this evidence-based proxy holds.
        raw = self._graph.read_topic(self._joint_states_topic)
        if raw is None:
            return False
        if not isinstance(raw.get("name"), list) or not raw["name"]:
            return False
        odom = self._graph.read_topic(self._odom_topic)
        age = _topic_age_sec(odom)
        return age is not None and 0.0 <= age <= self._max_age_sec

    def odometry(self) -> dict | None:
        raw = self._graph.read_topic(self._odom_topic)
        if raw is None:
            return None
        age = _topic_age_sec(raw)
        pose = raw.get("pose") or {}
        if isinstance(pose, dict) and isinstance(pose.get("pose"), dict):
            pose = pose["pose"]
        position = pose.get("position") if isinstance(pose, dict) else None
        return {
            "x": position.get("x") if isinstance(position, dict) else None,
            "y": position.get("y") if isinstance(position, dict) else None,
            "age_sec": age,
        }

    def body_velocity(self) -> tuple[float, float, float] | None:
        raw = self._graph.read_topic(self._odom_topic)
        if raw is None:
            return None
        twist = raw.get("twist") or {}
        if isinstance(twist, dict) and isinstance(twist.get("twist"), dict):
            twist = twist["twist"]
        linear = twist.get("linear") if isinstance(twist, dict) else None
        angular = twist.get("angular") if isinstance(twist, dict) else None
        if not isinstance(linear, dict) or not isinstance(angular, dict):
            return None
        try:
            return (
                float(linear.get("x", 0.0)),
                float(linear.get("y", 0.0)),
                float(angular.get("z", 0.0)),
            )
        except (TypeError, ValueError):
            return None

    def imu(self) -> dict | None:
        raw = self._graph.read_topic(self._imu_topic)
        if raw is None:
            return None
        return {"age_sec": _topic_age_sec(raw)}

    def body_orientation(self) -> tuple[float, float, float] | None:
        raw = self._graph.read_topic(self._imu_topic)
        if raw is None:
            return None
        orientation = raw.get("orientation")
        if not isinstance(orientation, dict):
            return None
        try:
            return (
                float(orientation.get("x", 0.0)),
                float(orientation.get("y", 0.0)),
                float(orientation.get("z", 0.0)),
            )
        except (TypeError, ValueError):
            return None

    def local_stop_ready(self) -> bool:
        if self._local_stop_ready is None:
            return False
        try:
            return bool(self._local_stop_ready())
        except Exception:
            return False


def _topic_age_sec(raw: dict | None) -> float | None:
    """Age (seconds) of a message header stamp vs now, or None."""
    if not isinstance(raw, dict):
        return None
    header = raw.get("header")
    stamp = header.get("stamp") if isinstance(header, dict) else None
    if not isinstance(stamp, dict):
        return None
    try:
        stamp_sec = float(stamp.get("sec", 0)) + float(stamp.get("nanosec", 0)) / 1e9
    except (TypeError, ValueError):
        return None
    return round(datetime.now(timezone.utc).timestamp() - stamp_sec, 3)
