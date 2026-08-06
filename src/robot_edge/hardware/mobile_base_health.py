"""Mobile base health for the selected profile (M6, read-only).

Only the owner-selected profile is ever probed. M6 supports `lekiwi` and
`go2`; both are read-only. All reads are read-only: state/odometry/driver
health only, never movement commands.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math

from ubrobot_contracts.telemetry import (
    TelemetryChannel,
    TelemetrySnapshot,
    TelemetryState,
    TimestampedSample,
)

from robot_edge.ros.context import RosGraph

SUPPORTED_PROFILES = ("lekiwi", "go2")


def _quaternion_yaw(orientation: dict) -> float | None:
    """Yaw (radians) from a ROS geometry_msgs/Quaternion dict, or None."""
    try:
        x, y, z, w = (
            float(orientation["x"]),
            float(orientation["y"]),
            float(orientation["z"]),
            float(orientation["w"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    return round(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)), 4)

# lekiwi: /lekiwi_base_controller/odom (wheel odometry, measured live on the
# Raspberry Pi 2026-08-03; the ros2_control controller publishes under its
# own namespace, not the historical /odom/wheel design), /joint_states.
# go2: dock bridge output topics (Task 1 inventory defaults; the go2_piper
# bridge exposes odometry/IMU/joint state from the Unitree DDS topics).
_PROFILE_TOPICS: dict[str, dict[TelemetryChannel, tuple[str, str]]] = {
    "lekiwi": {
        TelemetryChannel.ODOMETRY: ("/lekiwi_base_controller/odom", "wheel odometry"),
        TelemetryChannel.JOINT_STATES: ("/joint_states", "motor joint states"),
    },
    "go2": {
        TelemetryChannel.ODOMETRY: ("/odom", "go2 bridge odometry"),
        TelemetryChannel.JOINT_STATES: ("/joint_states", "go2 leg joint states"),
    },
}


class MobileBaseHealth:
    """Maps the selected mobile base's read-only state to telemetry."""

    def __init__(
        self,
        graph: RosGraph,
        *,
        profile: str = "lekiwi",
        max_age_sec: float = 2.0,
    ) -> None:
        if profile not in SUPPORTED_PROFILES:
            raise ValueError(
                f"unsupported mobile base profile {profile!r}; "
                f"supported: {', '.join(SUPPORTED_PROFILES)}"
            )
        self._profile = profile
        self._graph = graph
        self._max_age_sec = max_age_sec
        self._topics = _PROFILE_TOPICS[profile]

    @property
    def profile(self) -> str:
        return self._profile

    def snapshot(self, *, now: datetime | None = None) -> dict[TelemetryChannel, TelemetrySnapshot]:
        now = now or datetime.now(timezone.utc)
        result: dict[TelemetryChannel, TelemetrySnapshot] = {}
        for channel, (topic, detail) in self._topics.items():
            result[channel] = self._channel_snapshot(channel, topic, detail, now)
        return result

    def _channel_snapshot(
        self,
        channel: TelemetryChannel,
        topic: str,
        detail: str,
        now: datetime,
    ) -> TelemetrySnapshot:
        if not self._graph.has_topic(topic):
            return TelemetrySnapshot(
                channel=channel,
                latest=TimestampedSample(
                    timestamp=now,
                    state=TelemetryState.DISCONNECTED,
                    value={
                        "source": "robot-edge:ros",
                        "profile": self._profile,
                        "detail": f"{detail}: topic missing ({topic})",
                    },
                ),
                sequence=0,
            )
        raw = self._graph.read_topic(topic)
        if raw is None:
            return TelemetrySnapshot(
                channel=channel,
                latest=TimestampedSample(
                    timestamp=now,
                    state=TelemetryState.DISCONNECTED,
                    value={
                        "source": "robot-edge:ros",
                        "profile": self._profile,
                        "detail": f"{detail}: no message on {topic}",
                    },
                ),
                sequence=0,
            )
        value = {
            "source": "robot-edge:ros",
            "profile": self._profile,
            "topic": topic,
            "detail": detail,
        }
        if channel == TelemetryChannel.ODOMETRY:
            # Unwrap PoseWithCovariance/TwistWithCovariance double nesting
            # (pose.pose.position, twist.twist.linear from rclpy Odometry).
            pose = raw.get("pose") or {}
            if isinstance(pose, dict) and isinstance(pose.get("pose"), dict):
                pose = pose["pose"]
            position = pose.get("position") if isinstance(pose, dict) else None
            if isinstance(position, dict):
                value["x"] = position.get("x")
                value["y"] = position.get("y")
            orientation = pose.get("orientation") if isinstance(pose, dict) else None
            if isinstance(orientation, dict):
                value["yaw"] = _quaternion_yaw(orientation)
            twist = raw.get("twist") or {}
            if isinstance(twist, dict) and isinstance(twist.get("twist"), dict):
                twist = twist["twist"]
            linear = twist.get("linear") if isinstance(twist, dict) else None
            if isinstance(linear, dict):
                value["vx"] = linear.get("x")
        elif channel == TelemetryChannel.JOINT_STATES:
            value["names"] = raw.get("name") or []
            value["positions"] = raw.get("position") or []
            value["velocities"] = raw.get("velocity") or []
            value["motor_count"] = len(raw.get("name") or [])
        state = TelemetryState.AVAILABLE
        stamp = raw.get("header", {}).get("stamp") if isinstance(raw.get("header"), dict) else None
        if isinstance(stamp, dict) and isinstance(stamp.get("sec"), (int, float)):
            try:
                stamp_dt = datetime.fromtimestamp(
                    float(stamp["sec"]) + float(stamp.get("nanosec", 0)) / 1e9,
                    tz=timezone.utc,
                )
                value["age_sec"] = round((now - stamp_dt).total_seconds(), 3)
                if (now - stamp_dt).total_seconds() > self._max_age_sec:
                    state = TelemetryState.STALE
            except (ValueError, OverflowError, OSError):
                state = TelemetryState.STALE
        return TelemetrySnapshot(
            channel=channel,
            latest=TimestampedSample(timestamp=now, state=state, value=value),
            sequence=1,
        )
