"""Read-only Go2 telemetry mapping from dock bridge topics (Task 2).

The Go2 dock bridge exposes odometry / IMU / joint state on ROS 2 topics
(topic names confirmed in Task 1 inventory; defaults below match the
standard bridge naming and are overridable per deployment). This module maps
those read-only topic snapshots onto the shared telemetry channels. It never
publishes, never subscribes to control topics, and never moves the dog.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from robot_edge.ros.context import RosGraph
from ubrobot_contracts.telemetry import (
    TelemetryChannel,
    TelemetrySnapshot,
    TelemetryState,
    TimestampedSample,
)

# Bridge output topics (Task 1 inventory defaults; overridable per dock).
GO2_ODOM_TOPIC = "/odom"
GO2_IMU_TOPIC = "/imu"
GO2_JOINT_STATES_TOPIC = "/joint_states"

_TOPIC_MAP: dict[str, TelemetryChannel] = {
    GO2_ODOM_TOPIC: TelemetryChannel.ODOMETRY,
    GO2_IMU_TOPIC: TelemetryChannel.ODOMETRY,  # orientation supplements odometry
    GO2_JOINT_STATES_TOPIC: TelemetryChannel.JOINT_STATES,
}


class Go2Telemetry:
    """Maps Go2 bridge topics onto shared telemetry (read-only)."""

    def __init__(
        self,
        graph: RosGraph,
        *,
        odom_topic: str = GO2_ODOM_TOPIC,
        imu_topic: str = GO2_IMU_TOPIC,
        joint_states_topic: str = GO2_JOINT_STATES_TOPIC,
        max_age_sec: float = 2.0,
    ) -> None:
        self._graph = graph
        self._topics = {
            TelemetryChannel.ODOMETRY: odom_topic,
            TelemetryChannel.JOINT_STATES: joint_states_topic,
        }
        self._imu_topic = imu_topic
        self._max_age_sec = max_age_sec

    def snapshot(
        self, *, now: datetime | None = None
    ) -> dict[TelemetryChannel, TelemetrySnapshot]:
        now = now or datetime.now(timezone.utc)
        result: dict[TelemetryChannel, TelemetrySnapshot] = {}
        for channel, topic in self._topics.items():
            result[channel] = self._channel_snapshot(channel, topic, now)
        return result

    # ------------------------------------------------------------------ internal

    def _channel_snapshot(
        self, channel: TelemetryChannel, topic: str, now: datetime
    ) -> TelemetrySnapshot:
        if not self._graph.has_topic(topic):
            return self._snapshot(
                now,
                channel,
                TelemetryState.DISCONNECTED,
                {"detail": f"topic missing ({topic})"},
            )
        raw = self._graph.read_topic(topic)
        if raw is None:
            return self._snapshot(
                now,
                channel,
                TelemetryState.DISCONNECTED,
                {"detail": f"no message on {topic}"},
            )
        value = self._value_for(channel, topic, raw)
        age = value.get("age_sec")
        if not isinstance(age, (int, float)) or age < 0.0 or age > self._max_age_sec:
            return self._snapshot(now, channel, TelemetryState.STALE, value)
        return self._snapshot(now, channel, TelemetryState.AVAILABLE, value)

    def _value_for(
        self, channel: TelemetryChannel, topic: str, raw: dict[str, Any]
    ) -> dict[str, Any]:
        if channel == TelemetryChannel.ODOMETRY:
            return self._odometry_value(raw)
        if channel == TelemetryChannel.JOINT_STATES:
            return {
                "source": "robot-edge:ros",
                "topic": topic,
                "names": raw.get("name") or [],
                "positions": raw.get("position") or [],
                "motor_count": len(raw.get("name") or []),
                "age_sec": _message_age_sec(raw),
            }
        return {
            "source": "robot-edge:ros",
            "topic": topic,
            "age_sec": _message_age_sec(raw),
        }

    def _odometry_value(self, raw: dict[str, Any]) -> dict[str, Any]:
        pose = raw.get("pose") or {}
        if isinstance(pose, dict) and isinstance(pose.get("pose"), dict):
            pose = pose["pose"]
        position = pose.get("position") if isinstance(pose, dict) else None
        orientation = pose.get("orientation") if isinstance(pose, dict) else None
        twist = raw.get("twist") or {}
        if isinstance(twist, dict) and isinstance(twist.get("twist"), dict):
            twist = twist["twist"]
        linear = twist.get("linear") if isinstance(twist, dict) else None
        return {
            "source": "robot-edge:ros",
            "x": position.get("x") if isinstance(position, dict) else None,
            "y": position.get("y") if isinstance(position, dict) else None,
            "yaw": _quaternion_yaw(orientation)
            if isinstance(orientation, dict)
            else None,
            "vx": linear.get("x") if isinstance(linear, dict) else None,
            "age_sec": _message_age_sec(raw),
        }

    def _snapshot(
        self,
        now: datetime,
        channel: TelemetryChannel,
        state: TelemetryState,
        value: dict[str, Any],
    ) -> TelemetrySnapshot:
        value.setdefault("source", "robot-edge:ros")
        return TelemetrySnapshot(
            channel=channel,
            latest=TimestampedSample(timestamp=now, state=state, value=value),
            sequence=1 if state == TelemetryState.AVAILABLE else 0,
        )


def _message_age_sec(raw: dict[str, Any]) -> float | None:
    stamp = (
        raw.get("header", {}).get("stamp")
        if isinstance(raw.get("header"), dict)
        else None
    )
    if not isinstance(stamp, dict):
        return None
    try:
        stamp_sec = float(stamp.get("sec", 0)) + float(stamp.get("nanosec", 0)) / 1e9
    except (TypeError, ValueError):
        return None
    now = datetime.now(timezone.utc).timestamp()
    return round(now - stamp_sec, 3)


def _quaternion_yaw(orientation: dict) -> float | None:
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
