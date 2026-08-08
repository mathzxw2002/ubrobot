"""Read-only ROS telemetry mapping into shared contracts (M6)."""

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

# Topic -> (channel, value extractor). All reads are strictly read-only.
# /lekiwi_base_controller/odom is the measured live topic (2026-08-03); the
# historical /odom/wheel design topic is kept for graph compatibility.
_TOPIC_MAP: dict[str, tuple[TelemetryChannel, str]] = {
    "/lekiwi_base_controller/odom": (TelemetryChannel.ODOMETRY, "wheel_odometry"),
    "/odom/wheel": (TelemetryChannel.ODOMETRY, "wheel_odometry"),
    "/odom": (TelemetryChannel.ODOMETRY, "odometry"),
    "/joint_states": (TelemetryChannel.JOINT_STATES, "joint_states"),
    # Measured live topics (2026-08-03); legacy paths kept for compatibility.
    "/camera/camera/color/camera_info": (TelemetryChannel.CAMERA, "camera_info"),
    "/camera/camera/depth/camera_info": (TelemetryChannel.DEPTH, "depth_info"),
    "/camera/camera_info": (TelemetryChannel.CAMERA, "camera_info"),
    "/camera/depth/camera_info": (TelemetryChannel.DEPTH, "depth_info"),
}


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


def _unwrap_covariance(value: Any) -> dict:
    """Unwrap PoseWithCovariance/TwistWithCovariance double nesting.

    rclpy Odometry serializes ``pose.pose.position`` and
    ``twist.twist.linear``; the extractors below want the inner pose/twist.
    """
    if isinstance(value, dict):
        inner = value.get("pose")
        if isinstance(inner, dict):
            return inner
        inner = value.get("twist")
        if isinstance(inner, dict):
            return inner
    return value or {}


def _value_for(topic: str, kind: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON-safe channel value from the read topic snapshot."""
    if kind == "wheel_odometry":
        pose = _unwrap_covariance(raw.get("pose"))
        twist = _unwrap_covariance(raw.get("twist"))
        orientation = pose.get("orientation") if isinstance(pose, dict) else None
        return {
            "source": "robot-edge:ros",
            "topic": topic,
            "x": pose.get("position", {}).get("x") if isinstance(pose, dict) else None,
            "y": pose.get("position", {}).get("y") if isinstance(pose, dict) else None,
            "yaw": _quaternion_yaw(orientation)
            if isinstance(orientation, dict)
            else None,
            "vx": twist.get("linear", {}).get("x") if isinstance(twist, dict) else None,
        }
    if kind == "odometry":
        pose = _unwrap_covariance(raw.get("pose"))
        return {
            "source": "robot-edge:ros",
            "topic": topic,
            "x": pose.get("position", {}).get("x") if isinstance(pose, dict) else None,
            "y": pose.get("position", {}).get("y") if isinstance(pose, dict) else None,
            "yaw": None,
        }
    if kind == "joint_states":
        return {
            "source": "robot-edge:ros",
            "topic": topic,
            "names": raw.get("name") or [],
            "positions": raw.get("position") or [],
        }
    if kind == "camera_info":
        return {
            "source": "robot-edge:ros",
            "topic": topic,
            "width": raw.get("width"),
            "height": raw.get("height"),
            "distortion_model": raw.get("distortion_model"),
        }
    if kind == "depth_info":
        return {
            "source": "robot-edge:ros",
            "topic": topic,
            "width": raw.get("width"),
            "height": raw.get("height"),
        }
    return {"source": "robot-edge:ros", "topic": topic}


class RosTelemetryReader:
    """Maps ROS topics onto the shared telemetry channels (read-only)."""

    def __init__(self, graph: RosGraph, *, read_timeout_sec: float = 1.0) -> None:
        self._graph = graph
        self._read_timeout_sec = read_timeout_sec
        # Channels that are not ROS-backed (navigation lease, capability
        # health) are reported unavailable here; Edge local state fills them.
        self._local_channels = {
            TelemetryChannel.NAVIGATION_LEASE,
            TelemetryChannel.CAPABILITY_HEALTH,
        }

    def snapshot(self) -> dict[TelemetryChannel, TelemetrySnapshot]:
        now = datetime.now(timezone.utc)
        result: dict[TelemetryChannel, TelemetrySnapshot] = {}
        seen: set[TelemetryChannel] = set()

        for topic, (channel, kind) in _TOPIC_MAP.items():
            if channel in seen:
                # First matching topic wins (e.g. /odom/wheel over /odom).
                continue
            if not self._graph.has_topic(topic):
                continue
            seen.add(channel)
            raw = self._graph.read_topic(topic)
            if raw is None:
                result[channel] = TelemetrySnapshot(
                    channel=channel,
                    latest=TimestampedSample(
                        timestamp=now,
                        state=TelemetryState.DISCONNECTED,
                        value={"source": "robot-edge:ros", "topic": topic},
                    ),
                    sequence=0,
                )
            else:
                result[channel] = TelemetrySnapshot(
                    channel=channel,
                    latest=TimestampedSample(
                        timestamp=now,
                        state=TelemetryState.AVAILABLE,
                        value=_value_for(topic, kind, raw),
                    ),
                    sequence=1,
                )

        # Any channel with no ROS backing and no local state is unavailable,
        # never healthy or disconnected-fabricated. All six contract channels
        # must appear explicitly so clients can distinguish "missing" from
        # "unknown".
        for channel in TelemetryChannel:
            if channel in seen:
                continue
            result[channel] = TelemetrySnapshot(
                channel=channel,
                latest=TimestampedSample(
                    timestamp=now,
                    state=TelemetryState.UNAVAILABLE,
                    value={
                        "source": "robot-edge:ros",
                        "detail": (
                            "no backing topic in read-only mode"
                            if channel not in self._local_channels
                            else "Edge local state unavailable (M6 read-only)"
                        ),
                    },
                ),
                sequence=0,
            )
        return result
