"""Read-only ROS telemetry mapping into shared contracts (M6)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ubrobot_contracts.telemetry import (
    TelemetryChannel,
    TelemetrySnapshot,
    TelemetryState,
    TimestampedSample,
)

from robot_edge.ros.context import RosGraph

# Topic -> (channel, value extractor). All reads are strictly read-only.
_TOPIC_MAP: dict[str, tuple[TelemetryChannel, str]] = {
    "/odom/wheel": (TelemetryChannel.ODOMETRY, "wheel_odometry"),
    "/odom": (TelemetryChannel.ODOMETRY, "odometry"),
    "/joint_states": (TelemetryChannel.JOINT_STATES, "joint_states"),
    "/camera/camera_info": (TelemetryChannel.CAMERA, "camera_info"),
    "/camera/depth/camera_info": (TelemetryChannel.DEPTH, "depth_info"),
}


def _value_for(topic: str, kind: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON-safe channel value from the read topic snapshot."""
    if kind == "wheel_odometry":
        pose = raw.get("pose") or {}
        twist = raw.get("twist") or {}
        return {
            "source": "robot-edge:ros",
            "topic": topic,
            "x": pose.get("position", {}).get("x") if isinstance(pose, dict) else None,
            "y": pose.get("position", {}).get("y") if isinstance(pose, dict) else None,
            "yaw": None,
            "vx": twist.get("linear", {}).get("x") if isinstance(twist, dict) else None,
        }
    if kind == "odometry":
        pose = raw.get("pose") or {}
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
