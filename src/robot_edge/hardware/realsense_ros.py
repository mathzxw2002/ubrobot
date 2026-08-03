"""RealSense health via ROS metadata only (M6, read-only).

Reads camera_info topics from the ROS graph and validates dimensions,
encoding, frame IDs, and calibration presence. Raw depth/color frames never
cross this boundary. No pyrealsense2 import (see import-boundary tests).
"""

from __future__ import annotations

from datetime import datetime, timezone

from ubrobot_contracts.telemetry import (
    TelemetryChannel,
    TelemetrySnapshot,
    TelemetryState,
    TimestampedSample,
)

from robot_edge.ros.context import RosGraph

# Measured live topics on the Raspberry Pi (2026-08-03): the bringup starts
# the RealSense node under a double /camera/camera namespace.
_COLOR_INFO_TOPIC = "/camera/camera/color/camera_info"
_DEPTH_INFO_TOPIC = "/camera/camera/depth/camera_info"
# Common Intel RealSense optical frame IDs (checked, not assumed).
_EXPECTED_COLOR_FRAME = "camera_color_optical_frame"
_EXPECTED_DEPTH_FRAME = "camera_depth_optical_frame"


class RealsenseHealthReader:
    """Validates camera metadata truthfully as available/stale/disconnected."""

    def __init__(
        self,
        graph: RosGraph,
        *,
        max_age_sec: float = 2.0,
        color_info_topic: str = _COLOR_INFO_TOPIC,
        depth_info_topic: str = _DEPTH_INFO_TOPIC,
    ) -> None:
        self._graph = graph
        self._max_age_sec = max_age_sec
        self._color_topic = color_info_topic
        self._depth_topic = depth_info_topic

    def snapshot(self, *, now: datetime | None = None) -> dict[TelemetryChannel, TelemetrySnapshot]:
        """Return camera + depth channel snapshots with truthful state."""
        now = now or datetime.now(timezone.utc)
        result = {
            TelemetryChannel.CAMERA: self._channel_snapshot(
                self._color_topic,
                _EXPECTED_COLOR_FRAME,
                now,
                kind="color",
            ),
            TelemetryChannel.DEPTH: self._channel_snapshot(
                self._depth_topic,
                _EXPECTED_DEPTH_FRAME,
                now,
                kind="depth",
            ),
        }
        return result

    def _channel_snapshot(
        self,
        topic: str,
        expected_frame: str,
        now: datetime,
        *,
        kind: str,
    ) -> TelemetrySnapshot:
        if not self._graph.has_topic(topic):
            return TelemetrySnapshot(
                channel=(
                    TelemetryChannel.CAMERA if kind == "color" else TelemetryChannel.DEPTH
                ),
                latest=TimestampedSample(
                    timestamp=now,
                    state=TelemetryState.DISCONNECTED,
                    value={
                        "source": "robot-edge:ros",
                        "detail": f"camera_info topic missing: {topic}",
                        "kind": kind,
                    },
                ),
                sequence=0,
            )
        raw = self._graph.read_topic(topic)
        if raw is None:
            return TelemetrySnapshot(
                channel=(
                    TelemetryChannel.CAMERA if kind == "color" else TelemetryChannel.DEPTH
                ),
                latest=TimestampedSample(
                    timestamp=now,
                    state=TelemetryState.DISCONNECTED,
                    value={
                        "source": "robot-edge:ros",
                        "detail": f"no camera_info message on {topic}",
                        "kind": kind,
                    },
                ),
                sequence=0,
            )
        value = self._build_value(raw, expected_frame, kind)
        # Age from the message stamp; a fresh-but-old stamp is STALE.
        state = TelemetryState.AVAILABLE
        stamp = _message_stamp(raw)
        if stamp is not None:
            try:
                stamp_dt = datetime.fromtimestamp(stamp, tz=timezone.utc)
                age = (now - stamp_dt).total_seconds()
                value["age_sec"] = round(age, 3)
                if age > self._max_age_sec:
                    state = TelemetryState.STALE
            except (ValueError, OverflowError, OSError):
                state = TelemetryState.STALE
        return TelemetrySnapshot(
            channel=(
                TelemetryChannel.CAMERA if kind == "color" else TelemetryChannel.DEPTH
            ),
            latest=TimestampedSample(timestamp=now, state=state, value=value),
            sequence=1,
        )

    @staticmethod
    def _build_value(raw: dict, expected_frame: str, kind: str) -> dict:
        width = raw.get("width")
        height = raw.get("height")
        frame_id = raw.get("header", {}).get("frame_id") if isinstance(raw.get("header"), dict) else None
        # Calibration: a valid K is a non-zero 3x3 matrix.
        k = raw.get("k") if isinstance(raw.get("k"), list) else []
        calibrated = isinstance(k, list) and len(k) == 9 and any(v for v in k)
        return {
            "source": "robot-edge:ros",
            "kind": kind,
            "width": width,
            "height": height,
            "encoding": raw.get("encoding"),
            "distortion_model": raw.get("distortion_model"),
            "frame_id": frame_id,
            "frame_matches_expected": frame_id == expected_frame,
            "calibrated": bool(calibrated),
        }


def _message_stamp(raw: dict) -> float | None:
    """Extract a ROS stamp (header.stamp.sec + nanosec) as float, if present."""
    header = raw.get("header")
    if not isinstance(header, dict):
        return None
    stamp = header.get("stamp")
    if not isinstance(stamp, dict):
        return None
    sec = stamp.get("sec")
    nanosec = stamp.get("nanosec")
    if isinstance(sec, (int, float)) and isinstance(nanosec, (int, float)):
        return float(sec) + float(nanosec) / 1e9
    return None
