"""Serialized telemetry DTOs and fixture adapter.

This module intentionally has no ROS, RealSense, Piper, or Go2 imports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import math
from typing import Any, Mapping


class TelemetryState(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    STALE = "stale"
    DISCONNECTED = "disconnected"


def serialize_transport_value(value: Any) -> Any:
    """Return JSON-safe state or reject an SDK/runtime object explicitly."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("telemetry floats must be finite")
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return serialize_transport_value(to_dict())
    if isinstance(value, Mapping):
        return {
            str(key): serialize_transport_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [serialize_transport_value(item) for item in value]
    raise TypeError(
        f"telemetry value {type(value).__name__} is not transport serializable"
    )


class _SerializedDTO:
    def to_dict(self) -> dict[str, Any]:
        return serialize_transport_value(asdict(self))


@dataclass(frozen=True)
class CameraTelemetry(_SerializedDTO):
    state: TelemetryState
    source: str
    width: int | None = None
    height: int | None = None
    encoding: str | None = None
    frame_id: str | None = None
    detail: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    channel: str = field(default="camera", init=False)


@dataclass(frozen=True)
class DepthTelemetry(_SerializedDTO):
    state: TelemetryState
    source: str
    width: int | None = None
    height: int | None = None
    unit: str = "m"
    minimum: float | None = None
    maximum: float | None = None
    detail: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    channel: str = field(default="depth", init=False)


@dataclass(frozen=True)
class OdometryTelemetry(_SerializedDTO):
    state: TelemetryState
    source: str
    x: float | None = None
    y: float | None = None
    yaw: float | None = None
    linear_velocity: float | None = None
    angular_velocity: float | None = None
    frame_id: str = "odom"
    detail: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    channel: str = field(default="odometry", init=False)


@dataclass(frozen=True)
class JointStatesTelemetry(_SerializedDTO):
    state: TelemetryState
    source: str
    names: tuple[str, ...] = ()
    positions: tuple[float, ...] = ()
    velocities: tuple[float, ...] = ()
    efforts: tuple[float, ...] = ()
    detail: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    channel: str = field(default="joint_states", init=False)

    def __post_init__(self):
        for values in (self.positions, self.velocities, self.efforts):
            if values and len(values) != len(self.names):
                raise ValueError("joint value arrays must match joint names")


@dataclass(frozen=True)
class NavigationLeaseTelemetry(_SerializedDTO):
    state: TelemetryState
    source: str
    owner: str | None = None
    lease_id: str | None = None
    expires_at: datetime | None = None
    detail: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    channel: str = field(default="navigation_lease", init=False)


@dataclass(frozen=True)
class CapabilityHealthTelemetry(_SerializedDTO):
    state: TelemetryState
    source: str
    capabilities: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    detail: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    channel: str = field(default="capability_health", init=False)


@dataclass(frozen=True)
class ChannelStateTelemetry(_SerializedDTO):
    channel: str
    state: TelemetryState
    source: str = "fixture"
    detail: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FixtureTelemetryAdapter:
    """Read explicit fixture values; missing channels are disconnected."""

    CHANNELS = (
        "camera",
        "depth",
        "odometry",
        "joint_states",
        "navigation_lease",
        "capability_health",
    )

    def __init__(self, fixtures: Mapping[str, Any] | None = None):
        self._fixtures = dict(fixtures or {})
        unknown = set(self._fixtures) - set(self.CHANNELS)
        if unknown:
            raise ValueError(f"unknown fixture telemetry channels: {sorted(unknown)}")
        for channel, value in self._fixtures.items():
            serialized = serialize_transport_value(value)
            if not isinstance(serialized, dict):
                raise TypeError(f"fixture telemetry must serialize to an object: {channel}")
            if serialized.get("channel", channel) != channel:
                raise ValueError(f"fixture channel mismatch: {channel}")

    def read(self, channel: str):
        if channel not in self.CHANNELS:
            raise KeyError(f"unknown telemetry channel: {channel}")
        return self._fixtures.get(
            channel,
            ChannelStateTelemetry(
                channel=channel,
                state=TelemetryState.DISCONNECTED,
                detail="no fixture or robot-edge connection",
            ),
        )

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            channel: serialize_transport_value(self.read(channel))
            for channel in self.CHANNELS
        }

    def publish_all(self, telemetry_hub) -> None:
        for channel in self.CHANNELS:
            telemetry_hub.publish(channel, self.read(channel))
