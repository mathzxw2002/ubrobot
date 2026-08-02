"""Telemetry contracts between Operator Console and Robot Edge."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class TelemetryChannel(str, Enum):
    """Known telemetry channels."""

    CAMERA = "camera"
    DEPTH = "depth"
    ODOMETRY = "odometry"
    JOINT_STATES = "joint_states"
    NAVIGATION_LEASE = "navigation_lease"
    CAPABILITY_HEALTH = "capability_health"


class TelemetryState(str, Enum):
    """State of a telemetry channel."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    STALE = "stale"
    DISCONNECTED = "disconnected"


class TimestampedSample(BaseModel):
    """A single timestamped telemetry sample."""

    timestamp: datetime
    state: TelemetryState
    value: Any = None  # Must be JSON-serializable

    model_config = {
        "frozen": True,
    }

    @field_validator("timestamp")
    @classmethod
    def ensure_timezone_aware(cls, v: datetime) -> datetime:
        """All timestamps must be timezone-aware."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class TelemetrySnapshot(BaseModel):
    """JSON-safe snapshot of a single telemetry channel."""

    channel: TelemetryChannel
    latest: TimestampedSample | None = None
    sequence: int = 0
    age_sec: float | None = None

    model_config = {
        "frozen": True,
    }


class TelemetryInventory(BaseModel):
    """JSON-safe inventory of all telemetry channels."""

    channels: dict[TelemetryChannel, TelemetrySnapshot] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {
        "frozen": True,
    }

    @field_validator("last_updated")
    @classmethod
    def ensure_timezone_aware(cls, v: datetime) -> datetime:
        """All timestamps must be timezone-aware."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v
