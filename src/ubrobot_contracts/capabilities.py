"""Capability inventory contracts between Operator Console and Robot Edge."""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class CapabilityName(str, Enum):
    """Restricted set of capability names."""

    NAVIGATION = "navigation"
    GRASP = "grasp"
    OBSERVATION = "observation"
    FOLLOW = "follow"
    STOP = "stop"


class CapabilityAvailability(str, Enum):
    """Availability state of a capability."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DISCONNECTED = "disconnected"
    UNKNOWN = "unknown"


class CapabilityHealth(str, Enum):
    """Health state of a capability."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ExecutionMode(str, Enum):
    """Execution mode of the system."""

    MOCK = "mock"
    FIXTURE = "fixture"
    REMOTE = "remote"
    HARDWARE = "hardware"


class CapabilitySnapshot(BaseModel):
    """JSON-safe snapshot of a single capability's state."""

    name: CapabilityName
    availability: CapabilityAvailability
    health: CapabilityHealth
    execution_mode: ExecutionMode
    required_resources: list[str] = Field(default_factory=list)
    hardware_authority: bool = False
    detail: str = ""
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {
        "frozen": True,
        "json_encoders": {datetime: lambda v: v.isoformat()},
    }

    @field_validator("last_updated")
    @classmethod
    def ensure_timezone_aware(cls, v: datetime) -> datetime:
        """All timestamps must be timezone-aware."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class CapabilityInventory(BaseModel):
    """JSON-safe inventory of all capabilities."""

    capabilities: dict[CapabilityName, CapabilitySnapshot] = Field(default_factory=dict)
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
