"""Robot Edge API contracts for command, lease, and safety control."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ubrobot_contracts import PROTOCOL_VERSION


class CommandState(str, Enum):
    """State of a command execution."""

    ACCEPTED = "accepted"
    QUEUED = "queued"
    PLANNING = "planning"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LeaseState(str, Enum):
    """State of a navigation lease."""

    NONE = "none"
    ACTIVE = "active"
    EXPIRED = "expired"
    RELEASED = "released"


class CommandRequest(BaseModel):
    """Request to execute a natural language command."""

    text: str
    correlation_id: str
    operator_id: str
    nonce: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    protocol_version: str = PROTOCOL_VERSION

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


class CommandAccepted(BaseModel):
    """Response when a command is accepted for execution."""

    command_id: str
    correlation_id: str | None = None
    protocol_version: str = PROTOCOL_VERSION

    model_config = {
        "frozen": True,
    }


class CommandEvent(BaseModel):
    """Event streamed during command execution."""

    command_id: str
    state: CommandState
    message: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sequence: int = 0
    protocol_version: str = PROTOCOL_VERSION

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


class CancelRequest(BaseModel):
    """Request to cancel an active command."""

    command_id: str
    correlation_id: str
    operator_id: str
    nonce: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    protocol_version: str = PROTOCOL_VERSION

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


class EmergencyStopRequest(BaseModel):
    """Request to trigger emergency stop (bypasses lease)."""

    correlation_id: str
    operator_id: str
    nonce: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    protocol_version: str = PROTOCOL_VERSION

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


class LeaseAcquireRequest(BaseModel):
    """Request to acquire or renew a navigation lease."""

    operator_id: str
    nonce: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_sec: float = 30.0
    protocol_version: str = PROTOCOL_VERSION

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


class LeaseRecord(BaseModel):
    """Current state of a navigation lease."""

    lease_id: str
    owner: str | None
    issued_at: datetime
    expires_at: datetime
    last_renewed_at: datetime
    state: LeaseState
    protocol_version: str = PROTOCOL_VERSION

    model_config = {
        "frozen": True,
    }

    @field_validator("issued_at", "expires_at", "last_renewed_at")
    @classmethod
    def ensure_timezone_aware(cls, v: datetime) -> datetime:
        """All timestamps must be timezone-aware."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class Heartbeat(BaseModel):
    """Periodic heartbeat to keep connections alive."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    protocol_version: str = PROTOCOL_VERSION

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


class ErrorResponse(BaseModel):
    """Error response from Robot Edge API."""

    code: str
    message: str
    correlation_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    protocol_version: str = PROTOCOL_VERSION

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


class HealthResponse(BaseModel):
    """Health check response."""

    status: str  # "healthy" | "degraded" | "unhealthy"
    execution_mode: str
    hardware_authority: bool
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    protocol_version: str = PROTOCOL_VERSION

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
