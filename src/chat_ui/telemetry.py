"""Thread-safe in-process telemetry hub with transport-neutral snapshots.

For shared contracts between Operator Console and Robot Edge, see ubrobot_contracts.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

# Re-export from shared contracts for backward compatibility
from ubrobot_contracts.telemetry import TelemetryState

try:
    from .adapters.telemetry import serialize_transport_value
except ImportError:  # Direct-script compatibility.
    from adapters.telemetry import serialize_transport_value


@dataclass(frozen=True)
class TelemetrySample:
    channel: str
    value: Any
    sequence: int
    timestamp: datetime
    monotonic_time: float

    def to_dict(self, *, now: float | None = None, stale_after_sec: float = 3.0):
        current = time.monotonic() if now is None else now
        value = serialize_transport_value(self.value)
        age_sec = max(0.0, current - self.monotonic_time)
        reported_state = value.get("state") if isinstance(value, dict) else None
        if reported_state is None and isinstance(value, dict):
            available_flag = value.get("available")
            if available_flag is False:
                reported_state = TelemetryState.UNAVAILABLE.value
        if reported_state not in {state.value for state in TelemetryState}:
            reported_state = TelemetryState.AVAILABLE.value
        available = reported_state == TelemetryState.AVAILABLE.value
        stale = reported_state == TelemetryState.STALE.value or (
            available and age_sec > stale_after_sec
        )
        state = TelemetryState.STALE.value if stale else reported_state
        return {
            "channel": self.channel,
            "value": value,
            "sequence": self.sequence,
            "timestamp": self.timestamp.isoformat(),
            "age_sec": age_sec,
            "state": state,
            "available": state == TelemetryState.AVAILABLE.value,
            "stale": stale,
            "disconnected": state == TelemetryState.DISCONNECTED.value,
        }


class TelemetryHub:
    DEFAULT_CHANNELS = (
        "camera",
        "depth",
        "odometry",
        "joint_states",
        "navigation_lease",
        "capability_health",
    )

    def __init__(
        self,
        *,
        history_size: int = 120,
        stale_after_sec: float = 3.0,
        event_publisher: Callable[..., Any] | None = None,
    ):
        if history_size <= 0 or stale_after_sec <= 0:
            raise ValueError("telemetry limits must be positive")
        self._history_size = history_size
        self._stale_after_sec = stale_after_sec
        self._lock = threading.RLock()
        self._history: dict[str, deque[TelemetrySample]] = defaultdict(
            lambda: deque(maxlen=self._history_size)
        )
        self._sequences: dict[str, int] = defaultdict(int)
        self._event_publisher = event_publisher

    def publish(self, channel: str, value: Any) -> TelemetrySample:
        if not isinstance(channel, str) or not channel.strip():
            raise ValueError("telemetry channel must be non-empty")
        channel = channel.strip()
        serialized_value = serialize_transport_value(value)
        with self._lock:
            self._sequences[channel] += 1
            sample = TelemetrySample(
                channel=channel,
                value=serialized_value,
                sequence=self._sequences[channel],
                timestamp=datetime.now(timezone.utc),
                monotonic_time=time.monotonic(),
            )
            self._history[channel].append(sample)
        if self._event_publisher is not None:
            self._event_publisher(
                kind="telemetry.updated",
                source="telemetry_hub",
                payload={
                    "channel": channel,
                    "sequence": sample.sequence,
                    "value": serialized_value,
                },
            )
        return sample

    def latest(self, channel: str) -> TelemetrySample | None:
        with self._lock:
            history = self._history.get(channel)
            return history[-1] if history else None

    def history(self, channel: str) -> list[TelemetrySample]:
        with self._lock:
            return list(self._history.get(channel, ()))

    def snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            channels = set(self.DEFAULT_CHANNELS) | set(self._history)
            return {
                channel: (
                    self._history[channel][-1].to_dict(
                        now=now, stale_after_sec=self._stale_after_sec
                    )
                    if self._history.get(channel)
                    else {
                        "channel": channel,
                        "value": {
                            "channel": channel,
                            "state": TelemetryState.DISCONNECTED.value,
                            "detail": "no telemetry received",
                        },
                        "sequence": 0,
                        "timestamp": None,
                        "age_sec": None,
                        "state": TelemetryState.DISCONNECTED.value,
                        "available": False,
                        "stale": True,
                        "disconnected": True,
                    }
                )
                for channel in sorted(channels)
            }
