"""Safety supervisor with fail-closed behavior."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


class StopSink(ABC):
    """Abstract sink for stop actions."""

    @abstractmethod
    def stop(self, reason: str) -> None:
        """Execute stop action."""
        pass


class StopSink(StopSink):
    """Test sink that records when stop was called."""

    def __init__(self) -> None:
        self.stopped: bool = False
        self.last_reason: Optional[str] = None
        self.stop_count: int = 0

    def stop(self, reason: str) -> None:
        """Record stop call."""
        self.stopped = True
        self.last_reason = reason
        self.stop_count += 1

    def reset(self) -> None:
        """Reset for testing."""
        self.stopped = False
        self.last_reason = None


class SafetySupervisor:
    """Safety supervisor with latched emergency stop state."""

    def __init__(
        self,
        stop_sinks: Optional[List[StopSink]] = None,
    ) -> None:
        self._stop_sinks = stop_sinks or []
        self._latched: bool = False
        self._stop_executed: bool = False
        self._last_reason: Optional[str] = None
        self._last_operator: Optional[str] = None

    def is_latched(self) -> bool:
        """Check if safety is latched."""
        return self._latched

    def allows_commands(self) -> bool:
        """Check if commands are allowed (not latched)."""
        return not self._latched

    def emergency_stop(
        self,
        reason: str,
        operator_id: Optional[str] = None,
    ) -> None:
        """Trigger emergency stop (latches, calls all sinks once)."""
        if self._latched:
            return

        self._latched = True
        self._last_reason = reason
        self._last_operator = operator_id

        self._execute_stop(reason)

    def on_lease_expired(self) -> None:
        """Called when lease expires - fail-closed stop."""
        self.emergency_stop("lease expired")

    def on_edge_disconnected(self) -> None:
        """Called when edge disconnects - fail-closed stop."""
        self.emergency_stop("edge disconnected")

    def on_local_stop(self, detail: str = "local stop") -> None:
        """Called when local stop button pressed - fail-closed stop."""
        self.emergency_stop(detail)

    def _execute_stop(self, reason: str) -> None:
        """Execute stop on all sinks (only once)."""
        if self._stop_executed:
            return

        self._stop_executed = True
        for sink in self._stop_sinks:
            try:
                sink.stop(reason)
            except Exception:
                # Don't fail the whole safety on one sink failure
                pass

    def reset(self, authorized: bool = False) -> None:
        """Reset safety latch (requires explicit authorization).

        Re-arms the stop fan-out so a subsequent stop (e.g. a re-pressed
        E-stop after an authorized reset while the contact is still open)
        executes the sinks again. Idempotence within one latch cycle is
        preserved by the ``_latched`` early-return in ``emergency_stop``.
        """
        if not authorized:
            raise PermissionError("Reset requires explicit authorization")

        self._latched = False
        self._stop_executed = False

    def clear_stop_executed(self, authorized: bool = False) -> None:
        """Clear stop executed flag (for testing only)."""
        if not authorized:
            raise PermissionError("Clearing stop flag requires explicit authorization")
        self._stop_executed = False
