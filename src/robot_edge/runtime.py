"""Robot Edge runtime - core state machine."""

from datetime import datetime, timezone
from typing import Any, Iterator
from uuid import uuid4

from ubrobot_contracts.edge_api import (
    CommandEvent,
    CommandState,
    LeaseRecord,
    LeaseState,
)
from ubrobot_contracts.capabilities import CapabilityName, CapabilitySnapshot
from ubrobot_contracts.telemetry import TelemetryChannel, TelemetrySnapshot

from robot_edge.event_stream import EventStream, EventRecord
from robot_edge.fixture_backend import FixtureBackend


class RobotEdgeRuntime:
    """Core Robot Edge runtime state machine."""

    def __init__(
        self,
        backend: FixtureBackend,
        max_event_history: int = 1000,
    ) -> None:
        self._backend = backend
        self._events = EventStream(max_history=max_event_history)
        self._lease: LeaseRecord | None = None
        self._active_command_id: str | None = None
        self._safety_latched = False
        self._command_generator: Iterator[tuple[CommandState, str, dict[str, Any]]] | None = None

    @property
    def execution_mode(self) -> str:
        return self._backend.execution_mode

    @property
    def hardware_authority(self) -> bool:
        return self._backend.hardware_authority

    @property
    def lease_state(self) -> str:
        if self._lease is None:
            return "none"
        return self._lease.state.value

    @property
    def safety_latched(self) -> bool:
        return self._safety_latched

    def get_capabilities(self) -> dict[CapabilityName, CapabilitySnapshot]:
        """Get current capability inventory."""
        return self._backend.get_capabilities()

    def get_telemetry_snapshot(self) -> dict[TelemetryChannel, TelemetrySnapshot]:
        """Get current telemetry snapshot."""
        return self._backend.get_telemetry_snapshot()

    def submit_command(
        self,
        text: str,
        operator_id: str,
        correlation_id: str,
    ) -> str:
        """Submit a command for execution."""
        if self._safety_latched:
            raise RuntimeError("Safety latched - cannot execute commands")

        command_id = str(uuid4())
        self._active_command_id = command_id

        # Start the fixture sequence
        self._command_generator = self._backend.get_command_sequence(text)
        self._step_command(command_id)

        return command_id

    def _step_command(self, command_id: str) -> None:
        """Step the command generator if there's an active command."""
        if self._command_generator is None:
            return

        try:
            state, message, payload = next(self._command_generator)
            self._events.append(
                command_id=command_id,
                state=state,
                message=message,
                payload=payload,
            )

            # If not terminal, schedule next step
            if state not in {CommandState.SUCCEEDED, CommandState.FAILED, CommandState.CANCELLED}:
                # In a real implementation, we'd schedule this with asyncio
                # For fixture, we just step immediately when polled
                pass
            else:
                self._command_generator = None
                self._active_command_id = None

        except StopIteration:
            self._command_generator = None
            self._active_command_id = None

    def poll_events(self) -> None:
        """Poll for new events - in fixture mode, this steps the command."""
        if self._active_command_id and self._command_generator:
            self._step_command(self._active_command_id)

    def get_events_since(self, event_id: int) -> list[CommandEvent]:
        """Get all events since the given event ID."""
        # First poll to advance any active command
        self.poll_events()

        records = self._events.get_since(event_id)
        return [record.event for record in records]

    def cancel_command(
        self,
        command_id: str,
        operator_id: str,
    ) -> bool:
        """Cancel an active command."""
        if self._active_command_id == command_id:
            self._events.append(
                command_id=command_id,
                state=CommandState.CANCELLED,
                message="Command cancelled",
                payload={"operator_id": operator_id},
            )
            self._command_generator = None
            self._active_command_id = None
            return True
        return False

    def emergency_stop(
        self,
        operator_id: str,
        correlation_id: str,
    ) -> None:
        """Trigger emergency stop (latching)."""
        self._safety_latched = True

        # Cancel active command if any
        if self._active_command_id:
            self._events.append(
                command_id=self._active_command_id,
                state=CommandState.CANCELLED,
                message="Emergency stop triggered",
                payload={"operator_id": operator_id, "critical": True},
            )
            self._command_generator = None
            self._active_command_id = None

        # Always append a safety event (even without active command)
        self._events.append(
            command_id="safety",
            state=CommandState.CANCELLED,
            message="Emergency stop latched",
            payload={"operator_id": operator_id, "correlation_id": correlation_id, "critical": True},
        )

    def reset_safety(self, operator_id: str) -> None:
        """Reset safety latch (for testing only)."""
        self._safety_latched = False

    def acquire_lease(
        self,
        operator_id: str,
        duration_sec: float = 30.0,
    ) -> LeaseRecord:
        """Acquire a navigation lease."""
        now = datetime.now(timezone.utc)
        lease_id = str(uuid4())
        self._lease = LeaseRecord(
            lease_id=lease_id,
            owner=operator_id,
            issued_at=now,
            expires_at=now,
            last_renewed_at=now,
            state=LeaseState.ACTIVE,
        )
        return self._lease

    def release_lease(self, lease_id: str, operator_id: str) -> bool:
        """Release a navigation lease."""
        if self._lease and self._lease.lease_id == lease_id:
            self._lease = None
            return True
        return False

    def get_lease(self) -> LeaseRecord | None:
        """Get current lease, if any."""
        return self._lease
