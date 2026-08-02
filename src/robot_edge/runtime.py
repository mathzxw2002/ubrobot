"""Robot Edge runtime - core state machine."""

import threading
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
from robot_edge.lease import LeaseManager
from robot_edge.safety import SafetySupervisor, StopSink


class RobotEdgeRuntime:
    """Core Robot Edge runtime state machine."""

    def __init__(
        self,
        backend: FixtureBackend,
        max_event_history: int = 1000,
        lease_manager: LeaseManager | None = None,
        safety_supervisor: SafetySupervisor | None = None,
    ) -> None:
        self._backend = backend
        self._events = EventStream(max_history=max_event_history)
        self._lease_manager = lease_manager or LeaseManager()
        self._safety = safety_supervisor or SafetySupervisor()
        self._active_command_id: str | None = None
        self._command_generator: Iterator[tuple[CommandState, str, dict[str, Any]]] | None = None
        # Serializes command state-machine transitions across concurrent
        # submit/cancel/poll requests from FastAPI's threadpool.
        self._command_lock = threading.RLock()

    @property
    def execution_mode(self) -> str:
        return self._backend.execution_mode

    @property
    def hardware_authority(self) -> bool:
        return self._backend.hardware_authority

    @property
    def safety(self) -> SafetySupervisor:
        """The latched safety supervisor (read-only access for bindings).

        Robot-side bindings (physical E-stop, watchdog) call
        ``safety.on_local_stop()`` on this instance so the latch, stop
        fan-out, and event stream behave identically to an API stop.
        """
        return self._safety

    @property
    def safety_latched(self) -> bool:
        return self._safety.is_latched()

    @property
    def lease_state(self) -> LeaseState:
        """Current lease state (none/active/expired/released)."""
        return self._lease_manager.get_state()

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
        if self._safety.is_latched():
            raise RuntimeError("Safety latched - cannot execute commands")

        if not self._safety.allows_commands():
            raise RuntimeError("Safety disallows commands")

        with self._command_lock:
            command_id = str(uuid4())
            self._active_command_id = command_id
            # Start the fixture sequence and emit the first event.
            self._command_generator = self._backend.get_command_sequence(text)
            self._step_command_locked(command_id)
        return command_id

    def _step_command_locked(self, command_id: str) -> None:
        """Advance the active command by one step. Caller holds _command_lock."""
        if self._command_generator is None or self._active_command_id != command_id:
            return
        try:
            state, message, payload = next(self._command_generator)
            self._events.append(
                command_id=command_id,
                state=state,
                message=message,
                payload=payload,
            )
            if state in (CommandState.SUCCEEDED, CommandState.FAILED, CommandState.CANCELLED):
                self._command_generator = None
                self._active_command_id = None
        except StopIteration:
            self._command_generator = None
            self._active_command_id = None

    def _step_command(self, command_id: str) -> None:
        with self._command_lock:
            self._step_command_locked(command_id)

    def poll_events(self) -> None:
        """Poll for new events - in fixture mode, this steps the command."""
        with self._command_lock:
            if self._active_command_id and self._command_generator:
                self._step_command_locked(self._active_command_id)

    def get_events_since(self, event_id: int) -> list[CommandEvent]:
        """Get all events since the given event ID."""
        # First poll to advance any active command, then replay.
        self.poll_events()
        records = self._events.get_since(event_id)
        return [record.event for record in records]

    def cancel_command(
        self,
        command_id: str,
        operator_id: str,
    ) -> bool:
        """Cancel an active command."""
        with self._command_lock:
            if self._active_command_id != command_id:
                return False
            self._events.append(
                command_id=command_id,
                state=CommandState.CANCELLED,
                message="Command cancelled",
                payload={"operator_id": operator_id},
            )
            self._command_generator = None
            self._active_command_id = None
            return True

    def emergency_stop(
        self,
        operator_id: str,
        correlation_id: str,
    ) -> None:
        """Trigger emergency stop (latching)."""
        self._safety.emergency_stop(
            reason=f"emergency stop from {operator_id}",
            operator_id=operator_id,
        )

        # Cancel active command if any (under the command lock so it cannot
        # race with a concurrent poll/submit).
        with self._command_lock:
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

    def local_emergency_stop(self, detail: str) -> None:
        """Physical/local emergency stop (E-stop contact, watchdog, M7).

        Identical semantics to the API emergency stop (latch, cancel the
        active command, emit the critical safety event) except the source
        is a local input, not an operator request.
        """
        self._safety.emergency_stop(reason=detail)

        # Cancel the active command under the command lock so it cannot race
        # with a concurrent poll/submit, exactly like the API stop path.
        with self._command_lock:
            if self._active_command_id:
                self._events.append(
                    command_id=self._active_command_id,
                    state=CommandState.CANCELLED,
                    message="Emergency stop triggered",
                    payload={"source": "local", "critical": True},
                )
                self._command_generator = None
                self._active_command_id = None

        # Always append a safety event (even without an active command).
        self._events.append(
            command_id="safety",
            state=CommandState.CANCELLED,
            message="Emergency stop latched",
            payload={"source": "local", "detail": detail, "critical": True},
        )

    def reset_safety(self, operator_id: str, authorized: bool = True) -> None:
        """Reset safety latch (for testing only)."""
        self._safety.reset(authorized=authorized)

    def acquire_lease(
        self,
        operator_id: str,
        duration_sec: float = 30.0,
    ) -> LeaseRecord:
        """Acquire a navigation lease."""
        lease = self._lease_manager.acquire(
            operator_id=operator_id,
            duration_sec=duration_sec,
        )
        return lease.to_record()

    def release_lease(self, lease_id: str, operator_id: str) -> bool:
        """Release a navigation lease."""
        return self._lease_manager.release(lease_id, operator_id)

    def get_lease(self) -> LeaseRecord | None:
        """Get current lease, if any."""
        lease = self._lease_manager.get_current_lease()
        return lease.to_record() if lease else None
