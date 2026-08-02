"""Fixture backend for Robot Edge (no hardware)."""

import os
import time
from datetime import datetime, timezone
from typing import Any, Iterator

from ubrobot_contracts.capabilities import (
    CapabilityName,
    CapabilitySnapshot,
    CapabilityAvailability,
    CapabilityHealth,
    ExecutionMode,
)
from ubrobot_contracts.edge_api import CommandState
from ubrobot_contracts.telemetry import (
    TelemetryChannel,
    TelemetryState,
    TelemetrySnapshot,
    TimestampedSample,
)


class FixtureBackend:
    """Fixture backend that provides deterministic behavior without hardware."""

    def __init__(self, step_delay_sec: float = 0.0) -> None:
        self.hardware_authority = False
        self.execution_mode = "fixture"
        self._created_at = datetime.now(timezone.utc)
        # Optional per-step delay to widen the active-command window for
        # process-level cancel/E-stop tests. Defaults to zero so unit tests are
        # not slowed. Kept <= 100 ms per the plan's test-time constraint.
        self._step_delay_sec = max(0.0, float(step_delay_sec))

    def get_capabilities(self) -> dict[CapabilityName, CapabilitySnapshot]:
        """Get capability inventory snapshot."""
        now = datetime.now(timezone.utc)
        capabilities: dict[CapabilityName, CapabilitySnapshot] = {}

        # Navigation capability
        capabilities[CapabilityName.NAVIGATION] = CapabilitySnapshot(
            name=CapabilityName.NAVIGATION,
            availability=CapabilityAvailability.AVAILABLE,
            health=CapabilityHealth.HEALTHY,
            execution_mode=ExecutionMode.FIXTURE,
            required_resources=["camera", "depth", "odometry", "navigation_lease"],
            hardware_authority=False,
            detail="Fixture navigation - no actual movement",
            last_updated=now,
        )

        # Grasp capability
        capabilities[CapabilityName.GRASP] = CapabilitySnapshot(
            name=CapabilityName.GRASP,
            availability=CapabilityAvailability.AVAILABLE,
            health=CapabilityHealth.HEALTHY,
            execution_mode=ExecutionMode.FIXTURE,
            required_resources=["camera", "depth", "joint_states"],
            hardware_authority=False,
            detail="Fixture grasp - no actual movement",
            last_updated=now,
        )

        # Observation capability
        capabilities[CapabilityName.OBSERVATION] = CapabilitySnapshot(
            name=CapabilityName.OBSERVATION,
            availability=CapabilityAvailability.AVAILABLE,
            health=CapabilityHealth.HEALTHY,
            execution_mode=ExecutionMode.FIXTURE,
            required_resources=["camera", "depth"],
            hardware_authority=False,
            detail="Fixture observation - no actual sensors",
            last_updated=now,
        )

        # Follow capability
        capabilities[CapabilityName.FOLLOW] = CapabilitySnapshot(
            name=CapabilityName.FOLLOW,
            availability=CapabilityAvailability.AVAILABLE,
            health=CapabilityHealth.HEALTHY,
            execution_mode=ExecutionMode.FIXTURE,
            required_resources=["camera", "odometry", "navigation_lease"],
            hardware_authority=False,
            detail="Fixture follow - no actual movement",
            last_updated=now,
        )

        # Stop capability
        capabilities[CapabilityName.STOP] = CapabilitySnapshot(
            name=CapabilityName.STOP,
            availability=CapabilityAvailability.AVAILABLE,
            health=CapabilityHealth.HEALTHY,
            execution_mode=ExecutionMode.FIXTURE,
            required_resources=["safety_control"],
            hardware_authority=False,
            detail="Fixture stop - always available",
            last_updated=now,
        )

        return capabilities

    def get_telemetry_snapshot(self) -> dict[TelemetryChannel, TelemetrySnapshot]:
        """Get telemetry snapshot for all channels."""
        now = datetime.now(timezone.utc)
        channels: dict[TelemetryChannel, TelemetrySnapshot] = {}

        # Camera telemetry
        channels[TelemetryChannel.CAMERA] = TelemetrySnapshot(
            channel=TelemetryChannel.CAMERA,
            latest=TimestampedSample(
                timestamp=now,
                state=TelemetryState.AVAILABLE,
                value={"width": 640, "height": 480, "source": "fixture"},
            ),
            sequence=1,
        )

        # Depth telemetry
        channels[TelemetryChannel.DEPTH] = TelemetrySnapshot(
            channel=TelemetryChannel.DEPTH,
            latest=TimestampedSample(
                timestamp=now,
                state=TelemetryState.AVAILABLE,
                value={"width": 640, "height": 480, "unit": "m", "source": "fixture"},
            ),
            sequence=1,
        )

        # Odometry telemetry
        channels[TelemetryChannel.ODOMETRY] = TelemetrySnapshot(
            channel=TelemetryChannel.ODOMETRY,
            latest=TimestampedSample(
                timestamp=now,
                state=TelemetryState.AVAILABLE,
                value={"x": 0.0, "y": 0.0, "yaw": 0.0, "source": "fixture"},
            ),
            sequence=1,
        )

        # Joint states telemetry
        channels[TelemetryChannel.JOINT_STATES] = TelemetrySnapshot(
            channel=TelemetryChannel.JOINT_STATES,
            latest=TimestampedSample(
                timestamp=now,
                state=TelemetryState.AVAILABLE,
                value={"names": [], "positions": [], "source": "fixture"},
            ),
            sequence=1,
        )

        # Navigation lease telemetry
        channels[TelemetryChannel.NAVIGATION_LEASE] = TelemetrySnapshot(
            channel=TelemetryChannel.NAVIGATION_LEASE,
            latest=TimestampedSample(
                timestamp=now,
                state=TelemetryState.AVAILABLE,
                value={"owner": None, "lease_id": None, "source": "fixture"},
            ),
            sequence=1,
        )

        # Capability health telemetry
        channels[TelemetryChannel.CAPABILITY_HEALTH] = TelemetrySnapshot(
            channel=TelemetryChannel.CAPABILITY_HEALTH,
            latest=TimestampedSample(
                timestamp=now,
                state=TelemetryState.AVAILABLE,
                value={"capabilities": {}, "source": "fixture"},
            ),
            sequence=1,
        )

        return channels

    def get_command_sequence(
        self, command_text: str
    ) -> Iterator[tuple[CommandState, str, dict[str, Any]]]:
        """Get a deterministic sequence of command states."""
        # Yield accepted immediately
        yield CommandState.ACCEPTED, "Command accepted", {}

        # Then planning
        if self._step_delay_sec:
            time.sleep(self._step_delay_sec)
        yield CommandState.PLANNING, "Planning route...", {"progress": 0.25}

        # Then running with some feedback
        if self._step_delay_sec:
            time.sleep(self._step_delay_sec)
        yield CommandState.RUNNING, "Moving to target...", {"progress": 0.5}
        if self._step_delay_sec:
            time.sleep(self._step_delay_sec)
        yield CommandState.RUNNING, "Almost there...", {"progress": 0.75}

        # Then succeeded
        if self._step_delay_sec:
            time.sleep(self._step_delay_sec)
        yield CommandState.SUCCEEDED, "Task complete!", {"progress": 1.0}
