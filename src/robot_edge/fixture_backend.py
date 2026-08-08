"""Fixture backend for Robot Edge (no hardware).

Besides the deterministic command sequence, the fixture simulates motion
telemetry so the full upper-layer loop (command -> simulated /cmd_vel ->
odometry/joint feedback) is visible on a workstation without hardware:

- a navigation-like command advances a simulated pose along +x (0.15 m per
  RUNNING step, ~0.6 m total) and rotates the three wheel joints;
- any other command type leaves the pose untouched;
- motion ends on the final step (vx=0, moving=false);
- after a cancel/emergency stop the generator is dropped without a final
  step; the motion state self-heals to zero after ``motion_timeout_sec``
  (the same "command returns to zero" semantics the real driver enforces).
"""

import re
import time
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, Optional

from ubrobot_contracts.capabilities import (
    CapabilityAvailability,
    CapabilityHealth,
    CapabilityName,
    CapabilitySnapshot,
    ExecutionMode,
)
from ubrobot_contracts.edge_api import CommandState
from ubrobot_contracts.telemetry import (
    TelemetryChannel,
    TelemetrySnapshot,
    TelemetryState,
    TimestampedSample,
)

_NAV_PATTERN = re.compile(
    r"(导航|走到|走向|移动|navigate|go to|move to|follow)", re.IGNORECASE
)

_WHEEL_NAMES = (
    "base_back_wheel_joint",
    "base_left_wheel_joint",
    "base_right_wheel_joint",
)


class FixtureBackend:
    """Fixture backend that provides deterministic behavior without hardware."""

    # Simulated straight-line navigation: 0.15 m per RUNNING step, target
    # ~0.6 m for "导航到前面的椅子"; wheels turn at a matching rate.
    STEP_DISTANCE_M = 0.15
    SIM_VX = 0.1
    WHEEL_STEP_RAD = 0.05

    def __init__(
        self,
        step_delay_sec: float = 0.0,
        *,
        motion_timeout_sec: float = 2.0,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.hardware_authority = False
        self.execution_mode = "fixture"
        self._created_at = datetime.now(timezone.utc)
        # Optional per-step delay to widen the active-command window for
        # process-level cancel/E-stop tests. Defaults to zero so unit tests are
        # not slowed. Kept <= 100 ms per the plan's test-time constraint.
        self._step_delay_sec = max(0.0, float(step_delay_sec))
        self._motion_timeout_sec = max(0.0, float(motion_timeout_sec))
        self._clock = clock or time.monotonic
        # Simulated motion state (default: stationary at origin).
        self._sim_pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        self._sim_wheel_rad = 0.0
        self._sim_moving = False
        self._sim_last_step: float = 0.0

    # ------------------------------------------------------------------ motion

    def _begin_motion(self) -> None:
        self._sim_moving = True
        self._sim_last_step = self._clock()

    def _advance_motion(self) -> None:
        if not self._sim_moving:
            return
        self._sim_pose["x"] = round(self._sim_pose["x"] + self.STEP_DISTANCE_M, 3)
        self._sim_wheel_rad = round(self._sim_wheel_rad + self.WHEEL_STEP_RAD, 4)
        self._sim_last_step = self._clock()

    def _end_motion(self) -> None:
        self._sim_moving = False
        self._sim_last_step = 0.0

    def _self_heal_motion(self) -> None:
        """After a dropped generator (cancel/E-stop) motion returns to zero."""
        if (
            self._sim_moving
            and self._sim_last_step
            and self._clock() - self._sim_last_step > self._motion_timeout_sec
        ):
            self._end_motion()

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
        self._self_heal_motion()

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

        # Odometry telemetry: reflects the simulated motion state while a
        # navigation command runs, so the frontend shows the robot moving.
        channels[TelemetryChannel.ODOMETRY] = TelemetrySnapshot(
            channel=TelemetryChannel.ODOMETRY,
            latest=TimestampedSample(
                timestamp=now,
                state=TelemetryState.AVAILABLE,
                value={
                    **self._sim_pose,
                    "vx": round(self.SIM_VX, 3) if self._sim_moving else 0.0,
                    "moving": self._sim_moving,
                    "source": "fixture",
                },
            ),
            sequence=1,
        )

        # Joint states telemetry: wheel joints turn while moving and keep
        # their final angle when stopped (no teleport back to zero).
        wheel_positions = [round(self._sim_wheel_rad, 4)] * len(_WHEEL_NAMES)
        channels[TelemetryChannel.JOINT_STATES] = TelemetrySnapshot(
            channel=TelemetryChannel.JOINT_STATES,
            latest=TimestampedSample(
                timestamp=now,
                state=TelemetryState.AVAILABLE,
                value={
                    "names": list(_WHEEL_NAMES),
                    "positions": wheel_positions,
                    "velocities": (
                        [round(self.SIM_VX, 3)] * len(_WHEEL_NAMES)
                        if self._sim_moving
                        else [0.0] * len(_WHEEL_NAMES)
                    ),
                    "motor_count": len(_WHEEL_NAMES),
                    "moving": self._sim_moving,
                    "source": "fixture",
                },
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
        """Get a deterministic sequence of command states.

        Navigation-like prompts advance the simulated motion state while
        running so the frontend sees odometry/joint telemetry move; the
        motion ends (vx=0) on the final step.
        """
        is_navigation = bool(_NAV_PATTERN.search(command_text or ""))

        # Yield accepted immediately
        yield CommandState.ACCEPTED, "Command accepted", {}

        # Then planning
        if self._step_delay_sec:
            time.sleep(self._step_delay_sec)
        yield CommandState.PLANNING, "Planning route...", {"progress": 0.25}

        # Then running with some feedback
        if is_navigation:
            self._begin_motion()
        if self._step_delay_sec:
            time.sleep(self._step_delay_sec)
        yield CommandState.RUNNING, "Moving to target...", {"progress": 0.5}
        self._advance_motion()
        if self._step_delay_sec:
            time.sleep(self._step_delay_sec)
        yield CommandState.RUNNING, "Almost there...", {"progress": 0.75}
        self._advance_motion()

        # Then succeeded
        if is_navigation:
            self._end_motion()
        if self._step_delay_sec:
            time.sleep(self._step_delay_sec)
        yield CommandState.SUCCEEDED, "Task complete!", {"progress": 1.0}
