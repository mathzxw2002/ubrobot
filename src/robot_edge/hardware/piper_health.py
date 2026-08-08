"""Piper arm health via injected system probe (M6, read-only).

Piper connects over CAN. M6 never calls enable, go-zero, trajectory,
gripper, or any SDK motion method; it only reports whether the CAN
interface, driver, and torque-disabled state are present and truthful.

The probe is injected so tests can fake it; the real probe is a thin,
read-only system check (CAN interface + driver status), documented for the
robot-side host. No piper_sdk import anywhere in this package.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol

from ubrobot_contracts.capabilities import (
    CapabilityAvailability,
    CapabilityHealth,
    CapabilityName,
    CapabilitySnapshot,
    ExecutionMode,
)
from ubrobot_contracts.telemetry import (
    TelemetryChannel,
    TelemetrySnapshot,
    TelemetryState,
    TimestampedSample,
)


class PiperSystemProbe(Protocol):
    """Read-only view of the Piper system state (no SDK calls)."""

    def can_interface_present(self) -> bool: ...

    def driver_process_running(self) -> bool: ...

    def torque_confirmed_disabled(self) -> bool: ...

    def arm_present(self) -> bool: ...


class PiperHealth:
    """Maps the Piper probe onto capability/telemetry snapshots."""

    def __init__(self, probe: PiperSystemProbe, *, max_age_sec: float = 2.0) -> None:
        self._probe = probe
        self._max_age_sec = max_age_sec

    def capability(self, *, now: datetime | None = None) -> CapabilitySnapshot:
        now = now or datetime.now(timezone.utc)
        can = self._probe.can_interface_present()
        driver = self._probe.driver_process_running()
        torque_off = self._probe.torque_confirmed_disabled()
        arm = self._probe.arm_present()

        if not can and not driver and not arm:
            availability = CapabilityAvailability.DISCONNECTED
            health = CapabilityHealth.UNKNOWN
            detail = "no CAN interface, driver, or arm detected (Piper not connected)"
        elif not torque_off:
            # A torque-enabled arm is a stop condition, never "healthy".
            availability = CapabilityAvailability.UNAVAILABLE
            health = CapabilityHealth.UNHEALTHY
            detail = "Piper torque is NOT confirmed disabled; motion forbidden"
        elif can and driver and arm:
            availability = CapabilityAvailability.AVAILABLE
            health = CapabilityHealth.HEALTHY
            detail = "CAN + driver + arm present, torque confirmed disabled"
        else:
            availability = CapabilityAvailability.UNAVAILABLE
            health = CapabilityHealth.UNKNOWN
            detail = (
                f"partial Piper state: can={can} driver={driver} arm={arm} "
                "torque_off=" + str(torque_off)
            )
        return CapabilitySnapshot(
            name=CapabilityName.GRASP,
            availability=availability,
            health=health,
            execution_mode=ExecutionMode.HARDWARE,
            required_resources=["can", "piper_driver", "safety_control"],
            hardware_authority=False,
            detail=detail,
            last_updated=now,
        )

    def telemetry(
        self, *, now: datetime | None = None
    ) -> dict[TelemetryChannel, TelemetrySnapshot]:
        now = now or datetime.now(timezone.utc)
        caps = self.capability(now=now)
        if caps.availability in (CapabilityAvailability.AVAILABLE,):
            state = TelemetryState.AVAILABLE
            value = {
                "source": "robot-edge:probe",
                "can_interface": self._probe.can_interface_present(),
                "driver_running": self._probe.driver_process_running(),
                "torque_disabled": self._probe.torque_confirmed_disabled(),
            }
        elif caps.availability == CapabilityAvailability.DISCONNECTED:
            state = TelemetryState.DISCONNECTED
            value = {"source": "robot-edge:probe", "detail": caps.detail}
        else:
            state = TelemetryState.UNAVAILABLE
            value = {"source": "robot-edge:probe", "detail": caps.detail}
        return {
            TelemetryChannel.CAPABILITY_HEALTH: TelemetrySnapshot(
                channel=TelemetryChannel.CAPABILITY_HEALTH,
                latest=TimestampedSample(timestamp=now, state=state, value=value),
                sequence=1,
            )
        }
