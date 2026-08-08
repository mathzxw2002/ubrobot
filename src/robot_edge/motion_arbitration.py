"""Unified motion arbitration for Go2+Piper (Task 5, single authority source).

``MotionArbitration`` is the one place that decides whether the Go2 base may
move and whether the Piper arm may grasp. It owns an ``AuthorityTracker``
(lease + cmd_vel evidence) as the authority backbone and folds in:

- Go2 ``body_velocity`` (from ``/odom`` twist) — separate from ``/cmd_vel``
  so physical motion evidence never depends on the commanded value;
- IMU body posture (roll/pitch);
- Piper execution state (manipulating flag);
- the safety latch (E-stop / local stop / lease expiry fan-out).

The state machine is ``idle -> navigating -> settling -> manipulating ->
idle``:

- ``navigating``: navigation lease active OR base evidence moving;
- ``settling``: navigation has ended; a continuous stationary window must
  elapse before any grasp may start;
- ``manipulating``: the Piper arm is executing a grasp (base motion
  forbidden);
- ``LATCHED``: any stale evidence, IMU out-of-limit, or safety latch — both
  navigation and grasp are forbidden (fail-closed).

This module is pure Python (no rclpy / SDK / torch imports) so the state
machine is fully unit-testable on a workstation.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable

from ubrobot_manipulation.authority import AuthorityTracker

# Evidence (odom / imu / lease) older than this is "stale" and latches.
EVIDENCE_MAX_AGE_SEC = 2.0
# Body velocity above this (m/s, rad/s) counts as moving.
BODY_VELOCITY_EPSILON = 0.01
BODY_ANGULAR_EPSILON = 0.02
# Roll/pitch beyond this (radians) means the dog is not level -> latch.
IMU_LIMIT_RAD = 0.35
# Default continuous stationary window after navigation before grasp.
DEFAULT_SETTLING_WINDOW_SEC = 2.0


class MotionArbiterState(str, Enum):
    """Canonical states of the shared motion authority."""

    IDLE = "idle"
    NAVIGATING = "navigating"
    SETTLING = "settling"
    MANIPULATING = "manipulating"
    LATCHED = "latched"


@dataclass(frozen=True)
class ArbitrationSnapshot:
    """JSON-safe view of the arbiter at one instant."""

    state: MotionArbiterState
    base_stationary: bool
    navigation_lease_active: bool
    settling_elapsed_sec: float | None
    settling_required_sec: float
    can_navigate: bool
    can_start_grasp: bool
    safety_latched: bool
    detail: str
    timestamp: datetime


class MotionArbitration:
    """Owns one AuthorityTracker and the shared motion state machine."""

    def __init__(
        self,
        *,
        tracker: AuthorityTracker | None = None,
        settling_window_sec: float = DEFAULT_SETTLING_WINDOW_SEC,
        evidence_max_age_sec: float = EVIDENCE_MAX_AGE_SEC,
        imu_limit_rad: float = IMU_LIMIT_RAD,
        body_velocity_epsilon: float = BODY_VELOCITY_EPSILON,
        body_angular_epsilon: float = BODY_ANGULAR_EPSILON,
        clock: Callable[[], float] = time.monotonic,
        tracker_clock: Callable[[], float] | None = None,
    ) -> None:
        if settling_window_sec <= 0:
            raise ValueError("settling_window_sec must be positive")
        self._settling_window_sec = float(settling_window_sec)
        self._evidence_max_age_sec = float(evidence_max_age_sec)
        self._imu_limit_rad = float(imu_limit_rad)
        self._body_velocity_epsilon = float(body_velocity_epsilon)
        self._body_angular_epsilon = float(body_angular_epsilon)
        self._clock = clock
        self._tracker = tracker or AuthorityTracker()
        self._tracker_clock = tracker_clock or clock
        # IMU evidence.
        self._last_imu_roll = 0.0
        self._last_imu_pitch = 0.0
        self._imu_seen = False
        # Piper + safety state.
        self._piper_active = False
        self._safety_latched = False
        # Continuous-stationary epoch tracking (accumulated across note_*).
        # ``_epoch_start`` is None while the base is moving (or before any
        # evidence); it is set when a zero-velocity sample resumes stillness
        # and reset on any non-zero sample.
        self._epoch_start: float | None = None
        self._last_state = MotionArbiterState.IDLE
        self._last_evidence_t = 0.0

    # ------------------------------------------------------------ producers

    def note_lease(self, lease_id: str) -> None:
        now = self._clock()
        self._tracker.note_lease(lease_id, self._tracker_clock())
        if lease_id:
            self._reset_epoch()
        self._note_evidence(now)

    def note_cmd_vel(self, x: float, y: float, z: float) -> None:
        self._tracker.note_cmd_vel(x, y, z, self._tracker_clock())
        self._note_velocity_sample(x, y, z)

    def note_body_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        """Feed Go2 body velocity from ``/odom`` twist (Task 5)."""
        self._tracker.note_cmd_vel(vx, vy, vyaw, self._tracker_clock())
        self._note_velocity_sample(vx, vy, vyaw)

    def note_imu(self, roll: float, pitch: float) -> None:
        self._last_imu_roll = float(roll)
        self._last_imu_pitch = float(pitch)
        self._imu_seen = True
        self._note_evidence(self._clock())

    def note_piper(self, active: bool) -> None:
        self._piper_active = bool(active)
        self._note_evidence(self._clock())

    def note_safety_latch(self, latched: bool) -> None:
        self._safety_latched = bool(latched)

    def begin_manipulating(self) -> None:
        self._piper_active = True

    def end_manipulating(self) -> None:
        self._piper_active = False

    # ------------------------------------------------------------ consumers

    def snapshot(self) -> ArbitrationSnapshot:
        now = self._clock()
        lease_active = self._tracker.navigation_lease_active(self._tracker_clock())
        base_stationary = self._tracker.base_is_stationary(self._tracker_clock())

        state, detail = self._transition(now, lease_active, base_stationary)
        self._last_state = state

        can_navigate = (
            state != MotionArbiterState.LATCHED
            and state != MotionArbiterState.MANIPULATING
        )
        can_start_grasp = (
            state == MotionArbiterState.IDLE
            and not lease_active
            and base_stationary
            and self._imu_ok()
            and not self._safety_latched
        )
        settling_elapsed = None
        if self._epoch_start is not None:
            settling_elapsed = round(max(0.0, now - self._epoch_start), 3)

        return ArbitrationSnapshot(
            state=state,
            base_stationary=base_stationary,
            navigation_lease_active=lease_active,
            settling_elapsed_sec=settling_elapsed,
            settling_required_sec=self._settling_window_sec,
            can_navigate=can_navigate,
            can_start_grasp=can_start_grasp,
            safety_latched=self._safety_latched,
            detail=detail,
            timestamp=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------- internal

    def _transition(
        self,
        now: float,
        lease_active: bool,
        base_stationary: bool,
    ) -> tuple[MotionArbiterState, str]:
        if self._safety_latched:
            return MotionArbiterState.LATCHED, "safety latch engaged"
        if self._last_evidence_t == 0.0:
            return MotionArbiterState.IDLE, "no evidence yet (idle)"
        if (now - self._last_evidence_t) > self._evidence_max_age_sec:
            return MotionArbiterState.LATCHED, "evidence stale"
        if self._imu_seen and not self._imu_ok():
            return MotionArbiterState.LATCHED, "IMU posture out of limits"

        if self._piper_active:
            return MotionArbiterState.MANIPULATING, "Piper grasp executing"

        # Moving while a stationary epoch is running -> navigation resumed.
        if lease_active:
            return MotionArbiterState.NAVIGATING, "navigation lease active"
        if self._epoch_start is None:
            return MotionArbiterState.NAVIGATING, "base moving"
        if not base_stationary:
            return MotionArbiterState.NAVIGATING, "no recent stationary evidence"

        # Navigation has ended: settle until a continuous stationary window.
        settling_elapsed = (
            now - self._epoch_start if self._epoch_start is not None else 0.0
        )
        if settling_elapsed >= self._settling_window_sec:
            if self._imu_seen:
                return MotionArbiterState.IDLE, "base settled; grasp may start"
            return (
                MotionArbiterState.SETTLING,
                "base settled; imu unavailable; grasp waits",
            )
        if not self._imu_seen:
            return (
                MotionArbiterState.SETTLING,
                "base settling; imu unavailable; grasp waits",
            )
        return MotionArbiterState.SETTLING, "base settling; grasp waits"

    def _imu_ok(self) -> bool:
        return self._imu_seen and (
            abs(self._last_imu_roll) <= self._imu_limit_rad
            and abs(self._last_imu_pitch) <= self._imu_limit_rad
        )

    def _note_velocity_sample(self, x: float, y: float, z: float) -> None:
        try:
            magnitude = max(abs(float(x)), abs(float(y)), abs(float(z)))
        except (TypeError, ValueError):
            magnitude = math.inf
        if not math.isfinite(magnitude):
            magnitude = math.inf
        now = self._clock()
        threshold = max(self._body_velocity_epsilon, self._body_angular_epsilon)
        if magnitude > threshold:
            # Motion evidence: end the stationary epoch immediately.
            self._epoch_start = None
        elif self._epoch_start is None:
            # First zero-velocity sample after motion resumes stillness.
            self._epoch_start = now
        self._note_evidence(now)

    def _reset_epoch(self) -> None:
        self._epoch_start = None

    def _note_evidence(self, now: float) -> None:
        self._last_evidence_t = now
