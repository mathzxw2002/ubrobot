"""Contract tests for the unified motion arbitration state machine (Task 5).

``MotionArbitration`` is the single shared authority source for Go2+Piper:
it owns an ``AuthorityTracker`` (lease + cmd_vel), folds in Go2 body
velocity (via ``/odom``), IMU posture, Piper execution state and the safety
latch, and exposes one state machine
``idle -> navigating -> settling -> manipulating -> idle``.

Safety properties codified here:

- navigation lease active OR base evidence moving -> ``navigating``;
- after navigation ends, a continuous stationary settling window must elapse
  before any grasp may start (``can_start_grasp`` is False during settling);
- any stale evidence, IMU out-of-limit, or safety latch -> ``LATCHED`` and
  both navigation and grasp are forbidden (fail-closed);
- ``can_start_grasp`` requires: no lease, stationary base, settling window
  satisfied, IMU nominal, not latched, Piper not already running.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robot_edge.motion_arbitration import (  # noqa: E402
    MotionArbiterState,
    MotionArbitration,
)


class Clock:
    """Deterministic clock injected into the arbiter."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def __call__(self) -> float:
        return self.now


def make_arbiter(clock: Clock, *, settling_window_sec: float = 2.0, **kw) -> MotionArbitration:
    return MotionArbitration(
        tracker_clock=clock,
        settling_window_sec=settling_window_sec,
        clock=clock,
        **kw,
    )


def feed_stationary(arbiter: MotionArbitration, clock: Clock, steps: int = 3, step: float = 0.1) -> None:
    """Feed steady zero cmd_vel + zero body velocity so the base is still."""
    for _ in range(steps):
        clock.advance(step)
        arbiter.note_cmd_vel(0.0, 0.0, 0.0)
        arbiter.note_body_velocity(0.0, 0.0, 0.0)


class InitialStateTest(unittest.TestCase):
    def test_no_evidence_starts_idle_and_never_grants_grasp(self) -> None:
        clock = Clock()
        arbiter = make_arbiter(clock)
        snap = arbiter.snapshot()
        self.assertEqual(snap.state, MotionArbiterState.IDLE)
        # No cmd_vel evidence -> fail-closed: not stationary, no grasp.
        self.assertFalse(snap.base_stationary)
        self.assertFalse(snap.can_start_grasp)
        self.assertTrue(snap.can_navigate)

    def test_imu_unknown_fails_closed(self) -> None:
        clock = Clock()
        arbiter = make_arbiter(clock)
        feed_stationary(arbiter, clock)
        snap = arbiter.snapshot()
        # IMU never fed -> posture unknown -> grasp forbidden.
        self.assertFalse(snap.can_start_grasp)
        self.assertIn("imu", snap.detail.lower())


class NavigatingStateTest(unittest.TestCase):
    def test_lease_active_puts_arbiter_in_navigating(self) -> None:
        clock = Clock()
        arbiter = make_arbiter(clock)
        clock.advance(0.1)
        arbiter.note_lease("nav-1")
        clock.advance(0.05)
        snap = arbiter.snapshot()
        self.assertEqual(snap.state, MotionArbiterState.NAVIGATING)
        self.assertFalse(snap.can_start_grasp)

    def test_moving_base_puts_arbiter_in_navigating(self) -> None:
        clock = Clock()
        arbiter = make_arbiter(clock)
        clock.advance(0.1)
        arbiter.note_cmd_vel(0.2, 0.0, 0.0)
        snap = arbiter.snapshot()
        self.assertEqual(snap.state, MotionArbiterState.NAVIGATING)
        self.assertFalse(snap.can_start_grasp)

    def test_body_velocity_from_odom_puts_arbiter_in_navigating(self) -> None:
        clock = Clock()
        arbiter = make_arbiter(clock)
        clock.advance(0.1)
        arbiter.note_body_velocity(0.3, 0.0, 0.0)  # /odom twist
        snap = arbiter.snapshot()
        self.assertEqual(snap.state, MotionArbiterState.NAVIGATING)


class SettlingStateTest(unittest.TestCase):
    def test_grasp_forbidden_during_settling_window(self) -> None:
        clock = Clock()
        arbiter = make_arbiter(clock, settling_window_sec=2.0)
        # Navigate, then stop.
        clock.advance(0.1)
        arbiter.note_lease("nav-1")
        clock.advance(0.05)
        arbiter.snapshot()
        # Navigation ends; feed stationary evidence but not yet settling window.
        arbiter.note_lease("")
        feed_stationary(arbiter, clock, steps=5, step=0.1)  # 0.5 s stationary
        snap = arbiter.snapshot()
        self.assertNotEqual(snap.state, MotionArbiterState.NAVIGATING)
        self.assertFalse(snap.can_start_grasp, "grasp must wait for settling window")

    def test_grasp_allowed_after_settling_window(self) -> None:
        clock = Clock()
        arbiter = make_arbiter(clock, settling_window_sec=2.0)
        clock.advance(0.1)
        arbiter.note_lease("nav-1")
        clock.advance(0.05)
        arbiter.snapshot()
        arbiter.note_lease("")
        # Feed stationary for longer than the settling window.
        feed_stationary(arbiter, clock, steps=30, step=0.1)  # 3.0 s stationary
        arbiter.note_imu(0.0, 0.0)
        snap = arbiter.snapshot()
        self.assertTrue(snap.base_stationary)
        self.assertTrue(snap.can_start_grasp)

    def test_motion_during_settling_restarts_window(self) -> None:
        clock = Clock()
        arbiter = make_arbiter(clock, settling_window_sec=2.0)
        clock.advance(0.1)
        arbiter.note_lease("nav-1")
        clock.advance(0.05)
        arbiter.snapshot()
        arbiter.note_lease("")
        feed_stationary(arbiter, clock, steps=5, step=0.1)
        # A burst of motion interrupts settling -> back to navigating.
        clock.advance(0.1)
        arbiter.note_cmd_vel(0.2, 0.0, 0.0)
        snap = arbiter.snapshot()
        self.assertEqual(snap.state, MotionArbiterState.NAVIGATING)
        self.assertFalse(snap.can_start_grasp)


class ManipulatingStateTest(unittest.TestCase):
    def test_grasp_requires_stationary_settled_base(self) -> None:
        clock = Clock()
        arbiter = make_arbiter(clock, settling_window_sec=2.0)
        feed_stationary(arbiter, clock, steps=30, step=0.1)
        arbiter.note_imu(0.0, 0.0)
        snap = arbiter.snapshot()
        self.assertTrue(snap.can_start_grasp)
        arbiter.begin_manipulating()
        snap = arbiter.snapshot()
        self.assertEqual(snap.state, MotionArbiterState.MANIPULATING)
        self.assertFalse(snap.can_navigate)

    def test_navigation_forbidden_while_manipulating(self) -> None:
        clock = Clock()
        arbiter = make_arbiter(clock)
        feed_stationary(arbiter, clock, steps=3, step=0.1)
        arbiter.note_imu(0.0, 0.0)
        arbiter.begin_manipulating()
        snap = arbiter.snapshot()
        self.assertEqual(snap.state, MotionArbiterState.MANIPULATING)
        self.assertFalse(snap.can_navigate)


class LatchStateTest(unittest.TestCase):
    def test_safety_latch_forbids_everything(self) -> None:
        clock = Clock()
        arbiter = make_arbiter(clock)
        feed_stationary(arbiter, clock, steps=3, step=0.1)
        arbiter.note_imu(0.0, 0.0)
        arbiter.note_safety_latch(True)
        snap = arbiter.snapshot()
        self.assertEqual(snap.state, MotionArbiterState.LATCHED)
        self.assertFalse(snap.can_navigate)
        self.assertFalse(snap.can_start_grasp)

    def test_imu_out_of_limit_latches(self) -> None:
        clock = Clock()
        arbiter = make_arbiter(clock)
        feed_stationary(arbiter, clock, steps=3, step=0.1)
        arbiter.note_imu(0.9, 0.0)  # ~51 deg roll, beyond limit
        snap = arbiter.snapshot()
        self.assertEqual(snap.state, MotionArbiterState.LATCHED)
        self.assertFalse(snap.can_start_grasp)

    def test_stale_evidence_latches(self) -> None:
        clock = Clock()
        arbiter = make_arbiter(clock)
        feed_stationary(arbiter, clock, steps=3, step=0.1)
        arbiter.note_imu(0.0, 0.0)
        snap = arbiter.snapshot()
        self.assertFalse(snap.can_start_grasp)  # evidence not yet stale
        # Let all evidence go stale (no fresh samples for a long time).
        clock.advance(10.0)
        snap = arbiter.snapshot()
        self.assertEqual(snap.state, MotionArbiterState.LATCHED)
        self.assertFalse(snap.can_start_grasp)


class FullCycleTest(unittest.TestCase):
    def test_navigate_settle_grasp_cycle(self) -> None:
        clock = Clock()
        arbiter = make_arbiter(clock, settling_window_sec=2.0)
        # Navigate.
        clock.advance(0.1)
        arbiter.note_lease("nav-1")
        clock.advance(0.05)
        self.assertEqual(arbiter.snapshot().state, MotionArbiterState.NAVIGATING)
        # Stop and settle.
        arbiter.note_lease("")
        feed_stationary(arbiter, clock, steps=30, step=0.1)
        arbiter.note_imu(0.0, 0.0)
        self.assertTrue(arbiter.snapshot().can_start_grasp)
        # Manipulate.
        arbiter.begin_manipulating()
        self.assertEqual(arbiter.snapshot().state, MotionArbiterState.MANIPULATING)
        # Finish -> back to idle (after a stationary tick).
        arbiter.end_manipulating()
        feed_stationary(arbiter, clock, steps=3, step=0.1)
        self.assertEqual(arbiter.snapshot().state, MotionArbiterState.IDLE)


if __name__ == "__main__":
    unittest.main()
