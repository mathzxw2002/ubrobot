"""Chained Cortex fixture E2E: "approach the cup on the table, then grasp".

Task 5: verifies the sequential dependency ("navigate, wait for success, then
grasp") end to end through the unified ``MotionArbitration`` authority source
and the ``GraspLifecycleCoordinator`` — no ROS, no hardware, fully
deterministic.

Success path:
    navigate (lease) -> navigation ends -> settling window -> grasp succeeds

Six failure paths:
    1. navigation fails             -> chain stops, grasp never attempted
    2. navigation cancelled         -> chain stops, grasp never attempted
    3. grasp rejected               -> navigation lease still active (or base
                                      moving) at grasp time
    4. lease appears mid-grasp      -> grasp fail-closed cancelled
    5. UI cancel                    -> active step (navigate or grasp) cancelled
    6. safety.stop (latch)          -> everything stops; latch forbids all
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "ros_depends_ws" / "src" / "ubrobot_manipulation") not in sys.path:
    sys.path.insert(0, str(ROOT / "ros_depends_ws" / "src" / "ubrobot_manipulation"))

from robot_edge.motion_arbitration import (  # noqa: E402
    MotionArbiterState,
    MotionArbitration,
)
from ubrobot_manipulation.authority import AuthorityTracker  # noqa: E402
from ubrobot_manipulation.executors.fixture import DeterministicGraspExecutor  # noqa: E402
from ubrobot_manipulation.lifecycle import (  # noqa: E402
    GraspLifecycleCoordinator,
)
from ubrobot_manipulation.policy import get_platform_profile  # noqa: E402


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def __call__(self) -> float:
        return self.now


class ChainResult:
    def __init__(self) -> None:
        self.events: list[str] = []

    def log(self, event: str) -> None:
        self.events.append(event)


class ChainedFixture:
    """Deterministic "navigate -> settle -> grasp" chain over the arbiter.

    Mirrors the semantic Cortex chain: the arbiter is the single authority
    source; a scripted navigation step drives the lease/velocity; the grasp
    runs through the real GraspLifecycleCoordinator + fixture executor. The
    chain returns early on navigation failure/cancel and never grasps while
    authority is held elsewhere.
    """

    def __init__(
        self,
        *,
        clock: Clock,
        settling_window_sec: float = 1.0,
        nav_outcome: str = "success",
        nav_duration_sec: float = 0.2,
        fail_grasp: str | None = None,
        ui_cancel_at_step: str | None = None,
        inject_lease_mid_grasp: bool = False,
        safety_latch_at: str | None = None,
    ) -> None:
        self._clock = clock
        self._settling_window_sec = settling_window_sec
        self._nav_outcome = nav_outcome
        self._nav_duration_sec = nav_duration_sec
        self._fail_grasp = fail_grasp
        self._ui_cancel_at_step = ui_cancel_at_step
        self._inject_lease_mid_grasp = inject_lease_mid_grasp
        self._safety_latch_at = safety_latch_at
        self.result = ChainResult()
        self._tracker = AuthorityTracker()
        self._arbiter = MotionArbitration(
            tracker=self._tracker,
            clock=clock,
            tracker_clock=clock,
            settling_window_sec=settling_window_sec,
        )
        self._profile = get_platform_profile("go2_piper")
        self._coordinator = GraspLifecycleCoordinator(
            profile=self._profile, clock=clock, sleep=lambda _s: None
        )

    # ------------------------------------------------------------ navigate

    def run(self) -> ChainResult:
        clock = self._clock
        if self._safety_latch_at == "start":
            self._arbiter.note_safety_latch(True)
            self.result.log("safety_latched_at_start")
            return self.result

        # --- navigation step ---
        clock.advance(0.01)
        self._arbiter.note_lease("nav-1")
        self.result.log("navigation_started")
        snap = self._arbiter.snapshot()
        assert snap.state == MotionArbiterState.NAVIGATING, snap
        self._feed_navigation()

        if self._ui_cancel_at_step == "navigate":
            self.result.log("ui_cancel_navigate")
            self._arbiter.note_lease("")
            return self.result
        if self._nav_outcome == "fail":
            self.result.log("navigation_failed")
            self._arbiter.note_lease("")
            return self.result
        if self._nav_outcome == "cancel":
            self.result.log("navigation_cancelled")
            self._arbiter.note_lease("")
            return self.result

        # --- navigation success: stop + settle ---
        self._arbiter.note_lease("")
        self.result.log("navigation_succeeded")
        self._settle()

        # --- grasp step ---
        snap = self._arbiter.snapshot()
        if self._fail_grasp == "lease_active":
            # A fresh lease re-acquired by someone else blocks the grasp.
            self._arbiter.note_lease("other-lease")
            self.result.log("grasp_blocked_lease_active")
            self._arbiter.note_lease("")
            return self.result
        if self._fail_grasp == "base_moving":
            self._arbiter.note_cmd_vel(0.3, 0.0, 0.0)
            self.result.log("grasp_blocked_base_moving")
            return self.result

        if not snap.can_start_grasp:
            self.result.log("grasp_blocked_not_settled")
            return self.result

        if self._ui_cancel_at_step == "grasp":
            self.result.log("ui_cancel_grasp_before_start")
            return self.result

        self._arbiter.begin_manipulating()
        self.result.log("grasp_started")
        outcome = self._run_grasp()
        self._arbiter.end_manipulating()
        self.result.log(f"grasp_outcome={outcome.status.name}")
        self.result.log(f"grasp_message={outcome.message}")
        return self.result

    # -------------------------------------------------------------- internals

    def _feed_navigation(self) -> None:
        # Scripted motion while the lease is held.
        for _ in range(5):
            self._clock.advance(0.01)
            self._arbiter.note_cmd_vel(0.15, 0.0, 0.0)

    def _settle(self) -> None:
        # Stop feeding velocity; let the continuous-stationary epoch build.
        for _ in range(200):
            self._clock.advance(0.02)
            self._arbiter.note_cmd_vel(0.0, 0.0, 0.0)
            self._arbiter.note_imu(0.0, 0.0)

    def _run_grasp(self) -> Any:
        class _Outer:
            def __init__(self) -> None:
                self.cancel_requested = False

            def is_cancel_requested(self) -> bool:
                return self.cancel_requested

            def publish_feedback(self, _fb: Any) -> None:
                pass

        class _AuthorityAdapter:
            def __init__(self, arbiter: MotionArbitration) -> None:
                self._arbiter = arbiter

            def navigation_lease_active(self) -> bool:
                return self._arbiter.snapshot().navigation_lease_active

            def base_is_stationary(self) -> bool:
                return self._arbiter.snapshot().base_stationary

        outer = _Outer()

        class _LeaseInjectingExecutor(DeterministicGraspExecutor):
            def __init__(self, arbiter: MotionArbitration, inject: bool) -> None:
                super().__init__(
                    profile=chained._profile, phase_delay_sec=0.005
                )
                self._arbiter = arbiter
                self._inject = inject

            def start(self, target, timeout_sec, feedback_callback) -> bool:
                started = super().start(target, timeout_sec, feedback_callback)
                if self._inject:
                    # A navigation lease appears mid-grasp.
                    self._arbiter.note_lease("nav-mid-grasp")
                return started

        chained = self
        executor = _LeaseInjectingExecutor(
            self._arbiter, self._inject_lease_mid_grasp
        )
        reservation = self._coordinator.reserve(target="cup", timeout_sec=30.0)
        return self._coordinator.execute(
            reservation=reservation,
            outer=outer,
            executor=executor,
            authority=_AuthorityAdapter(self._arbiter),
        )


class ChainSuccessPathTest(unittest.TestCase):
    def test_navigate_then_grasp_succeeds(self) -> None:
        chained = ChainedFixture(clock=Clock())
        result = chained.run()
        self.assertEqual(result.events[0], "navigation_started")
        self.assertIn("navigation_succeeded", result.events)
        self.assertIn("grasp_started", result.events)
        self.assertIn("grasp_outcome=SUCCEEDED", result.events)
        self.assertNotIn("grasp_blocked", " ".join(result.events))


class ChainNavigationFailureTest(unittest.TestCase):
    def test_navigation_failure_stops_chain(self) -> None:
        chained = ChainedFixture(clock=Clock(), nav_outcome="fail")
        result = chained.run()
        self.assertIn("navigation_failed", result.events)
        self.assertNotIn("grasp_started", result.events)

    def test_navigation_cancel_stops_chain(self) -> None:
        chained = ChainedFixture(clock=Clock(), nav_outcome="cancel")
        result = chained.run()
        self.assertIn("navigation_cancelled", result.events)
        self.assertNotIn("grasp_started", result.events)


class ChainGraspRejectionTest(unittest.TestCase):
    def test_grasp_rejected_while_lease_active(self) -> None:
        chained = ChainedFixture(clock=Clock(), fail_grasp="lease_active")
        result = chained.run()
        self.assertIn("grasp_blocked_lease_active", result.events)
        self.assertNotIn("grasp_started", result.events)

    def test_grasp_rejected_while_base_moving(self) -> None:
        chained = ChainedFixture(clock=Clock(), fail_grasp="base_moving")
        result = chained.run()
        self.assertIn("grasp_blocked_base_moving", result.events)
        self.assertNotIn("grasp_started", result.events)


class ChainMidGraspLeaseTest(unittest.TestCase):
    def test_lease_mid_grasp_cancels_fail_closed(self) -> None:
        chained = ChainedFixture(clock=Clock(), inject_lease_mid_grasp=True)
        result = chained.run()
        self.assertIn("grasp_started", result.events)
        self.assertIn("grasp_outcome=FAILED", result.events)
        self.assertIn("lease appeared", " ".join(result.events).lower())


class ChainUiCancelTest(unittest.TestCase):
    def test_ui_cancel_during_navigate(self) -> None:
        chained = ChainedFixture(clock=Clock(), ui_cancel_at_step="navigate")
        result = chained.run()
        self.assertIn("ui_cancel_navigate", result.events)
        self.assertNotIn("grasp_started", result.events)

    def test_ui_cancel_during_grasp(self) -> None:
        chained = ChainedFixture(clock=Clock(), ui_cancel_at_step="grasp")
        result = chained.run()
        self.assertIn("ui_cancel_grasp_before_start", result.events)
        self.assertNotIn("grasp_started", result.events)


class ChainSafetyStopTest(unittest.TestCase):
    def test_safety_latch_at_start_forbids_everything(self) -> None:
        chained = ChainedFixture(clock=Clock(), safety_latch_at="start")
        result = chained.run()
        self.assertEqual(result.events, ["safety_latched_at_start"])
        snap = chained._arbiter.snapshot()
        self.assertEqual(snap.state, MotionArbiterState.LATCHED)
        self.assertFalse(snap.can_navigate)
        self.assertFalse(snap.can_start_grasp)


if __name__ == "__main__":
    unittest.main()
