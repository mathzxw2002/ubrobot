"""Offline end-to-end tests for the semantic grasp capability."""

from __future__ import annotations

from pathlib import Path
import sys
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
MANIPULATION = ROOT / "ros_depends_ws" / "src" / "ubrobot_manipulation"
if str(MANIPULATION) not in sys.path:
    sys.path.insert(0, str(MANIPULATION))

from ubrobot_manipulation.executors.fixture import (  # noqa: E402
    DeterministicGraspExecutor,
)
from ubrobot_manipulation.lifecycle import (  # noqa: E402
    GraspLifecycleCoordinator,
    GraspStatus,
)
from ubrobot_manipulation.policy import get_platform_profile  # noqa: E402


class StaticAuthority:
    def __init__(self, *, lease=False, stationary=True):
        self.lease = lease
        self.stationary = stationary

    def navigation_lease_active(self):
        return self.lease

    def base_is_stationary(self):
        return self.stationary


class OuterGoal:
    def __init__(self):
        self.cancel_requested = False
        self.feedback = []

    def is_cancel_requested(self):
        return self.cancel_requested

    def publish_feedback(self, feedback):
        self.feedback.append(feedback)


class GraspFixtureE2ETest(unittest.TestCase):
    def make_components(self, *, delay=0.01, authority=None):
        profile = get_platform_profile("piper_station")
        outer = OuterGoal()
        executor = DeterministicGraspExecutor(
            profile=profile,
            phase_delay_sec=delay,
        )
        coordinator = GraspLifecycleCoordinator(
            profile=profile,
            poll_period_sec=0.005,
            cancellation_timeout_sec=1.0,
        )
        return coordinator, outer, executor, authority or StaticAuthority()

    def test_fixture_completes_all_grasp_phases(self):
        coordinator, outer, executor, authority = self.make_components()

        outcome = coordinator.run(
            target="cup",
            timeout_sec=5.0,
            outer=outer,
            executor=executor,
            authority=authority,
        )

        self.assertEqual(outcome.status, GraspStatus.SUCCEEDED)
        self.assertEqual(outcome.message, "grasped 'cup'")
        self.assertEqual(
            {feedback.phase for feedback in outer.feedback},
            {"approach", "align", "grasp", "retreat"},
        )
        self.assertGreaterEqual(outer.feedback[-1].progress, 0.99)

    def test_navigation_lease_rejects_grasp_before_executor_starts(self):
        coordinator, outer, executor, authority = self.make_components(
            authority=StaticAuthority(lease=True)
        )

        outcome = coordinator.run(
            target="cup",
            timeout_sec=5.0,
            outer=outer,
            executor=executor,
            authority=authority,
        )

        self.assertEqual(outcome.status, GraspStatus.REJECTED)
        self.assertFalse(executor.is_done())
        self.assertEqual(outer.feedback, [])

    def test_lease_appearing_during_grasp_cancels_fixture(self):
        authority = StaticAuthority()
        coordinator, outer, executor, _ = self.make_components(
            delay=0.05,
            authority=authority,
        )

        outcome = {}

        def run():
            outcome["value"] = coordinator.run(
                target="cup",
                timeout_sec=5.0,
                outer=outer,
                executor=executor,
                authority=authority,
            )

        worker = threading.Thread(target=run)
        worker.start()
        deadline = time.monotonic() + 2.0
        while len(outer.feedback) < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        authority.lease = True
        worker.join(2.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(outcome["value"].status, GraspStatus.FAILED)
        self.assertIn("navigation lease appeared", outcome["value"].message)

    def test_outer_cancel_is_acknowledged(self):
        coordinator, outer, executor, authority = self.make_components(delay=0.05)
        outcome = {}

        def run():
            outcome["value"] = coordinator.run(
                target="cup",
                timeout_sec=5.0,
                outer=outer,
                executor=executor,
                authority=authority,
            )

        worker = threading.Thread(target=run)
        worker.start()
        time.sleep(0.08)
        outer.cancel_requested = True
        worker.join(2.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(outcome["value"].status, GraspStatus.CANCELLED)


class GraspHarnessContractTest(unittest.TestCase):
    def test_bringup_exposes_fixture_only_behind_explicit_flag(self):
        launch = (
            ROOT
            / "ros_depends_ws/src/emos_bringup/launch/cortex_navigation_bringup.launch.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"start_grasp_server"', launch)
        self.assertIn('default_value="false"', launch)
        self.assertIn('"grasp_executor"', launch)
        self.assertIn("grasp_object_server", launch)

    def test_fixture_has_no_hardware_imports(self):
        source = (
            MANIPULATION
            / "ubrobot_manipulation/executors/fixture.py"
        ).read_text(encoding="utf-8")
        for forbidden in ("rclpy", "torch", "piper_sdk", "serial", "pyrealsense2"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
