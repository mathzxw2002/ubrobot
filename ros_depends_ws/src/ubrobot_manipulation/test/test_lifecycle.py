import unittest

from ubrobot_manipulation.lifecycle import (
    ExecutorResult,
    GoalBusyError,
    GraspFeedback,
    GraspLifecycleCoordinator,
    GraspStatus,
)
from ubrobot_manipulation.policy import get_platform_profile


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeOuter:
    def __init__(self, cancel_after_feedbacks=None):
        self.feedback = []
        self.cancel_after_feedbacks = cancel_after_feedbacks

    def is_cancel_requested(self):
        return (
            self.cancel_after_feedbacks is not None
            and len(self.feedback) >= self.cancel_after_feedbacks
        )

    def publish_feedback(self, feedback):
        self.feedback.append(feedback)


class FakeExecutor:
    def __init__(
        self,
        *,
        accept=True,
        feedbacks=(),
        result=ExecutorResult(True, "grasped"),
        cancel_ack=True,
    ):
        self.accept = accept
        self.feedbacks = list(feedbacks)
        self.result_value = result
        self.cancel_ack = cancel_ack
        self.start_calls = []
        self.cancel_calls = 0
        self._done = False

    def start(self, target, timeout_sec, feedback_callback):
        self.start_calls.append((target, timeout_sec))
        if not self.accept:
            return False
        for feedback in self.feedbacks:
            feedback_callback(feedback)
        self._done = True
        return True

    def is_done(self):
        return self._done

    def result(self):
        return self.result_value

    def cancel(self, timeout_sec):
        self.cancel_calls += 1
        return self.cancel_ack


class FakeAuthority:
    def __init__(self, *, lease_active=False, stationary=True):
        self.lease_active = lease_active
        self.stationary = stationary

    def navigation_lease_active(self):
        return self.lease_active

    def base_is_stationary(self):
        return self.stationary


def make_coordinator(clock=None):
    return GraspLifecycleCoordinator(
        profile=get_platform_profile("piper_station"),
        clock=clock or FakeClock(),
        sleep=lambda _seconds: None,
        poll_period_sec=0.01,
        cancellation_timeout_sec=0.5,
    )


class GraspLifecycleTest(unittest.TestCase):
    def test_invalid_goal_is_rejected_before_executor_start(self):
        executor = FakeExecutor()
        outcome = make_coordinator().run(
            target="  ",
            timeout_sec=30.0,
            outer=FakeOuter(),
            executor=executor,
            authority=FakeAuthority(),
        )
        self.assertEqual(outcome.status, GraspStatus.REJECTED)
        self.assertEqual(executor.start_calls, [])

    def test_only_one_goal_runs(self):
        coordinator = make_coordinator()
        first = coordinator.reserve(target="cup", timeout_sec=30.0)
        try:
            with self.assertRaises(GoalBusyError):
                coordinator.reserve(target="bottle", timeout_sec=30.0)
        finally:
            coordinator.abandon(first)

    def test_navigation_lease_rejects_before_start(self):
        executor = FakeExecutor()
        outcome = make_coordinator().run(
            target="cup",
            timeout_sec=30.0,
            outer=FakeOuter(),
            executor=executor,
            authority=FakeAuthority(lease_active=True),
        )
        self.assertEqual(outcome.status, GraspStatus.REJECTED)
        self.assertIn("motion authority conflict", outcome.message)
        self.assertEqual(executor.start_calls, [])

    def test_moving_base_rejects_before_start(self):
        executor = FakeExecutor()
        outcome = make_coordinator().run(
            target="cup",
            timeout_sec=30.0,
            outer=FakeOuter(),
            executor=executor,
            authority=FakeAuthority(stationary=False),
        )
        self.assertEqual(outcome.status, GraspStatus.REJECTED)
        self.assertEqual(executor.start_calls, [])

    def test_executor_rejection_returns_failed(self):
        outcome = make_coordinator().run(
            target="cup",
            timeout_sec=30.0,
            outer=FakeOuter(),
            executor=FakeExecutor(accept=False),
            authority=FakeAuthority(),
        )
        self.assertEqual(outcome.status, GraspStatus.FAILED)

    def test_feedback_is_forwarded_and_success_returned(self):
        outer = FakeOuter()
        feedback = GraspFeedback(
            phase="approach", target_distance_m=0.2, progress=0.4
        )
        outcome = make_coordinator().run(
            target="cup",
            timeout_sec=30.0,
            outer=outer,
            executor=FakeExecutor(feedbacks=[feedback]),
            authority=FakeAuthority(),
        )
        self.assertEqual(outcome.status, GraspStatus.SUCCEEDED)
        self.assertEqual(outer.feedback, [feedback])

    def test_outer_cancel_cancels_executor(self):
        executor = FakeExecutor(feedbacks=[])
        # cancel request observed on the first poll, before executor is done
        executor._done = False
        outer = FakeOuter(cancel_after_feedbacks=0)
        outcome = make_coordinator().run(
            target="cup",
            timeout_sec=30.0,
            outer=outer,
            executor=executor,
            authority=FakeAuthority(),
        )
        self.assertEqual(outcome.status, GraspStatus.CANCELLED)
        self.assertEqual(executor.cancel_calls, 1)

    def test_timeout_cancels_executor(self):
        clock = FakeClock()

        class SlowExecutor(FakeExecutor):
            def start(self, target, timeout_sec, feedback_callback):
                self.start_calls.append((target, timeout_sec))
                clock.advance(31.0)
                return True

            def is_done(self):
                return False

        executor = SlowExecutor()
        outcome = make_coordinator(clock=clock).run(
            target="cup",
            timeout_sec=30.0,
            outer=FakeOuter(),
            executor=executor,
            authority=FakeAuthority(),
        )
        self.assertEqual(outcome.status, GraspStatus.TIMED_OUT)
        self.assertEqual(executor.cancel_calls, 1)

    def test_lease_appearing_mid_grasp_fails_safe(self):
        authority = FakeAuthority()

        class LeaseFlippingExecutor(FakeExecutor):
            def start(self, target, timeout_sec, feedback_callback):
                authority.lease_active = True  # navigation grabs authority
                return True

            def is_done(self):
                return False

        executor = LeaseFlippingExecutor()
        outcome = make_coordinator().run(
            target="cup",
            timeout_sec=30.0,
            outer=FakeOuter(),
            executor=executor,
            authority=authority,
        )
        self.assertEqual(outcome.status, GraspStatus.FAILED)
        self.assertIn("navigation lease appeared", outcome.message)
        self.assertEqual(executor.cancel_calls, 1)

    def test_exception_cancels_executor_and_reports_failed(self):
        class ExplodingExecutor(FakeExecutor):
            def is_done(self):
                raise RuntimeError("executor blew up")

        executor = ExplodingExecutor(feedbacks=[])
        outcome = make_coordinator().run(
            target="cup",
            timeout_sec=30.0,
            outer=FakeOuter(),
            executor=executor,
            authority=FakeAuthority(),
        )
        self.assertEqual(outcome.status, GraspStatus.FAILED)
        self.assertIn("executor blew up", outcome.message)
        self.assertEqual(executor.cancel_calls, 1)

    def test_slot_is_released_after_run(self):
        coordinator = make_coordinator()
        outcome = coordinator.run(
            target="cup",
            timeout_sec=30.0,
            outer=FakeOuter(),
            executor=FakeExecutor(),
            authority=FakeAuthority(),
        )
        self.assertEqual(outcome.status, GraspStatus.SUCCEEDED)
        # a second goal can be reserved immediately
        coordinator.abandon(coordinator.reserve(target="bottle", timeout_sec=30.0))


if __name__ == "__main__":
    unittest.main()
