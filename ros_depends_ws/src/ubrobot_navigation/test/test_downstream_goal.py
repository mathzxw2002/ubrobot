import unittest

from ubrobot_navigation.downstream_goal import (
    DownstreamFeedback,
    DownstreamResult,
    NavigationLifecycleCoordinator,
    NavigationStatus,
)


class FakeClock:
    def __init__(self):
        self.now_sec = 0.0

    def monotonic(self):
        return self.now_sec

    def sleep(self, seconds):
        self.now_sec += seconds


class FakeLease:
    def __init__(self, events):
        self.events = events
        self.acquire_count = 0
        self.revoke_count = 0

    def acquire(self):
        self.acquire_count += 1
        self.events.append("lease_acquired")
        return "lease-1"

    def revoke(self):
        self.revoke_count += 1
        self.events.append("lease_revoked")


class FakeOuterGoal:
    def __init__(self, *, cancel_requested=False, cortex_active=True):
        self.cancel_requested = cancel_requested
        self._cortex_active = cortex_active
        self.feedback = []

    def is_cancel_requested(self):
        return self.cancel_requested

    def is_cortex_active(self):
        return self._cortex_active

    def publish_feedback(self, feedback):
        self.feedback.append(feedback)


class FakeDownstream:
    def __init__(
        self,
        events,
        *,
        accepted=True,
        done=True,
        success=True,
        cancel_ack=True,
        raise_on_done=False,
        on_start=None,
    ):
        self.events = events
        self.accepted = accepted
        self.done = done
        self.success = success
        self.cancel_ack = cancel_ack
        self.raise_on_done = raise_on_done
        self.on_start = on_start
        self.cancel_called = False
        self.cancel_timeout_sec = None
        self.sent_goal = None

    def start(self, target, timeout_sec, feedback_callback):
        self.sent_goal = (target, timeout_sec)
        self.events.append("downstream_started")
        feedback_callback(DownstreamFeedback(0.4, -0.2))
        if self.on_start:
            self.on_start()
        return self.accepted

    def is_done(self):
        if self.raise_on_done:
            raise RuntimeError("downstream exploded")
        return self.done

    def result(self):
        return DownstreamResult(success=self.success, message="done")

    def cancel(self, timeout_sec):
        self.cancel_called = True
        self.cancel_timeout_sec = timeout_sec
        self.events.append("downstream_cancel_requested")
        if self.cancel_ack:
            self.events.append("downstream_cancel_acknowledged")
        return self.cancel_ack


class NavigationLifecycleCoordinatorTest(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.clock = FakeClock()
        self.lease = FakeLease(self.events)
        self.outer = FakeOuterGoal()
        self.coordinator = NavigationLifecycleCoordinator(
            clock=self.clock.monotonic,
            sleep=self.clock.sleep,
            poll_period_sec=0.05,
            cancellation_timeout_sec=0.4,
        )

    def run_goal(self, downstream, *, target="chair", timeout_sec=30.0, outer=None):
        return self.coordinator.run(
            target=target,
            timeout_sec=timeout_sec,
            outer=outer or self.outer,
            downstream=downstream,
            lease=self.lease,
        )

    def test_invalid_goal_is_rejected_before_lease_acquisition(self):
        outcome = self.run_goal(FakeDownstream(self.events), target="   ")
        self.assertEqual(outcome.status, NavigationStatus.REJECTED)
        self.assertEqual(self.lease.acquire_count, 0)

    def test_only_one_outer_goal_can_run(self):
        nested = []

        def attempt_nested_goal():
            nested.append(self.run_goal(FakeDownstream(self.events)))

        outcome = self.run_goal(
            FakeDownstream(self.events, on_start=attempt_nested_goal)
        )
        self.assertEqual(outcome.status, NavigationStatus.SUCCEEDED)
        self.assertEqual(nested[0].status, NavigationStatus.REJECTED)
        self.assertEqual(self.lease.acquire_count, 1)

    def test_downstream_rejection_revokes_lease_and_returns_failed(self):
        outcome = self.run_goal(FakeDownstream(self.events, accepted=False))
        self.assertEqual(outcome.status, NavigationStatus.FAILED)
        self.assertEqual(self.events[-1], "lease_revoked")

    def test_feedback_is_translated_to_outer_contract(self):
        self.run_goal(FakeDownstream(self.events))
        feedback = self.outer.feedback[0]
        self.assertEqual(feedback.phase, "tracking")
        self.assertEqual(feedback.distance_error, 0.4)
        self.assertEqual(feedback.orientation_error, -0.2)

    def test_outer_cancellation_waits_for_ack_then_revokes(self):
        outer = FakeOuterGoal(cancel_requested=True)
        downstream = FakeDownstream(self.events, done=False)
        outcome = self.run_goal(downstream, outer=outer)
        self.assertEqual(outcome.status, NavigationStatus.CANCELLED)
        self.assertTrue(downstream.cancel_called)
        self.assertEqual(downstream.cancel_timeout_sec, 0.4)
        self.assertEqual(
            self.events[-2:],
            ["downstream_cancel_acknowledged", "lease_revoked"],
        )

    def test_timeout_uses_same_bounded_cancellation_sequence(self):
        downstream = FakeDownstream(self.events, done=False, cancel_ack=False)
        outcome = self.run_goal(downstream, timeout_sec=1.0)
        self.assertEqual(outcome.status, NavigationStatus.TIMED_OUT)
        self.assertTrue(downstream.cancel_called)
        self.assertEqual(downstream.cancel_timeout_sec, 0.4)
        self.assertEqual(self.events[-1], "lease_revoked")

    def test_exception_revokes_lease_in_finally(self):
        downstream = FakeDownstream(self.events, raise_on_done=True)
        outcome = self.run_goal(downstream)
        self.assertEqual(outcome.status, NavigationStatus.FAILED)
        self.assertTrue(downstream.cancel_called)
        self.assertEqual(self.events[-1], "lease_revoked")

    def test_success_revokes_lease_before_returning(self):
        outcome = self.run_goal(FakeDownstream(self.events))
        self.assertEqual(outcome.status, NavigationStatus.SUCCEEDED)
        self.assertEqual(self.events[-1], "lease_revoked")


if __name__ == "__main__":
    unittest.main()
