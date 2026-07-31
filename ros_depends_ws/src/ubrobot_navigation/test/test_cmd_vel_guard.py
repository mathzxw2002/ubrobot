import math
import unittest

from ubrobot_navigation.cmd_vel_guard import CmdVelGuardState


ZERO = (0.0, 0.0, 0.0)


class FakeClock:
    def __init__(self):
        self.now_sec = 0.0

    def __call__(self):
        return self.now_sec

    def advance(self, seconds):
        self.now_sec += seconds


class CmdVelGuardStateTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.guard = CmdVelGuardState(clock=self.clock)

    def test_raw_command_without_lease_produces_zero(self):
        self.guard.on_raw_command(0.02, 0.0, 0.0)
        self.assertEqual(self.guard.tick().twist, ZERO)

    def test_fresh_matching_lease_and_command_are_forwarded_after_clamping(self):
        self.guard.on_lease("nav-1")
        self.guard.on_raw_command(0.2, -0.2, 1.0)
        output = self.guard.tick()
        self.assertEqual(output.twist, (0.05, -0.05, 0.20))
        self.assertIsNone(output.error)

    def test_expired_heartbeat_produces_zero_on_next_tick(self):
        self.guard.on_lease("nav-1")
        self.guard.on_raw_command(0.02, 0.0, 0.0)
        self.clock.advance(0.251)
        self.assertEqual(self.guard.tick().twist, ZERO)

    def test_expired_raw_command_produces_zero_with_fresh_heartbeat(self):
        self.guard.on_lease("nav-1")
        self.guard.on_raw_command(0.02, 0.0, 0.0)
        self.clock.advance(0.2)
        self.guard.on_lease("nav-1")
        self.clock.advance(0.051)
        self.assertEqual(self.guard.tick().twist, ZERO)

    def test_lease_identifier_change_invalidates_prior_command(self):
        self.guard.on_lease("nav-1")
        self.guard.on_raw_command(0.02, 0.0, 0.0)
        self.guard.on_lease("nav-2")
        self.assertEqual(self.guard.tick().twist, ZERO)
        self.guard.on_raw_command(0.02, 0.0, 0.0)
        self.assertEqual(self.guard.tick().twist, (0.02, 0.0, 0.0))

    def test_non_finite_velocity_produces_zero_and_error_state(self):
        self.guard.on_lease("nav-1")
        for bad in (math.nan, math.inf, -math.inf):
            with self.subTest(bad=bad):
                self.guard.on_raw_command(bad, 0.0, 0.0)
                output = self.guard.tick()
                self.assertEqual(output.twist, ZERO)
                self.assertEqual(output.error, "non-finite velocity command")

    def test_lease_revocation_emits_at_least_three_zero_samples(self):
        self.guard.on_lease("nav-1")
        self.guard.on_raw_command(0.02, 0.0, 0.0)
        self.guard.on_lease("")
        outputs = [self.guard.tick().twist for _ in range(3)]
        self.assertEqual(outputs, [ZERO, ZERO, ZERO])

    def test_lease_and_raw_timeouts_are_independently_configurable(self):
        raw_short = CmdVelGuardState(
            clock=self.clock,
            lease_timeout_sec=0.5,
            raw_command_timeout_sec=0.1,
        )
        raw_short.on_lease("nav-1")
        raw_short.on_raw_command(0.02, 0.0, 0.0)
        self.clock.advance(0.11)
        raw_short.on_lease("nav-1")
        self.assertEqual(raw_short.tick().twist, ZERO)

        self.clock.now_sec = 0.0
        lease_short = CmdVelGuardState(
            clock=self.clock,
            lease_timeout_sec=0.1,
            raw_command_timeout_sec=0.5,
        )
        lease_short.on_lease("nav-1")
        lease_short.on_raw_command(0.02, 0.0, 0.0)
        self.clock.advance(0.11)
        self.assertEqual(lease_short.tick().twist, ZERO)

    def test_non_positive_or_non_finite_timeouts_are_rejected(self):
        for value in (0.0, -0.1, math.nan, math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                CmdVelGuardState(clock=self.clock, lease_timeout_sec=value)


if __name__ == "__main__":
    unittest.main()
