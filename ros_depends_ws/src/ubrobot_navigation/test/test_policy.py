import math
import unittest

from ubrobot_navigation.policy import (
    COMMAND_FRESHNESS_SEC,
    MAX_ANGULAR_SPEED,
    MAX_LINEAR_SPEED,
    command_is_fresh,
    lease_is_fresh,
    sanitize_twist,
    validate_goal,
)


class NavigationPolicyTest(unittest.TestCase):
    def test_goal_target_is_trimmed(self):
        goal = validate_goal(target=" chair ", timeout_sec=30.0)
        self.assertEqual(goal.target, "chair")
        self.assertEqual(goal.timeout_sec, 30.0)

    def test_empty_or_oversized_target_is_rejected(self):
        for target in ("", "   ", "x" * 129):
            with self.subTest(target=target), self.assertRaises(ValueError):
                validate_goal(target=target, timeout_sec=30.0)

    def test_timeout_must_be_finite_and_in_range(self):
        for timeout_sec in (0.999, 300.001, math.nan, math.inf, -math.inf):
            with self.subTest(timeout_sec=timeout_sec), self.assertRaises(ValueError):
                validate_goal(target="chair", timeout_sec=timeout_sec)

    def test_timeout_range_is_inclusive(self):
        for timeout_sec in (1.0, 300.0):
            with self.subTest(timeout_sec=timeout_sec):
                self.assertEqual(
                    validate_goal("chair", timeout_sec).timeout_sec,
                    timeout_sec,
                )

    def test_velocity_is_clamped_to_first_navigation_limits(self):
        self.assertEqual(
            sanitize_twist(
                linear_x=0.2,
                linear_y=-0.2,
                angular_z=1.0,
                lease_fresh=True,
                command_fresh=True,
            ),
            (MAX_LINEAR_SPEED, -MAX_LINEAR_SPEED, MAX_ANGULAR_SPEED),
        )

    def test_any_non_finite_velocity_produces_zero(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.subTest(bad=bad):
                self.assertEqual(
                    sanitize_twist(
                        linear_x=bad,
                        linear_y=0.01,
                        angular_z=0.01,
                        lease_fresh=True,
                        command_fresh=True,
                    ),
                    (0.0, 0.0, 0.0),
                )

    def test_velocity_requires_fresh_lease_and_command(self):
        command = dict(linear_x=0.01, linear_y=0.0, angular_z=0.0)
        self.assertEqual(
            sanitize_twist(**command, lease_fresh=False, command_fresh=True),
            (0.0, 0.0, 0.0),
        )
        self.assertEqual(
            sanitize_twist(**command, lease_fresh=True, command_fresh=False),
            (0.0, 0.0, 0.0),
        )

    def test_lease_requires_active_state_and_fresh_heartbeat(self):
        self.assertTrue(
            lease_is_fresh(
                active=True,
                heartbeat_age_sec=COMMAND_FRESHNESS_SEC,
            )
        )
        self.assertFalse(lease_is_fresh(active=False, heartbeat_age_sec=0.0))
        self.assertFalse(
            lease_is_fresh(
                active=True,
                heartbeat_age_sec=COMMAND_FRESHNESS_SEC + 0.001,
            )
        )
        self.assertFalse(lease_is_fresh(active=True, heartbeat_age_sec=math.nan))

    def test_stale_raw_command_produces_zero(self):
        age = COMMAND_FRESHNESS_SEC + 0.001
        self.assertFalse(command_is_fresh(age))
        self.assertEqual(
            sanitize_twist(
                linear_x=0.01,
                linear_y=0.0,
                angular_z=0.0,
                lease_fresh=True,
                command_fresh=command_is_fresh(age),
            ),
            (0.0, 0.0, 0.0),
        )


if __name__ == "__main__":
    unittest.main()
