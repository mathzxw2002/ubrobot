import math
import unittest

from ubrobot_manipulation.policy import (
    MAX_TARGET_LENGTH,
    PLATFORM_PROFILES,
    WorkspaceBox,
    get_platform_profile,
    grasp_may_start,
    target_pose_is_reachable,
    validate_goal,
)


class GraspGoalValidationTest(unittest.TestCase):
    def test_target_is_trimmed(self):
        goal = validate_goal(target=" 杯子 ", timeout_sec=30.0)
        self.assertEqual(goal.target, "杯子")
        self.assertEqual(goal.timeout_sec, 30.0)

    def test_empty_or_oversized_target_is_rejected(self):
        for target in ("", "   ", "x" * (MAX_TARGET_LENGTH + 1)):
            with self.subTest(target=target), self.assertRaises(ValueError):
                validate_goal(target=target, timeout_sec=30.0)

    def test_timeout_must_be_finite_and_in_range(self):
        for timeout_sec in (0.999, 300.001, math.nan, math.inf, -math.inf):
            with self.subTest(timeout_sec=timeout_sec), self.assertRaises(
                ValueError
            ):
                validate_goal(target="cup", timeout_sec=timeout_sec)


class WorkspaceBoxTest(unittest.TestCase):
    def test_invalid_bounds_are_rejected(self):
        with self.assertRaises(ValueError):
            WorkspaceBox(0.5, 0.1, -0.3, 0.3, 0.0, 0.5)
        with self.assertRaises(ValueError):
            WorkspaceBox(0.1, math.inf, -0.3, 0.3, 0.0, 0.5)

    def test_contains_checks_all_axes(self):
        box = WorkspaceBox(0.1, 0.6, -0.3, 0.3, 0.0, 0.5)
        self.assertTrue(box.contains((0.3, 0.0, 0.2)))
        self.assertFalse(box.contains((0.05, 0.0, 0.2)))
        self.assertFalse(box.contains((0.3, 0.4, 0.2)))
        self.assertFalse(box.contains((0.3, 0.0, -0.1)))

    def test_non_finite_or_malformed_points_are_unreachable(self):
        box = WorkspaceBox(0.1, 0.6, -0.3, 0.3, 0.0, 0.5)
        self.assertFalse(box.contains((math.nan, 0.0, 0.2)))
        self.assertFalse(box.contains(("a", 0.0, 0.2)))
        self.assertFalse(target_pose_is_reachable(None, box))


class PlatformProfileTest(unittest.TestCase):
    def test_required_profiles_exist(self):
        for name in ("piper_station", "go2_piper"):
            profile = get_platform_profile(name)
            self.assertEqual(profile.name, name)
            self.assertTrue(profile.requires_stationary_base)

    def test_lookup_is_case_and_whitespace_insensitive(self):
        self.assertIs(
            get_platform_profile("  Go2_Piper "),
            PLATFORM_PROFILES["go2_piper"],
        )

    def test_unknown_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            get_platform_profile("so101")  # future platform, not yet bound

    def test_mobile_profile_is_more_conservative(self):
        station = get_platform_profile("piper_station")
        go2 = get_platform_profile("go2_piper")
        self.assertLess(
            go2.max_approach_speed_mps, station.max_approach_speed_mps
        )


class MotionAuthorityExclusionTest(unittest.TestCase):
    def setUp(self):
        self.profile = get_platform_profile("go2_piper")

    def test_navigation_lease_blocks_grasp(self):
        self.assertFalse(
            grasp_may_start(
                navigation_lease_active=True,
                base_stationary=True,
                profile=self.profile,
            )
        )

    def test_moving_base_blocks_grasp(self):
        self.assertFalse(
            grasp_may_start(
                navigation_lease_active=False,
                base_stationary=False,
                profile=self.profile,
            )
        )

    def test_clear_authority_allows_grasp(self):
        self.assertTrue(
            grasp_may_start(
                navigation_lease_active=False,
                base_stationary=True,
                profile=self.profile,
            )
        )


if __name__ == "__main__":
    unittest.main()
