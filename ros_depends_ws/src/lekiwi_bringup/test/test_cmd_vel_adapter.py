from pathlib import Path
import math
import sys
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from lekiwi_bringup.velocity_safety import VelocityLimits, sanitize_velocity


class VelocitySafetyTest(unittest.TestCase):
    def setUp(self):
        self.limits = VelocityLimits(0.05, 0.05, 0.20)

    def test_finite_values_within_limits_pass_through(self):
        self.assertEqual(
            sanitize_velocity(0.02, -0.03, 0.10, self.limits),
            (0.02, -0.03, 0.10, True),
        )

    def test_values_are_clipped_symmetrically(self):
        self.assertEqual(
            sanitize_velocity(1.0, -2.0, 3.0, self.limits),
            (0.05, -0.05, 0.20, True),
        )

    def test_non_finite_values_become_invalid_zero(self):
        for value in (math.nan, math.inf, -math.inf):
            self.assertEqual(
                sanitize_velocity(value, 0.0, 0.0, self.limits),
                (0.0, 0.0, 0.0, False),
            )

    def test_invalid_limits_are_rejected(self):
        for value in (0.0, -1.0, math.nan, math.inf):
            with self.assertRaises(ValueError):
                VelocityLimits(value, 0.05, 0.20)


if __name__ == "__main__":
    unittest.main()
