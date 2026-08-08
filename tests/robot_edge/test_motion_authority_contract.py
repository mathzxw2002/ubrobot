"""Contract tests for the centralized motion authority (refactor Task 1).

``AuthorityTracker`` is pure Python (lease + base-velocity evidence) used by
both the ROS grasp server and the Robot Edge motion arbiter. It must live in
``ubrobot_contracts`` so workstation tests never import a ROS ament package.

These tests also guard the boundary: ``robot_edge.motion_arbitration`` must
import only from ``ubrobot_contracts`` (no ``ubrobot_manipulation``), keeping
the pure-Python layer ROS-free.
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ubrobot_contracts.motion_authority import (  # noqa: E402
    AuthorityTracker,
    CMD_VEL_EPSILON,
    CMD_VEL_WINDOW_SEC,
    LEASE_MAX_AGE_SEC,
)


class ImportBoundaryTest(unittest.TestCase):
    def test_motion_arbitration_imports_only_from_contracts(self) -> None:
        """The arbiter must not import any ROS ament package."""
        source = (ROOT / "src/robot_edge/motion_arbitration.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
            elif isinstance(node, ast.Import):
                imports.extend(a.name for a in node.names)
        offending = [m for m in imports if "ubrobot_manipulation" in m]
        self.assertEqual(offending, [], f"ROS imports leaked into arbiter: {offending}")

    def test_contracts_module_exists_and_is_importable(self) -> None:
        module = (ROOT / "src/ubrobot_contracts/motion_authority.py")
        self.assertTrue(module.exists(), "ubrobot_contracts/motion_authority.py missing")


class AuthorityTrackerContractTest(unittest.TestCase):
    def test_module_exports_expected_constants(self) -> None:
        self.assertEqual(LEASE_MAX_AGE_SEC, 0.5)
        self.assertEqual(CMD_VEL_WINDOW_SEC, 0.5)
        self.assertEqual(CMD_VEL_EPSILON, 1.0e-4)

    def test_lease_heartbeat_tracking(self) -> None:
        tracker = AuthorityTracker(lease_max_age_sec=0.5)
        self.assertFalse(tracker.navigation_lease_active(now=100.0))
        tracker.note_lease("abc123", now=100.0)
        self.assertTrue(tracker.navigation_lease_active(now=100.4))
        self.assertFalse(tracker.navigation_lease_active(now=100.6))

    def test_empty_lease_revokes_immediately(self) -> None:
        tracker = AuthorityTracker(lease_max_age_sec=0.5)
        tracker.note_lease("abc123", now=100.0)
        tracker.note_lease("", now=100.1)
        self.assertFalse(tracker.navigation_lease_active(now=100.1))

    def test_base_stationary_fails_closed_without_evidence(self) -> None:
        tracker = AuthorityTracker(cmd_vel_window_sec=0.5)
        self.assertFalse(tracker.base_is_stationary(now=100.0))

    def test_zero_samples_within_window_are_stationary(self) -> None:
        tracker = AuthorityTracker(cmd_vel_window_sec=0.5)
        tracker.note_cmd_vel(0.0, 0.0, 0.0, now=100.0)
        tracker.note_cmd_vel(0.0, 0.0, 0.0, now=100.1)
        self.assertTrue(tracker.base_is_stationary(now=100.2))

    def test_nonzero_sample_is_not_stationary(self) -> None:
        tracker = AuthorityTracker(cmd_vel_window_sec=0.5)
        tracker.note_cmd_vel(0.0, 0.0, 0.0, now=100.0)
        tracker.note_cmd_vel(0.05, 0.0, 0.0, now=100.1)
        self.assertFalse(tracker.base_is_stationary(now=100.2))

    def test_non_finite_velocity_is_not_stationary(self) -> None:
        tracker = AuthorityTracker(cmd_vel_window_sec=0.5)
        tracker.note_cmd_vel(float("nan"), 0.0, 0.0, now=100.0)
        self.assertFalse(tracker.base_is_stationary(now=100.1))

    def test_invalid_windows_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AuthorityTracker(lease_max_age_sec=0.0)
        with self.assertRaises(ValueError):
            AuthorityTracker(cmd_vel_window_sec=-1.0)


if __name__ == "__main__":
    unittest.main()
