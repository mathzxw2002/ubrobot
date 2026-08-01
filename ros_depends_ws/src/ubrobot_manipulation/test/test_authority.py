import unittest

from ubrobot_manipulation.authority import AuthorityTracker


class NavigationLeaseTrackingTest(unittest.TestCase):
    def test_no_evidence_means_no_active_lease(self):
        tracker = AuthorityTracker()
        self.assertFalse(tracker.navigation_lease_active(now=100.0))

    def test_fresh_heartbeat_is_active(self):
        tracker = AuthorityTracker(lease_max_age_sec=0.5)
        tracker.note_lease("abc123", now=100.0)
        self.assertTrue(tracker.navigation_lease_active(now=100.4))

    def test_stale_heartbeat_expires(self):
        tracker = AuthorityTracker(lease_max_age_sec=0.5)
        tracker.note_lease("abc123", now=100.0)
        self.assertFalse(tracker.navigation_lease_active(now=100.6))

    def test_empty_lease_revokes_immediately(self):
        tracker = AuthorityTracker(lease_max_age_sec=0.5)
        tracker.note_lease("abc123", now=100.0)
        tracker.note_lease("", now=100.1)
        self.assertFalse(tracker.navigation_lease_active(now=100.1))

    def test_lease_id_change_stays_active(self):
        tracker = AuthorityTracker(lease_max_age_sec=0.5)
        tracker.note_lease("first", now=100.0)
        tracker.note_lease("second", now=100.2)
        self.assertTrue(tracker.navigation_lease_active(now=100.6))


class BaseStationaryTrackingTest(unittest.TestCase):
    def test_no_evidence_fails_closed(self):
        tracker = AuthorityTracker()
        self.assertFalse(tracker.base_is_stationary(now=100.0))

    def test_recent_zero_samples_are_stationary(self):
        tracker = AuthorityTracker(cmd_vel_window_sec=0.5)
        tracker.note_cmd_vel(0.0, 0.0, 0.0, now=100.0)
        tracker.note_cmd_vel(0.0, 0.0, 0.0, now=100.1)
        self.assertTrue(tracker.base_is_stationary(now=100.2))

    def test_nonzero_sample_in_window_is_not_stationary(self):
        tracker = AuthorityTracker(cmd_vel_window_sec=0.5)
        tracker.note_cmd_vel(0.0, 0.0, 0.0, now=100.0)
        tracker.note_cmd_vel(0.05, 0.0, 0.0, now=100.1)
        tracker.note_cmd_vel(0.0, 0.0, 0.0, now=100.2)
        self.assertFalse(tracker.base_is_stationary(now=100.2))

    def test_nonzero_sample_ages_out(self):
        tracker = AuthorityTracker(cmd_vel_window_sec=0.5)
        tracker.note_cmd_vel(0.05, 0.0, 0.0, now=100.0)
        tracker.note_cmd_vel(0.0, 0.0, 0.0, now=100.6)
        self.assertTrue(tracker.base_is_stationary(now=100.7))

    def test_silence_fails_closed(self):
        tracker = AuthorityTracker(cmd_vel_window_sec=0.5)
        tracker.note_cmd_vel(0.0, 0.0, 0.0, now=100.0)
        self.assertFalse(tracker.base_is_stationary(now=100.6))

    def test_non_finite_velocity_is_not_stationary(self):
        tracker = AuthorityTracker(cmd_vel_window_sec=0.5)
        tracker.note_cmd_vel(float("nan"), 0.0, 0.0, now=100.0)
        self.assertFalse(tracker.base_is_stationary(now=100.1))

    def test_windows_must_be_positive(self):
        with self.assertRaises(ValueError):
            AuthorityTracker(lease_max_age_sec=0.0)
        with self.assertRaises(ValueError):
            AuthorityTracker(cmd_vel_window_sec=-1.0)


if __name__ == "__main__":
    unittest.main()
