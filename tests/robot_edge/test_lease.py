"""Test navigation lease state machine."""

import unittest
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

try:
    from robot_edge.lease import Lease, LeaseState, LeaseManager
    from ubrobot_contracts.edge_api import LeaseRecord
    HAS_LEASE = True
except ImportError:
    HAS_LEASE = False


class TestLeaseState(unittest.TestCase):
    """Test lease state enum."""

    @unittest.skipUnless(HAS_LEASE, "robot_edge.lease not available")
    def test_lease_states_defined(self) -> None:
        """All required states must be defined."""
        self.assertTrue(hasattr(LeaseState, "NONE"))
        self.assertTrue(hasattr(LeaseState, "ACTIVE"))
        self.assertTrue(hasattr(LeaseState, "EXPIRED"))
        self.assertTrue(hasattr(LeaseState, "RELEASED"))


class TestLease(unittest.TestCase):
    """Test individual lease object."""

    @unittest.skipUnless(HAS_LEASE, "robot_edge.lease not available")
    def test_lease_initial_state(self) -> None:
        """New lease should be active."""
        now = datetime.now(timezone.utc)
        lease = Lease(
            lease_id="lease-1",
            owner="operator-test",
            issued_at=now,
            expires_at=now + timedelta(seconds=30),
        )
        self.assertEqual(lease.state, LeaseState.ACTIVE)
        self.assertEqual(lease.owner, "operator-test")

    @unittest.skipUnless(HAS_LEASE, "robot_edge.lease not available")
    def test_lease_expired_check(self) -> None:
        """Lease should detect expired state with injected clock."""
        now = datetime.now(timezone.utc)
        lease = Lease(
            lease_id="lease-1",
            owner="operator-test",
            issued_at=now,
            expires_at=now + timedelta(seconds=30),
        )

        # Not expired yet
        self.assertFalse(lease.is_expired(current_time=now))

        # Now expired
        later = now + timedelta(seconds=40)
        self.assertTrue(lease.is_expired(current_time=later))
        self.assertEqual(lease.state, LeaseState.EXPIRED)

    @unittest.skipUnless(HAS_LEASE, "robot_edge.lease not available")
    def test_lease_renew(self) -> None:
        """Renew should extend expiration."""
        now = datetime.now(timezone.utc)
        lease = Lease(
            lease_id="lease-1",
            owner="operator-test",
            issued_at=now,
            expires_at=now + timedelta(seconds=30),
        )

        renewal = now + timedelta(seconds=15)
        lease.renew(duration_sec=30, current_time=renewal)
        self.assertEqual(lease.state, LeaseState.ACTIVE)
        self.assertEqual(lease.last_renewed_at, renewal)
        self.assertEqual(lease.expires_at, renewal + timedelta(seconds=30))

    @unittest.skipUnless(HAS_LEASE, "robot_edge.lease not available")
    def test_lease_release(self) -> None:
        """Release should mark lease as released."""
        now = datetime.now(timezone.utc)
        lease = Lease(
            lease_id="lease-1",
            owner="operator-test",
            issued_at=now,
            expires_at=now + timedelta(seconds=30),
        )

        lease.release()
        self.assertEqual(lease.state, LeaseState.RELEASED)

    @unittest.skipUnless(HAS_LEASE, "robot_edge.lease not available")
    def test_lease_to_record(self) -> None:
        """Lease should convert to LeaseRecord."""
        now = datetime.now(timezone.utc)
        lease = Lease(
            lease_id="lease-1",
            owner="operator-test",
            issued_at=now,
            expires_at=now + timedelta(seconds=30),
        )

        record = lease.to_record()
        self.assertIsInstance(record, LeaseRecord)
        self.assertEqual(record.lease_id, "lease-1")
        self.assertEqual(record.owner, "operator-test")
        self.assertEqual(record.state.value, "active")


class TestLeaseManager(unittest.TestCase):
    """Test lease manager (acquire/renew/release)."""

    @unittest.skipUnless(HAS_LEASE, "robot_edge.lease not available")
    def test_manager_starts_with_no_lease(self) -> None:
        """Manager should start with no active lease."""
        manager = LeaseManager(default_duration_sec=30)
        self.assertIsNone(manager.get_current_lease())
        self.assertEqual(manager.get_state(), LeaseState.NONE)

    @unittest.skipUnless(HAS_LEASE, "robot_edge.lease not available")
    def test_acquire_lease_when_none(self) -> None:
        """Should acquire lease when none exists."""
        manager = LeaseManager(default_duration_sec=30)
        now = datetime.now(timezone.utc)

        lease = manager.acquire(
            operator_id="operator-test",
            duration_sec=30,
            current_time=now,
        )

        self.assertIsNotNone(lease)
        self.assertEqual(lease.owner, "operator-test")
        self.assertEqual(manager.get_state(), LeaseState.ACTIVE)

    @unittest.skipUnless(HAS_LEASE, "robot_edge.lease not available")
    def test_acquire_lease_when_active_by_same_owner(self) -> None:
        """Same owner should be able to renew when acquiring again."""
        manager = LeaseManager(default_duration_sec=30)
        now = datetime.now(timezone.utc)

        lease1 = manager.acquire("operator-test", 30, now)
        later = now + timedelta(seconds=15)
        lease2 = manager.acquire("operator-test", 30, later)

        self.assertEqual(lease1.lease_id, lease2.lease_id)
        self.assertEqual(lease2.expires_at, later + timedelta(seconds=30))

    @unittest.skipUnless(HAS_LEASE, "robot_edge.lease not available")
    def test_acquire_lease_when_active_by_different_owner(self) -> None:
        """Different owner should fail to acquire active lease."""
        manager = LeaseManager(default_duration_sec=30)
        now = datetime.now(timezone.utc)

        manager.acquire("operator-1", 30, now)

        with self.assertRaises(PermissionError):
            manager.acquire("operator-2", 30, now)

    @unittest.skipUnless(HAS_LEASE, "robot_edge.lease not available")
    def test_release_lease(self) -> None:
        """Release should work for current owner."""
        manager = LeaseManager(default_duration_sec=30)
        now = datetime.now(timezone.utc)

        lease = manager.acquire("operator-test", 30, now)
        released = manager.release(lease.lease_id, "operator-test")

        self.assertTrue(released)
        self.assertEqual(manager.get_state(), LeaseState.RELEASED)

    @unittest.skipUnless(HAS_LEASE, "robot_edge.lease not available")
    def test_release_wrong_owner_rejected(self) -> None:
        """Wrong owner should not be able to release."""
        manager = LeaseManager(default_duration_sec=30)
        now = datetime.now(timezone.utc)

        lease = manager.acquire("operator-1", 30, now)
        released = manager.release(lease.lease_id, "operator-2")

        self.assertFalse(released)
        self.assertEqual(manager.get_state(), LeaseState.ACTIVE)

    @unittest.skipUnless(HAS_LEASE, "robot_edge.lease not available")
    def test_check_lease_expired(self) -> None:
        """Manager should detect expired leases."""
        manager = LeaseManager(default_duration_sec=30)
        now = datetime.now(timezone.utc)

        lease = manager.acquire("operator-test", 30, now)

        # Not expired
        self.assertTrue(manager.check_valid("operator-test", now))

        # Expired
        later = now + timedelta(seconds=40)
        self.assertFalse(manager.check_valid("operator-test", later))
        self.assertEqual(manager.get_state(), LeaseState.EXPIRED)


class TestLeaseExists(unittest.TestCase):
    """Test that lease module exists."""

    def test_lease_module_exists(self) -> None:
        """robot_edge.lease must be importable."""
        self.assertTrue(
            HAS_LEASE,
            "robot_edge.lease module not found - need to create it",
        )


if __name__ == "__main__":
    unittest.main()
