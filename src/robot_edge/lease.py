"""Navigation lease state machine."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import uuid4

from ubrobot_contracts.edge_api import LeaseRecord
from ubrobot_contracts.edge_api import LeaseState as ContractLeaseState

# Re-export for compatibility
LeaseState = ContractLeaseState


@dataclass
class Lease:
    """Individual navigation lease."""

    lease_id: str
    owner: str
    issued_at: datetime
    expires_at: datetime
    last_renewed_at: datetime | None = None
    state: LeaseState = LeaseState.ACTIVE

    def __post_init__(self) -> None:
        """Set last_renewed_at to issued_at if not provided."""
        if self.last_renewed_at is None:
            self.last_renewed_at = self.issued_at

    def is_expired(self, current_time: datetime) -> bool:
        """Check if lease is expired, updating state if needed."""
        if self.state != LeaseState.ACTIVE:
            return self.state == LeaseState.EXPIRED

        if current_time >= self.expires_at:
            self.state = LeaseState.EXPIRED
            return True

        return False

    def renew(self, duration_sec: float, current_time: datetime) -> None:
        """Renew the lease, extending expiration."""
        if self.state not in (LeaseState.ACTIVE, LeaseState.EXPIRED):
            raise RuntimeError("Cannot renew lease in current state")

        self.expires_at = current_time + timedelta(seconds=duration_sec)
        self.last_renewed_at = current_time
        self.state = LeaseState.ACTIVE

    def release(self) -> None:
        """Release the lease."""
        if self.state == LeaseState.RELEASED:
            return
        self.state = LeaseState.RELEASED

    def to_record(self) -> LeaseRecord:
        """Convert to contract LeaseRecord."""
        # Ensure last_renewed_at is set
        if self.last_renewed_at is None:
            self.last_renewed_at = self.issued_at

        return LeaseRecord(
            lease_id=self.lease_id,
            owner=self.owner,
            issued_at=self.issued_at,
            expires_at=self.expires_at,
            last_renewed_at=self.last_renewed_at,
            state=self.state,
        )


class LeaseManager:
    """Manager for navigation lease lifecycle."""

    def __init__(self, default_duration_sec: float = 30.0) -> None:
        self._default_duration = default_duration_sec
        self._lease: Optional[Lease] = None

    def get_current_lease(self) -> Optional[Lease]:
        """Get current lease, if any."""
        return self._lease

    def get_state(self, current_time: Optional[datetime] = None) -> LeaseState:
        """Get current lease state."""
        if self._lease is None:
            return LeaseState.NONE

        if current_time is not None:
            self._lease.is_expired(current_time)

        return self._lease.state

    def acquire(
        self,
        operator_id: str,
        duration_sec: Optional[float] = None,
        current_time: Optional[datetime] = None,
    ) -> Lease:
        """Acquire or renew a lease."""
        now = current_time or datetime.now(timezone.utc)
        duration = duration_sec or self._default_duration

        if self._lease is None or self._lease.state != LeaseState.ACTIVE:
            # New lease
            lease_id = str(uuid4())
            self._lease = Lease(
                lease_id=lease_id,
                owner=operator_id,
                issued_at=now,
                expires_at=now + timedelta(seconds=duration),
                last_renewed_at=now,
                state=LeaseState.ACTIVE,
            )
            return self._lease

        # Existing lease
        if self._lease.owner != operator_id:
            raise PermissionError("Lease held by different operator")

        self._lease.renew(duration, now)
        return self._lease

    def release(self, lease_id: str, operator_id: str) -> bool:
        """Release a lease."""
        if self._lease is None:
            return False

        if self._lease.lease_id != lease_id:
            return False

        if self._lease.owner != operator_id:
            return False

        self._lease.release()
        return True

    def check_valid(self, operator_id: str, current_time: datetime) -> bool:
        """Check if lease is valid for given operator."""
        if self._lease is None:
            return False

        if self._lease.owner != operator_id:
            return False

        self._lease.is_expired(current_time)

        return self._lease.state == LeaseState.ACTIVE

    def force_expire(self) -> None:
        """Force lease to expire (for safety)."""
        if self._lease:
            self._lease.state = LeaseState.EXPIRED
