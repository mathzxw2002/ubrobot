"""Authentication, scopes, and replay protection for Robot Edge."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Set


class Scope(str, Enum):
    """Authorization scopes."""

    OBSERVE = "observe"
    TASK_SUBMIT = "task.submit"
    TASK_CANCEL = "task.cancel"
    SAFETY_STOP = "safety.stop"
    LEASE_MANAGE = "lease.manage"


@dataclass
class AuthConfig:
    """Authentication configuration."""

    tokens: dict[str, list[str]] = field(default_factory=dict)
    request_max_age_sec: int = 300
    nonce_ttl_sec: int = 600


class TokenVerifier:
    """Verify tokens and their associated scopes."""

    def __init__(self, config: AuthConfig) -> None:
        self._config = config

    def verify_token(self, token: str) -> Set[str]:
        """Verify a token and return its scopes.

        Raises:
            PermissionError: If the token is invalid.
        """
        if token not in self._config.tokens:
            raise PermissionError("Invalid authentication")
        return set(self._config.tokens[token])

    def has_scope(self, token: str, scope: str) -> bool:
        """Check if a token has the given scope."""
        try:
            scopes = self.verify_token(token)
            return scope in scopes
        except PermissionError:
            return False


class ReplayProtection:
    """Protect against replay attacks using nonces and timestamps."""

    def __init__(self, config: AuthConfig) -> None:
        self._config = config
        self._seen_nonces: dict[str, datetime] = {}
        # Allow some clock skew (1 minute)
        self._clock_skew_sec = 60

    def check_timestamp(self, timestamp: datetime) -> bool:
        """Check that a timestamp is within acceptable range.

        Args:
            timestamp: Timestamp from the request.

        Returns:
            True if the timestamp is acceptable, False otherwise.
        """
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        min_allowed = now - timedelta(seconds=self._config.request_max_age_sec)
        max_allowed = now + timedelta(seconds=self._clock_skew_sec)

        return min_allowed <= timestamp <= max_allowed

    def check_and_store_nonce(self, nonce: str) -> bool:
        """Check that a nonce hasn't been used before and store it.

        Args:
            nonce: Nonce from the request.

        Returns:
            True if the nonce is new, False if it's been seen before.
        """
        now = datetime.now(timezone.utc)

        # Clean up old nonces first
        self.cleanup_expired()

        if nonce in self._seen_nonces:
            return False

        self._seen_nonces[nonce] = now
        return True

    def cleanup_expired(self) -> None:
        """Remove expired nonces."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self._config.nonce_ttl_sec)

        expired = [
            nonce for nonce, stored_at in self._seen_nonces.items()
            if stored_at < cutoff
        ]
        for nonce in expired:
            del self._seen_nonces[nonce]
