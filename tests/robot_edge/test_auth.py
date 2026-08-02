"""Test Robot Edge authentication, scopes, and replay protection."""

import unittest
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from fastapi.testclient import TestClient

# These imports will fail until auth is created
try:
    from robot_edge.auth import (
        AuthConfig,
        TokenVerifier,
        Scope,
        ReplayProtection,
    )
    from robot_edge.app import create_app
    from robot_edge.runtime import RobotEdgeRuntime
    from robot_edge.fixture_backend import FixtureBackend
    from ubrobot_contracts.edge_api import (
        CommandRequest,
        CancelRequest,
        EmergencyStopRequest,
        LeaseAcquireRequest,
    )
    HAS_AUTH = True
except ImportError:
    HAS_AUTH = False


class TestScope(unittest.TestCase):
    """Test scope definitions."""

    @unittest.skipUnless(HAS_AUTH, "robot_edge.auth not available")
    def test_scope_enum_defined(self) -> None:
        """All required scopes must be defined."""
        self.assertTrue(hasattr(Scope, "OBSERVE"))
        self.assertTrue(hasattr(Scope, "TASK_SUBMIT"))
        self.assertTrue(hasattr(Scope, "TASK_CANCEL"))
        self.assertTrue(hasattr(Scope, "SAFETY_STOP"))
        self.assertTrue(hasattr(Scope, "LEASE_MANAGE"))

    @unittest.skipUnless(HAS_AUTH, "robot_edge.auth not available")
    def test_scope_values(self) -> None:
        """Scope values must match expected strings."""
        self.assertEqual(Scope.OBSERVE.value, "observe")
        self.assertEqual(Scope.TASK_SUBMIT.value, "task.submit")
        self.assertEqual(Scope.TASK_CANCEL.value, "task.cancel")
        self.assertEqual(Scope.SAFETY_STOP.value, "safety.stop")
        self.assertEqual(Scope.LEASE_MANAGE.value, "lease.manage")


class TestAuthConfig(unittest.TestCase):
    """Test auth configuration."""

    @unittest.skipUnless(HAS_AUTH, "robot_edge.auth not available")
    def test_auth_config_defaults(self) -> None:
        """Auth config must have sensible defaults."""
        config = AuthConfig()
        self.assertEqual(config.request_max_age_sec, 300)
        self.assertEqual(config.nonce_ttl_sec, 600)
        self.assertEqual(config.tokens, {})

    @unittest.skipUnless(HAS_AUTH, "robot_edge.auth not available")
    def test_auth_config_with_tokens(self) -> None:
        """Auth config must accept token mapping."""
        tokens = {
            "token1": ["observe"],
            "token2": ["task.submit", "task.cancel"],
        }
        config = AuthConfig(tokens=tokens)
        self.assertEqual(config.tokens, tokens)


class TestTokenVerifier(unittest.TestCase):
    """Test token verification."""

    @unittest.skipUnless(HAS_AUTH, "robot_edge.auth not available")
    def test_verify_valid_token(self) -> None:
        """Valid token with correct scope must pass."""
        tokens = {
            "valid-token": ["observe", "task.submit"],
        }
        config = AuthConfig(tokens=tokens)
        verifier = TokenVerifier(config=config)

        scopes = verifier.verify_token("valid-token")
        self.assertEqual(scopes, {"observe", "task.submit"})

    @unittest.skipUnless(HAS_AUTH, "robot_edge.auth not available")
    def test_verify_invalid_token(self) -> None:
        """Invalid token must raise."""
        tokens = {"valid-token": ["observe"]}
        config = AuthConfig(tokens=tokens)
        verifier = TokenVerifier(config=config)

        with self.assertRaises(PermissionError):
            verifier.verify_token("invalid-token")

    @unittest.skipUnless(HAS_AUTH, "robot_edge.auth not available")
    def test_has_scope(self) -> None:
        """Scope checking must work correctly."""
        tokens = {"token": ["observe", "task.submit"]}
        config = AuthConfig(tokens=tokens)
        verifier = TokenVerifier(config=config)

        self.assertTrue(verifier.has_scope("token", "observe"))
        self.assertTrue(verifier.has_scope("token", "task.submit"))
        self.assertFalse(verifier.has_scope("token", "safety.stop"))
        self.assertFalse(verifier.has_scope("invalid-token", "observe"))


class TestReplayProtection(unittest.TestCase):
    """Test replay protection with nonces and timestamps."""

    @unittest.skipUnless(HAS_AUTH, "robot_edge.auth not available")
    def test_replay_protection_check_timestamp(self) -> None:
        """Timestamp checking must reject old requests."""
        config = AuthConfig(request_max_age_sec=300)
        replay = ReplayProtection(config=config)

        # Fresh timestamp should be ok
        now = datetime.now(timezone.utc)
        self.assertTrue(replay.check_timestamp(now))

        # Too old timestamp should be rejected
        old = now - timedelta(seconds=400)
        self.assertFalse(replay.check_timestamp(old))

        # Future timestamp should be rejected (with some tolerance)
        future = now + timedelta(seconds=100)
        self.assertFalse(replay.check_timestamp(future))

    @unittest.skipUnless(HAS_AUTH, "robot_edge.auth not available")
    def test_replay_protection_check_and_store_nonce(self) -> None:
        """Nonce checking must reject reused nonces."""
        config = AuthConfig(nonce_ttl_sec=600)
        replay = ReplayProtection(config=config)

        # First use should be ok
        self.assertTrue(replay.check_and_store_nonce("nonce-1"))

        # Reused nonce should be rejected
        self.assertFalse(replay.check_and_store_nonce("nonce-1"))

        # New nonce should be ok
        self.assertTrue(replay.check_and_store_nonce("nonce-2"))

    @unittest.skipUnless(HAS_AUTH, "robot_edge.auth not available")
    def test_replay_protection_cleanup(self) -> None:
        """Old nonces should be cleaned up (eventually)."""
        config = AuthConfig(nonce_ttl_sec=1)
        replay = ReplayProtection(config=config)

        self.assertTrue(replay.check_and_store_nonce("nonce-to-expire"))

        # Just check that cleanup doesn't crash
        replay.cleanup_expired()


class TestAuthenticatedAPI(unittest.TestCase):
    """Test authenticated API endpoints."""

    TEST_TOKENS = {
        "observer-token": ["observe"],
        "operator-token": ["observe", "task.submit", "task.cancel", "lease.manage"],
        "safety-token": ["observe", "safety.stop"],
    }

    @unittest.skipUnless(HAS_AUTH, "robot_edge.auth not available")
    def setUp(self) -> None:
        """Set up test client with per-app auth and runtime state.

        TestClient does not run the FastAPI lifespan, so the per-app state the
        lifespan would populate is initialized here. State is scoped to this app
        instance, so no cross-test cleanup is required.
        """
        app = create_app(execution_mode="fixture", test_tokens=self.TEST_TOKENS)
        self.client = TestClient(app)

        from robot_edge.auth import AuthConfig, TokenVerifier, ReplayProtection
        auth_config = AuthConfig(tokens=self.TEST_TOKENS)
        app.state.token_verifier = TokenVerifier(auth_config)
        app.state.replay_protection = ReplayProtection(auth_config)
        app.state.runtime = RobotEdgeRuntime(backend=FixtureBackend())

    def _headers(self, token: str) -> dict[str, str]:
        """Get auth headers."""
        return {"Authorization": f"Bearer {token}"}

    @unittest.skipUnless(HAS_AUTH, "robot_edge.auth not available")
    def test_no_token_rejected(self) -> None:
        """Requests without token must be rejected."""
        response = self.client.get("/v1/capabilities")
        self.assertEqual(response.status_code, 401)

    @unittest.skipUnless(HAS_AUTH, "robot_edge.auth not available")
    def test_invalid_token_rejected(self) -> None:
        """Requests with invalid token must be rejected."""
        response = self.client.get(
            "/v1/capabilities",
            headers=self._headers("invalid-token"),
        )
        self.assertEqual(response.status_code, 401)

    @unittest.skipUnless(HAS_AUTH, "robot_edge.auth not available")
    def test_observe_scope_access_read_endpoints(self) -> None:
        """Observe scope must allow access to read endpoints."""
        response = self.client.get(
            "/v1/capabilities",
            headers=self._headers("observer-token"),
        )
        self.assertEqual(response.status_code, 200)

        response = self.client.get(
            "/v1/telemetry/snapshot",
            headers=self._headers("observer-token"),
        )
        self.assertEqual(response.status_code, 200)

    @unittest.skipUnless(HAS_AUTH, "robot_edge.auth not available")
    def test_insufficient_scope_rejected(self) -> None:
        """Insufficient scope must be rejected with 403."""
        # Observer can't submit commands
        cmd_request = CommandRequest(
            text="test",
            correlation_id="trace-1",
            operator_id="test",
            nonce=str(uuid4()),
            timestamp=datetime.now(timezone.utc),
        )
        response = self.client.post(
            "/v1/commands",
            headers=self._headers("observer-token"),
            json=cmd_request.model_dump(mode="json"),
        )
        self.assertEqual(response.status_code, 403)

    @unittest.skipUnless(HAS_AUTH, "robot_edge.auth not available")
    def test_reused_nonce_rejected(self) -> None:
        """Reused nonce must be rejected with 409."""
        nonce = str(uuid4())
        cmd_request = CommandRequest(
            text="test",
            correlation_id="trace-1",
            operator_id="test",
            nonce=nonce,
            timestamp=datetime.now(timezone.utc),
        )

        # First request
        response = self.client.post(
            "/v1/commands",
            headers=self._headers("operator-token"),
            json=cmd_request.model_dump(mode="json"),
        )
        self.assertIn(response.status_code, [200, 409])  # Might already be latched

        # Second request with same nonce
        response = self.client.post(
            "/v1/commands",
            headers=self._headers("operator-token"),
            json=cmd_request.model_dump(mode="json"),
        )
        self.assertEqual(response.status_code, 409)

    @unittest.skipUnless(HAS_AUTH, "robot_edge.auth not available")
    def test_safety_stop_scope_works(self) -> None:
        """Safety.stop scope must allow emergency stop."""
        stop_request = EmergencyStopRequest(
            correlation_id="trace-1",
            operator_id="test",
            nonce=str(uuid4()),
            timestamp=datetime.now(timezone.utc),
        )

        response = self.client.post(
            "/v1/safety/stop",
            headers=self._headers("safety-token"),
            json=stop_request.model_dump(mode="json"),
        )
        # Should accept or already be latched
        self.assertIn(response.status_code, [200, 409])

    @unittest.skipUnless(HAS_AUTH, "robot_edge.auth not available")
    def test_health_endpoints_no_auth_required(self) -> None:
        """Health endpoints should not require auth."""
        response = self.client.get("/v1/health/live")
        self.assertEqual(response.status_code, 200)

        response = self.client.get("/v1/health/ready")
        self.assertEqual(response.status_code, 200)


class TestErrorMessagesDontLeakSecrets(unittest.TestCase):
    """Test that error messages don't contain secrets."""

    @unittest.skipUnless(HAS_AUTH, "robot_edge.auth not available")
    def test_token_not_in_error_messages(self) -> None:
        """Error messages must not contain the token."""
        tokens = {"valid-token": ["observe"]}
        config = AuthConfig(tokens=tokens)
        verifier = TokenVerifier(config=config)

        try:
            verifier.verify_token("my-secret-token-123")
        except PermissionError as e:
            self.assertNotIn("my-secret-token-123", str(e))


class TestAuthExists(unittest.TestCase):
    """Test that auth module exists."""

    def test_auth_module_exists(self) -> None:
        """robot_edge.auth must be importable."""
        self.assertTrue(
            HAS_AUTH,
            "robot_edge.auth module not found - need to create it",
        )


if __name__ == "__main__":
    unittest.main()
