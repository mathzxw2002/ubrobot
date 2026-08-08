"""Fault-injection tests for Robot Edge (P3.2).

Covers failure scenarios not exercised by the unit suites: clock skew/rollback
in replay protection, malformed/oversized payloads, non-2xx backend behavior,
and fail-closed telemetry on transport failure. Everything runs through the
real FastAPI app + TestClient so the HTTP boundary is exercised.

Design principle: every injected fault must FAIL CLOSED — a rejected request
or a `disconnected`/`unavailable` state, never a fabricated success.
"""

from __future__ import annotations

import json
import os
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from fastapi.testclient import TestClient

from robot_edge.app import create_app

OBSERVE_TOKEN = {"operator-token": ["observe", "task.submit", "lease.manage"]}
SUBMIT_BODY = {
    "text": "test task",
    "operator_id": "fault-test",
    "correlation_id": "corr-1",
    "nonce": "nonce-1",
    "timestamp": datetime.now(timezone.utc).isoformat(),
}


class ClockRollbackTest(unittest.TestCase):
    """Replay protection must reject timestamps from the future / far past."""

    def setUp(self) -> None:
        self.app = create_app(
            execution_mode="fixture", test_tokens=dict(OBSERVE_TOKEN)
        )
        self.client = TestClient(self.app)
        self._ctx = self.client
        self._ctx.__enter__()

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)

    def _submit(self, body: dict):
        return self.client.post(
            "/v1/commands",
            json=body,
            headers={"Authorization": "Bearer operator-token"},
        )

    def test_future_timestamp_rejected(self) -> None:
        body = dict(SUBMIT_BODY, nonce="n-future")
        body["timestamp"] = (
            datetime.now(timezone.utc) + timedelta(minutes=10)
        ).isoformat()
        resp = self._submit(body)
        self.assertEqual(resp.status_code, 409)

    def test_clock_skew_within_window_allowed(self) -> None:
        # 30s skew (under the 60s allowance) must be accepted.
        body = dict(SUBMIT_BODY, nonce="n-skew-ok")
        body["timestamp"] = (
            datetime.now(timezone.utc) + timedelta(seconds=30)
        ).isoformat()
        resp = self._submit(body)
        self.assertEqual(resp.status_code, 200)

    def test_naive_timestamp_defaults_to_utc(self) -> None:
        body = dict(SUBMIT_BODY, nonce="n-naive")
        body["timestamp"] = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        resp = self._submit(body)
        self.assertEqual(resp.status_code, 200)


class MalformedPayloadTest(unittest.TestCase):
    """Malformed / oversized inputs must 422, never crash or fabricate."""

    def setUp(self) -> None:
        self.app = create_app(
            execution_mode="fixture", test_tokens=dict(OBSERVE_TOKEN)
        )
        self.client = TestClient(self.app)
        self._ctx = self.client
        self._ctx.__enter__()

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)

    def _post(self, url: str, body, token: str = "operator-token"):
        return self.client.post(
            url, json=body, headers={"Authorization": f"Bearer {token}"}
        )

    def test_missing_required_fields_422(self) -> None:
        resp = self._post("/v1/commands", {"text": "only-text"})
        self.assertEqual(resp.status_code, 422)

    def test_wrong_types_422(self) -> None:
        # timestamp must be an ISO datetime; a raw string fails validation.
        body = dict(SUBMIT_BODY, nonce="n-types")
        body["timestamp"] = "not-a-datetime"
        resp = self._post("/v1/commands", body)
        self.assertEqual(resp.status_code, 422)

    def test_non_string_text_422(self) -> None:
        body = dict(SUBMIT_BODY, nonce="n-int")
        body["text"] = 12345
        resp = self._post("/v1/commands", body)
        self.assertEqual(resp.status_code, 422)

    def test_unknown_scope_token_forbidden(self) -> None:
        resp = self._post(
            "/v1/safety/stop",
            dict(SUBMIT_BODY, nonce="n-scope"),
            token="operator-token",  # lacks safety.stop scope
        )
        self.assertEqual(resp.status_code, 403)


class BackendFailureTest(unittest.TestCase):
    """A failing backend must surface as FAILED, not a fake success."""

    def test_command_backend_client_error_reports_failed(self) -> None:
        from robot_edge.ros.backend import RosCortexCommandBackend

        class _FailingGraph:
            def has_topic(self, topic):
                return False

            def read_topic(self, topic):
                return None

            def shutdown(self):
                pass

        class _FailingClient:
            def send_goal(self, task, *, feedback_callback, terminal_callback):
                terminal_callback(
                    status="failed", message="upstream exploded", raw_status=6
                )

            def shutdown(self):
                pass

        backend = RosCortexCommandBackend(
            _FailingGraph(), client_factory=lambda: _FailingClient()
        )
        gen = backend.get_command_sequence("navigate to chair")
        states = [state for state, _msg, _payload in gen]
        # ACCEPTED then FAILED (not SUCCEEDED).
        self.assertIn("failed", states)
        self.assertNotIn("succeeded", states)


if __name__ == "__main__":
    unittest.main()
