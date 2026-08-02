"""Tests for wiring the physical E-stop into the Robot Edge runtime (M7).

Workstation-safe: the gpiod-backed reader is never constructed here; tests
inject a fake ``EstopLineReader`` through ``create_app(estop_reader_factory=...)``
and assert fail-closed behavior at the API/latch level. Importing these
modules must never import ``gpiod``.
"""

import os
import sys
import unittest
from datetime import datetime, timezone
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

try:
    from robot_edge.app import create_app
    from robot_edge.hardware.local_stop import EstopLineReader
    from ubrobot_contracts.edge_api import (
        CommandRequest,
        EmergencyStopRequest,
    )
    HAS_ESTOP = True
except ImportError:
    HAS_ESTOP = False

OPERATOR_TOKENS = {
    "operator-token": ["observe", "task.submit", "task.cancel", "lease.manage"],
    "safety-token": ["observe", "safety.stop"],
}


class FakeEstopReader(EstopLineReader):
    """Injectable NC-contact reader; `closed` controls the contact state."""

    def __init__(self, *, closed: bool = True, name: str = "fake:estop#0") -> None:
        self.closed = closed
        self.name = name
        self.reads = 0
        self.closed_count = 0

    def read(self) -> bool:
        self.reads += 1
        return self.closed

    def describe(self) -> str:
        return self.name

    def close(self) -> None:
        self.closed_count += 1


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _cmd_request() -> dict[str, str]:
    return CommandRequest(
        text="导航到前面的椅子",
        correlation_id="trace-1",
        operator_id="test-operator",
        nonce=str(uuid4()),
        timestamp=datetime.now(timezone.utc),
    ).model_dump(mode="json")


def _stop_request() -> dict[str, str]:
    return EmergencyStopRequest(
        operator_id="test-operator",
        correlation_id="trace-stop",
        nonce=str(uuid4()),
        timestamp=datetime.now(timezone.utc),
    ).model_dump(mode="json")


@unittest.skipUnless(HAS_ESTOP, "robot_edge not available")
class TestEstopImportBoundary(unittest.TestCase):
    def test_app_and_local_stop_never_import_gpiod(self) -> None:
        """Workstation import must not touch gpiod (robot-side only)."""
        sys.modules.pop("gpiod", None)
        import robot_edge.app  # noqa: F401
        import robot_edge.hardware.local_stop  # noqa: F401

        self.assertNotIn("gpiod", sys.modules)

    def test_gpiod_import_is_deferred_inside_reader(self) -> None:
        """Constructing the gpiod reader imports gpiod only in __init__."""
        sys.modules.pop("gpiod", None)
        from robot_edge.hardware.local_stop import GpiodEstopLineReader

        self.assertNotIn("gpiod", sys.modules)
        with self.assertRaises(ImportError):
            GpiodEstopLineReader("gpiochip4", 23)  # no libgpiod on workstation
        self.assertNotIn("gpiod", sys.modules)


@unittest.skipUnless(HAS_ESTOP, "robot_edge not available")
class TestEstopRuntimeWiring(unittest.TestCase):
    def _app(self, *, reader_factory=None, **kwargs):
        return create_app(
            test_tokens=OPERATOR_TOKENS,
            estop_reader_factory=reader_factory,
            **kwargs,
        )

    def test_disabled_by_default_reports_unbound(self) -> None:
        """No env -> local stop unavailable, no poller, no reader."""
        with patch.dict(os.environ, {}, clear=True):
            app = self._app()
            with TestClient(app) as client:
                resp = client.get("/v1/health/ready")
                self.assertEqual(resp.status_code, 200)
                data = resp.json()
                self.assertEqual(data["local_stop"]["bound"], False)
                self.assertIsNone(app.state.estop_poller)
                self.assertIsNone(app.state.estop_reader)

    def test_enabled_binds_poller_and_reports_source(self) -> None:
        """Enabled with a closed contact -> bound, poller running, no secrets."""
        env = {
            "UBROBOT_EDGE_ESTOP_ENABLED": "true",
            "UBROBOT_EDGE_ESTOP_CHIP": "gpiochip4",
            "UBROBOT_EDGE_ESTOP_LINE": "23",
            "UBROBOT_EDGE_ESTOP_LINE_NAME": "ubrobot-estop",
        }
        with patch.dict(os.environ, env, clear=True):
            reader = FakeEstopReader(closed=True)
            app = self._app(reader_factory=lambda chip, line, line_name: reader)
            with TestClient(app) as client:
                self.assertIsNotNone(app.state.estop_poller)
                self.assertIs(app.state.estop_reader, reader)
                self.assertTrue(app.state.estop_poller._thread.is_alive())
                resp = client.get("/v1/health/ready")
                data = resp.json()
                self.assertEqual(data["local_stop"]["bound"], True)
                self.assertEqual(data["local_stop"]["source"], "fake:estop#0")
                self.assertEqual(data["local_stop"]["contact_closed"], True)
                body = resp.text
                self.assertNotIn("operator-token", body)
                self.assertNotIn("safety-token", body)

    def test_enabled_missing_line_fails_startup(self) -> None:
        """Enabled but line unset -> fail-closed: service refuses to start."""
        env = {
            "UBROBOT_EDGE_ESTOP_ENABLED": "true",
            "UBROBOT_EDGE_ESTOP_CHIP": "gpiochip4",
        }
        with patch.dict(os.environ, env, clear=True):
            app = self._app(reader_factory=lambda **kw: FakeEstopReader())
            with self.assertRaises(RuntimeError):
                with TestClient(app):
                    pass  # startup must abort

    def test_enabled_non_integer_line_fails_startup(self) -> None:
        env = {
            "UBROBOT_EDGE_ESTOP_ENABLED": "true",
            "UBROBOT_EDGE_ESTOP_CHIP": "gpiochip4",
            "UBROBOT_EDGE_ESTOP_LINE": "not-a-line",
        }
        with patch.dict(os.environ, env, clear=True):
            app = self._app(reader_factory=lambda **kw: FakeEstopReader())
            with self.assertRaises(RuntimeError):
                with TestClient(app):
                    pass

    def test_reader_factory_bad_return_fails_startup(self) -> None:
        env = {
            "UBROBOT_EDGE_ESTOP_ENABLED": "true",
            "UBROBOT_EDGE_ESTOP_CHIP": "gpiochip4",
            "UBROBOT_EDGE_ESTOP_LINE": "23",
        }
        with patch.dict(os.environ, env, clear=True):
            app = self._app(reader_factory=lambda **kw: object())
            with self.assertRaises(RuntimeError):
                with TestClient(app):
                    pass

    def test_open_contact_latches_and_blocks_commands(self) -> None:
        """Pressed E-stop -> latch -> commands rejected -> authorized reset."""
        env = {
            "UBROBOT_EDGE_ESTOP_ENABLED": "true",
            "UBROBOT_EDGE_ESTOP_CHIP": "gpiochip4",
            "UBROBOT_EDGE_ESTOP_LINE": "23",
            "UBROBOT_EDGE_ESTOP_DEBOUNCE_SEC": "0.0",
        }
        with patch.dict(os.environ, env, clear=True):
            reader = FakeEstopReader(closed=False)  # pressed
            app = self._app(reader_factory=lambda chip, line, line_name: reader)
            with TestClient(app) as client:
                # _bind_local_stop seeds one poll; with a zero debounce the
                # next poll completes the window and triggers the latch.
                self.assertTrue(app.state.estop_button.poll_once())
                self.assertTrue(app.state.runtime.safety_latched)
                self.assertTrue(app.state.estop_button.triggered)

                # Readiness reflects the open contact truthfully.
                data = client.get("/v1/health/ready").json()
                self.assertEqual(data["local_stop"]["contact_closed"], False)

                # Latched: commands are rejected with a conflict.
                resp = client.post(
                    "/v1/commands",
                    headers=_headers("operator-token"),
                    json=_cmd_request(),
                )
                self.assertEqual(resp.status_code, 409)

                # Explicit authorized reset clears the latch.
                resp = client.post(
                    "/v1/safety/reset",
                    headers=_headers("safety-token"),
                    json=_stop_request(),
                )
                self.assertEqual(resp.status_code, 200)
                self.assertFalse(app.state.runtime.safety_latched)

                # Commands are accepted again after reset.
                resp = client.post(
                    "/v1/commands",
                    headers=_headers("operator-token"),
                    json=_cmd_request(),
                )
                self.assertEqual(resp.status_code, 200)

    def test_release_can_re_latch_after_reset(self) -> None:
        """A still-open contact re-latches even after an authorized reset."""
        env = {
            "UBROBOT_EDGE_ESTOP_ENABLED": "true",
            "UBROBOT_EDGE_ESTOP_CHIP": "gpiochip4",
            "UBROBOT_EDGE_ESTOP_LINE": "23",
            "UBROBOT_EDGE_ESTOP_DEBOUNCE_SEC": "0.0",
        }
        with patch.dict(os.environ, env, clear=True):
            reader = FakeEstopReader(closed=False)
            app = self._app(reader_factory=lambda chip, line, line_name: reader)
            with TestClient(app) as client:
                app.state.estop_button.poll_once()  # completes zero-debounce
                client.post(
                    "/v1/safety/reset",
                    headers=_headers("safety-token"),
                    json=_stop_request(),
                )
                self.assertFalse(app.state.runtime.safety_latched)
                # Contact still open: the next poll re-latches.
                app.state.estop_button.poll_once()
                app.state.estop_button.poll_once()
                self.assertTrue(app.state.runtime.safety_latched)

    def test_physical_stop_cancels_active_command_and_emits_event(self) -> None:
        """Physical stop must cancel the running command and emit the event."""
        env = {
            "UBROBOT_EDGE_ESTOP_ENABLED": "true",
            "UBROBOT_EDGE_ESTOP_CHIP": "gpiochip4",
            "UBROBOT_EDGE_ESTOP_LINE": "23",
            "UBROBOT_EDGE_ESTOP_DEBOUNCE_SEC": "0.0",
        }
        with patch.dict(os.environ, env, clear=True):
            reader = FakeEstopReader(closed=True)
            app = self._app(
                reader_factory=lambda chip, line, line_name: reader,
                fixture_step_delay_sec=0.05,
            )
            with TestClient(app) as client:
                cmd = client.post(
                    "/v1/commands",
                    headers=_headers("operator-token"),
                    json=_cmd_request(),
                )
                self.assertEqual(cmd.status_code, 200)
                command_id = cmd.json()["command_id"]

                # Command is mid-flight (stepped once at submit).
                events = client.get(
                    "/v1/events", headers=_headers("operator-token")
                ).json()["events"]
                self.assertNotEqual(events[-1]["state"], "succeeded")

                # Press the E-stop: contact opens. The seed poll sampled the
                # closed contact, so two polls complete the zero-debounce.
                reader.closed = False
                self.assertFalse(app.state.estop_button.poll_once())
                self.assertTrue(app.state.estop_button.poll_once())
                self.assertTrue(app.state.runtime.safety_latched)

                # The active command must be cancelled and a critical local
                # safety event emitted.
                events = client.get(
                    "/v1/events", headers=_headers("operator-token")
                ).json()["events"]
                tail = events[-3:]
                self.assertIn(
                    "Emergency stop latched",
                    [e["message"] for e in tail],
                )
                safety_event = next(e for e in tail if e["command_id"] == "safety")
                self.assertEqual(safety_event["payload"]["source"], "local")
                self.assertTrue(safety_event["payload"]["critical"])

    def test_shutdown_stops_poller_and_closes_reader(self) -> None:
        """Lifespan shutdown must join the poller and release the reader."""
        env = {
            "UBROBOT_EDGE_ESTOP_ENABLED": "true",
            "UBROBOT_EDGE_ESTOP_CHIP": "gpiochip4",
            "UBROBOT_EDGE_ESTOP_LINE": "23",
        }
        with patch.dict(os.environ, env, clear=True):
            reader = FakeEstopReader(closed=True)
            app = self._app(reader_factory=lambda chip, line, line_name: reader)
            with TestClient(app):
                poller = app.state.estop_poller
                self.assertTrue(poller._thread.is_alive())
            # Context exited: poller stopped, thread joined, reader released.
            self.assertFalse(poller._thread.is_alive())
            self.assertEqual(reader.closed_count, 1)

    def test_hardware_authority_requires_bound_estop(self) -> None:
        """Authority=true in hardware mode without E-stop fails startup."""
        env = {
            "UBROBOT_EDGE_MODE": "hardware",
            "UBROBOT_EDGE_HARDWARE_AUTHORITY": "true",
            # UBROBOT_EDGE_ESTOP_ENABLED is NOT set -> gate must trip
        }
        with patch.dict(os.environ, env, clear=True):
            app = create_app(execution_mode="hardware", test_tokens=OPERATOR_TOKENS)
            with self.assertRaises(RuntimeError):
                with TestClient(app):
                    pass

    def test_hardware_readonly_does_not_require_estop(self) -> None:
        """Read-only hardware mode (authority=false) may run without E-stop."""
        env = {
            "UBROBOT_EDGE_MODE": "hardware",
            "UBROBOT_EDGE_HARDWARE_AUTHORITY": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            app = create_app(execution_mode="hardware", test_tokens=OPERATOR_TOKENS)
            # No ROS graph on the workstation: the read-only backend aborts
            # startup with the rclpy import error. The E-stop gate must NOT
            # be the reason for this failure (it would be a RuntimeError
            # containing "hardware authority").
            with self.assertRaises(ModuleNotFoundError) as ctx:
                with TestClient(app):
                    pass
            self.assertNotIn("hardware authority", str(ctx.exception))
            self.assertIn("rclpy", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
