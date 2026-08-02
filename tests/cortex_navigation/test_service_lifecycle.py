from __future__ import annotations

import os
import socket
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.chat_ui import app as ui_app
from src.chat_ui.pipeline import ChatPipeline
from src.chat_ui.service_lifecycle import (
    PortInUseError,
    inspect_port,
    require_port_available,
)
from src.chat_ui.voice_runtime import MockVoiceProvider, VoiceState


class TrackingBackend:
    def __init__(self):
        self.closed = False

    def execute(self, task, *, on_feedback):
        return "done"

    def cancel_active(self):
        return False

    def close(self):
        self.closed = True


class PortLifecycleTest(unittest.TestCase):
    def test_free_port_is_reported_available(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]

        result = inspect_port("127.0.0.1", port)

        self.assertTrue(result.available)
        self.assertIsNone(result.pid)
        require_port_available("127.0.0.1", port)

    def test_occupied_port_reports_process_and_actionable_error(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]

            result = inspect_port("127.0.0.1", port)
            with self.assertRaises(PortInUseError) as raised:
                require_port_available("127.0.0.1", port)

        self.assertFalse(result.available)
        self.assertEqual(result.pid, os.getpid())
        self.assertIn(str(port), str(raised.exception))
        self.assertIn(str(os.getpid()), str(raised.exception))


class ApplicationLifecycleTest(unittest.TestCase):
    def tearDown(self):
        ui_app.chat_pipeline = None

    def test_health_endpoints_are_sanitized_and_report_mock_mode(self):
        with patch.dict(
            os.environ,
            {
                "UBROBOT_CHAT_BACKEND": "cortex-mock",
                "UBROBOT_CHAT_MEDIA": "off",
                "UBROBOT_VOICE_PROVIDER": "mock",
                "DASHSCOPE_API_KEY": "must-not-leak",
            },
            clear=False,
        ):
            ui_app.chat_pipeline = ChatPipeline(initialize_media=False)
            application = ui_app.create_fastapi()
            with TestClient(application) as client:
                live = client.get("/api/health/live")
                ready = client.get("/api/health/ready")

        self.assertEqual(live.status_code, 200)
        self.assertEqual(live.json(), {"status": "live"})
        self.assertEqual(ready.status_code, 200)
        payload = ready.json()
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["backend"], "cortex-mock")
        self.assertEqual(payload["voice_provider"], "mock")
        self.assertEqual(payload["execution_mode"], "mock")
        self.assertFalse(payload["hardware_authority"])
        self.assertNotIn("must-not-leak", ready.text)

    def test_fastapi_shutdown_stops_voice_and_closes_backend(self):
        backend = TrackingBackend()
        provider = MockVoiceProvider()
        pipeline = ChatPipeline(
            backend=backend,
            initialize_media=False,
            voice_provider=provider,
        )
        pipeline.backend_name = "cortex-mock"
        ui_app.chat_pipeline = pipeline

        application = ui_app.create_fastapi()
        with TestClient(application):
            pipeline.voice_runtime.start()
            self.assertEqual(
                pipeline.voice_runtime.snapshot().state,
                VoiceState.LISTENING,
            )

        self.assertEqual(pipeline.voice_runtime.snapshot().state, VoiceState.IDLE)
        self.assertTrue(backend.closed)

    def test_local_shutdown_endpoint_requires_token_and_requests_server_exit(self):
        with patch.dict(
            os.environ,
            {"UBROBOT_SHUTDOWN_TOKEN": "local-secret"},
            clear=False,
        ):
            ui_app.chat_pipeline = ChatPipeline(
                backend=TrackingBackend(),
                initialize_media=False,
                voice_provider=MockVoiceProvider(),
            )
            application = ui_app.create_fastapi()
            control = SimpleNamespace(should_exit=False)
            application.state.uvicorn_server = control
            with TestClient(application) as client:
                denied = client.post(
                    "/api/admin/shutdown",
                    headers={"X-UBRobot-Shutdown-Token": "wrong"},
                )
                accepted = client.post(
                    "/api/admin/shutdown",
                    headers={"X-UBRobot-Shutdown-Token": "local-secret"},
                )

        self.assertEqual(denied.status_code, 403)
        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(accepted.json(), {"status": "stopping"})
        self.assertTrue(control.should_exit)


if __name__ == "__main__":
    unittest.main()
