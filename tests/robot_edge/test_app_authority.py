"""App-level tests for hardware authority and the Cortex command backend."""

import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

try:
    from robot_edge.app import create_app
    HAS_APP = True
except ImportError:
    HAS_APP = False

TOKENS = {
    "op-token": ["observe", "task.submit", "task.cancel", "lease.manage"],
    "safe-token": ["observe", "safety.stop"],
}


@unittest.skipUnless(HAS_APP, "robot_edge.app not importable")
class TestHardwareAuthorityGate(unittest.TestCase):
    def test_authority_requires_estop_or_exemption(self) -> None:
        env = {
            "UBROBOT_EDGE_MODE": "hardware",
            "UBROBOT_EDGE_HARDWARE_AUTHORITY": "true",
        }
        with patch.dict(os.environ, env, clear=True):
            app = create_app(execution_mode="hardware", test_tokens=TOKENS)
            with self.assertRaises(RuntimeError):
                with TestClient(app):
                    pass

    def test_exemption_allows_authority_with_command_backend(self) -> None:
        env = {
            "UBROBOT_EDGE_MODE": "hardware",
            "UBROBOT_EDGE_HARDWARE_AUTHORITY": "true",
            "UBROBOT_EDGE_ESTOP_EXEMPTED": "true",
        }

        class _FakeBackend:
            execution_mode = "hardware"
            hardware_authority = True

            def __init__(self, **kwargs):
                pass

            def get_capabilities(self):
                return {}

            def get_telemetry_snapshot(self):
                return {}

            def get_command_sequence(self, text):
                yield "accepted", "Command accepted", {}
                yield "succeeded", "Task complete!", {}

            def close(self):
                pass

        with patch.dict(os.environ, env, clear=True), patch(
            "robot_edge.ros.backend.create_cortex_command_backend",
            return_value=_FakeBackend(),
        ):
            app = create_app(execution_mode="hardware", test_tokens=TOKENS)
            with TestClient(app) as client:
                ready = client.get("/v1/health/ready").json()
                self.assertTrue(ready["hardware_authority"])
                # A submitted command flows through the command backend.
                from datetime import datetime, timezone
                from uuid import uuid4

                resp = client.post(
                    "/v1/commands",
                    headers={"Authorization": "Bearer op-token"},
                    json={
                        "text": "请走到椅子旁边",
                        "correlation_id": "c-1",
                        "operator_id": "op",
                        "nonce": str(uuid4()),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
                self.assertEqual(resp.status_code, 200)

    def test_readonly_authority_false_rejects_commands(self) -> None:
        env = {
            "UBROBOT_EDGE_MODE": "hardware",
            "UBROBOT_EDGE_HARDWARE_AUTHORITY": "false",
        }
        with patch.dict(os.environ, env, clear=True), patch(
            "robot_edge.ros.backend.create_readonly_ros_backend",
            return_value=_ReadonlyFake(),
        ):
            app = create_app(execution_mode="hardware", test_tokens=TOKENS)
            with TestClient(app) as client:
                ready = client.get("/v1/health/ready").json()
                self.assertFalse(ready["hardware_authority"])
                from datetime import datetime, timezone
                from uuid import uuid4

                resp = client.post(
                    "/v1/commands",
                    headers={"Authorization": "Bearer op-token"},
                    json={
                        "text": "x",
                        "correlation_id": "c-2",
                        "operator_id": "op",
                        "nonce": str(uuid4()),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
                self.assertEqual(resp.status_code, 409)


class _ReadonlyFake:
    execution_mode = "hardware"
    hardware_authority = False

    def get_capabilities(self):
        return {}

    def get_telemetry_snapshot(self):
        return {}

    def get_command_sequence(self, text):
        raise RuntimeError("hardware authority disabled: read-only mode")

    def close(self):
        pass


if __name__ == "__main__":
    unittest.main()
