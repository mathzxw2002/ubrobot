"""Real behavior tests for the Robot Edge telemetry and capability bridge.

Drives the telemetry/capability clients against a live in-process fixture Edge.
Asserts the non-negotiable invariants: no fabricated live data, disconnect is
explicit, staleness is reported, and hardware authority stays false in fixture
mode regardless of local config.
"""

from __future__ import annotations

import socket
import sys
import threading
import time
import unittest
from pathlib import Path

import httpx
import uvicorn

ROOT = Path(__file__).resolve().parents[2]
for segment in (ROOT / "src", ROOT / "src" / "chat_ui"):
    segment_str = str(segment)
    if segment_str not in sys.path:
        sys.path.insert(0, segment_str)

from robot_edge.app import create_app  # noqa: E402
from chat_ui.adapters.robot_edge_telemetry import (  # noqa: E402
    RobotEdgeCapabilityClient,
    RobotEdgeTelemetryClient,
)
from chat_ui.capability_registry import (  # noqa: E402
    CapabilityRegistry,
    create_default_registry,
)
from chat_ui.telemetry import TelemetryHub  # noqa: E402
from ubrobot_contracts.capabilities import ExecutionMode  # noqa: E402

TOKENS = {
    "operator-token": [
        "observe",
        "task.submit",
        "task.cancel",
        "safety.stop",
        "lease.manage",
    ],
}

CHANNELS = (
    "camera",
    "depth",
    "odometry",
    "joint_states",
    "navigation_lease",
    "capability_health",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class _EdgeServer:
    def __init__(self) -> None:
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.app = create_app(execution_mode="fixture", test_tokens=TOKENS)
        config = uvicorn.Config(
            self.app, host="127.0.0.1", port=self.port, log_level="warning"
        )
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self) -> None:
        self.thread.start()
        deadline = time.monotonic() + 15.0
        client = httpx.Client(base_url=self.url, timeout=2.0, trust_env=False)
        while time.monotonic() < deadline:
            try:
                if client.get("/v1/health/live").status_code == 200:
                    client.close()
                    return
            except httpx.RequestError:
                pass
            time.sleep(0.05)
        client.close()
        raise RuntimeError(f"Edge did not start on port {self.port}")

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5.0)


class TestRobotEdgeTelemetry(unittest.TestCase):
    """Telemetry bridge against a live fixture Edge."""

    def setUp(self) -> None:
        self.edge = _EdgeServer()
        self.edge.start()
        self.addCleanup(self.edge.stop)

    def _hub(self, stale_after_sec: float = 3.0) -> TelemetryHub:
        return TelemetryHub(stale_after_sec=stale_after_sec)

    def test_initial_snapshot_hydration(self) -> None:
        """A successful poll hydrates every channel from the Edge."""
        hub = self._hub()
        client = RobotEdgeTelemetryClient(
            edge_url=self.edge.url, token="operator-token", telemetry_hub=hub
        )
        try:
            self.assertTrue(client.poll_once())
        finally:
            client.close()
        snapshot = hub.snapshot()
        for channel in CHANNELS:
            self.assertIn(channel, snapshot)
            self.assertEqual(snapshot[channel]["state"], "available")
            self.assertTrue(snapshot[channel]["available"])

    def test_edge_disconnect_marks_channels_disconnected(self) -> None:
        """When the Edge stops, every channel must become disconnected."""
        hub = self._hub()
        client = RobotEdgeTelemetryClient(
            edge_url=self.edge.url, token="operator-token", telemetry_hub=hub
        )
        try:
            self.assertTrue(client.poll_once())
            # Stop the Edge; the next poll must fail and mark disconnected.
            self.edge.stop()
            self.assertFalse(client.poll_once())
            client.mark_disconnected()
        finally:
            client.close()
        snapshot = hub.snapshot()
        for channel in CHANNELS:
            self.assertEqual(snapshot[channel]["state"], "disconnected")
            self.assertFalse(snapshot[channel]["available"])
            self.assertTrue(snapshot[channel]["disconnected"])

    def test_stale_after_deadline(self) -> None:
        """Available samples older than the deadline are reported as stale."""
        hub = self._hub(stale_after_sec=0.1)
        client = RobotEdgeTelemetryClient(
            edge_url=self.edge.url, token="operator-token", telemetry_hub=hub
        )
        try:
            self.assertTrue(client.poll_once())
            self.assertEqual(
                hub.snapshot()["camera"]["state"], "available"
            )
            time.sleep(0.25)
            self.assertEqual(
                hub.snapshot()["camera"]["state"], "stale"
            )
            self.assertFalse(hub.snapshot()["camera"]["available"])
        finally:
            client.close()

    def test_reconnect_after_dropped_connection(self) -> None:
        """The client recovers after a transient Edge outage."""
        hub = self._hub()
        client = RobotEdgeTelemetryClient(
            edge_url=self.edge.url, token="operator-token", telemetry_hub=hub
        )
        try:
            self.assertTrue(client.poll_once())
            self.assertEqual(hub.snapshot()["odometry"]["state"], "available")
            # Simulate a transient outage: stop and restart the Edge on the same URL.
            self.edge.stop()
            self.assertFalse(client.poll_once())
            client.mark_disconnected()
            self.assertEqual(
                hub.snapshot()["odometry"]["state"], "disconnected"
            )
            # Restart a fresh Edge on the same port.
            restart = _EdgeServer()
            restart.port = self.edge.port
            restart.url = self.edge.url
            restart.app = create_app(execution_mode="fixture", test_tokens=TOKENS)
            config = uvicorn.Config(
                restart.app,
                host="127.0.0.1",
                port=restart.port,
                log_level="warning",
            )
            restart.server = uvicorn.Server(config)
            restart.thread = threading.Thread(
                target=restart.server.run, daemon=True
            )
            restart.start()
            self.addCleanup(restart.stop)
            self.assertTrue(client.poll_once())
            self.assertEqual(hub.snapshot()["odometry"]["state"], "available")
        finally:
            client.close()

    def test_sdk_like_object_rejected_before_reaching_hub(self) -> None:
        """Non-serializable SDK objects must be rejected at publish time."""
        hub = self._hub()
        with self.assertRaises(TypeError):
            hub.publish("camera", object())


class TestRobotEdgeCapabilityAuthority(unittest.TestCase):
    """Hardware authority gate against a live fixture Edge."""

    def setUp(self) -> None:
        self.edge = _EdgeServer()
        self.edge.start()
        self.addCleanup(self.edge.stop)

    def test_fixture_edge_keeps_authority_false_even_if_locally_permitted(self) -> None:
        """A fixture Edge reports no hardware authority; local permission must not flip it."""
        registry = create_default_registry(
            execution_mode=ExecutionMode.REMOTE, simulated_capabilities=()
        )
        client = RobotEdgeCapabilityClient(
            edge_url=self.edge.url,
            token="operator-token",
            capability_registry=registry,
            local_hardware_permitted=True,
        )
        try:
            self.assertTrue(client.poll_once())
        finally:
            client.close()
        for name, descriptor in registry.snapshot().items():
            self.assertFalse(
                descriptor["hardware_authority"],
                f"{name} must not have hardware authority from a fixture Edge",
            )

    def test_authority_requires_hardware_mode_and_local_permission(self) -> None:
        """Authority is granted only when Edge reports hardware mode AND local permits it."""
        registry = create_default_registry(
            execution_mode=ExecutionMode.REMOTE, simulated_capabilities=()
        )
        client = RobotEdgeCapabilityClient(
            edge_url=self.edge.url,
            token="operator-token",
            capability_registry=registry,
            local_hardware_permitted=True,
        )
        # Hardware-mode capability snapshot reported by a (hypothetical) hardware Edge.
        client._process_capabilities(
            {
                "capabilities": {
                    "navigation": {
                        "availability": "available",
                        "health": "healthy",
                        "execution_mode": "hardware",
                        "hardware_authority": True,
                    }
                }
            }
        )
        self.assertTrue(registry.get("navigation").hardware_authority)

        # Same snapshot but local permission False -> authority stays False.
        registry2 = create_default_registry(
            execution_mode=ExecutionMode.REMOTE, simulated_capabilities=()
        )
        client2 = RobotEdgeCapabilityClient(
            edge_url=self.edge.url,
            token="operator-token",
            capability_registry=registry2,
            local_hardware_permitted=False,
        )
        client2._process_capabilities(
            {
                "capabilities": {
                    "navigation": {
                        "availability": "available",
                        "health": "healthy",
                        "execution_mode": "hardware",
                        "hardware_authority": True,
                    }
                }
            }
        )
        self.assertFalse(registry2.get("navigation").hardware_authority)
        client2.close()


if __name__ == "__main__":
    unittest.main()
