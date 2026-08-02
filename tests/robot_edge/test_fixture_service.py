"""Test fixture-only Robot Edge service."""

import unittest
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

# These imports will fail until robot_edge is created
try:
    from robot_edge.app import create_app
    from robot_edge.fixture_backend import FixtureBackend
    from robot_edge.runtime import RobotEdgeRuntime
    from ubrobot_contracts.edge_api import (
        CommandRequest,
        CommandState,
        EmergencyStopRequest,
        LeaseAcquireRequest,
    )
    from ubrobot_contracts.capabilities import (
        CapabilityName,
        CapabilitySnapshot,
        CapabilityAvailability,
        CapabilityHealth,
        ExecutionMode,
    )
    from ubrobot_contracts.telemetry import (
        TelemetryChannel,
        TelemetryState,
        TelemetrySnapshot,
        TimestampedSample,
    )
    HAS_SERVICE = True
except ImportError:
    HAS_SERVICE = False


class TestFixtureServiceImports(unittest.TestCase):
    """Test that robot_edge has no hardware imports."""

    FORBIDDEN_MODULES = {
        "rclpy",
        "pyrealsense2",
        "piper_sdk",
        "unitree_sdk2py",
        "lerobot.cameras.realsense",
    }

    @unittest.skipUnless(HAS_SERVICE, "robot_edge not available")
    def test_no_hardware_imports(self) -> None:
        """Robot Edge must not import hardware SDKs in fixture mode."""
        import sys
        imported = set(sys.modules)
        for forbidden in self.FORBIDDEN_MODULES:
            self.assertNotIn(forbidden, imported, f"forbidden import: {forbidden}")


class TestFixtureBackend(unittest.TestCase):
    """Test the fixture backend implementation."""

    @unittest.skipUnless(HAS_SERVICE, "robot_edge not available")
    def test_fixture_backend_initializes(self) -> None:
        """Fixture backend must initialize cleanly."""
        backend = FixtureBackend()
        self.assertFalse(backend.hardware_authority)
        self.assertEqual(backend.execution_mode, "fixture")

    @unittest.skipUnless(HAS_SERVICE, "robot_edge not available")
    def test_fixture_backend_capabilities(self) -> None:
        """Fixture backend must provide expected capabilities."""
        backend = FixtureBackend()
        capabilities = backend.get_capabilities()
        self.assertIsInstance(capabilities, dict)
        self.assertIn(CapabilityName.NAVIGATION, capabilities)
        self.assertIn(CapabilityName.GRASP, capabilities)
        self.assertIn(CapabilityName.STOP, capabilities)
        nav_cap = capabilities[CapabilityName.NAVIGATION]
        self.assertEqual(nav_cap.execution_mode, ExecutionMode.FIXTURE)
        self.assertFalse(nav_cap.hardware_authority)

    @unittest.skipUnless(HAS_SERVICE, "robot_edge not available")
    def test_fixture_backend_telemetry(self) -> None:
        """Fixture backend must provide telemetry snapshots."""
        backend = FixtureBackend()
        telemetry = backend.get_telemetry_snapshot()
        self.assertIsInstance(telemetry, dict)
        self.assertIn(TelemetryChannel.CAMERA, telemetry)
        self.assertIn(TelemetryChannel.DEPTH, telemetry)
        self.assertIn(TelemetryChannel.ODOMETRY, telemetry)


class TestRobotEdgeRuntime(unittest.TestCase):
    """Test the Robot Edge runtime state machine."""

    @unittest.skipUnless(HAS_SERVICE, "robot_edge not available")
    def test_runtime_initializes(self) -> None:
        """Runtime must initialize with fixture backend."""
        backend = FixtureBackend()
        runtime = RobotEdgeRuntime(backend=backend)
        self.assertEqual(runtime.execution_mode, "fixture")
        self.assertFalse(runtime.hardware_authority)
        self.assertEqual(runtime.lease_state, "none")

    @unittest.skipUnless(HAS_SERVICE, "robot_edge not available")
    def test_runtime_command_sequence(self) -> None:
        """Runtime must execute a deterministic fixture command sequence."""
        backend = FixtureBackend()
        runtime = RobotEdgeRuntime(backend=backend)

        # Submit command
        command_id = runtime.submit_command(
            text="导航到椅子",
            operator_id="test-operator",
            correlation_id="trace-1",
        )
        self.assertIsNotNone(command_id)

        # Poll multiple times to get full sequence
        for _ in range(10):
            runtime.poll_events()

        # Get events (should have deterministic sequence)
        events = runtime.get_events_since(event_id=0)
        self.assertGreater(len(events), 0)

        # Check sequence includes planning, running, succeeded
        states = [event.state for event in events]
        self.assertIn(CommandState.ACCEPTED, states)
        self.assertIn(CommandState.PLANNING, states)
        self.assertIn(CommandState.RUNNING, states)
        self.assertIn(CommandState.SUCCEEDED, states)

    @unittest.skipUnless(HAS_SERVICE, "robot_edge not available")
    def test_runtime_emergency_stop(self) -> None:
        """Emergency stop must set latched state."""
        backend = FixtureBackend()
        runtime = RobotEdgeRuntime(backend=backend)

        # Submit a command
        command_id = runtime.submit_command(
            text="导航到椅子",
            operator_id="test-operator",
            correlation_id="trace-1",
        )

        # Trigger emergency stop
        runtime.emergency_stop(operator_id="test-operator", correlation_id="trace-2")

        self.assertTrue(runtime.safety_latched)
        # New commands should be rejected while latched
        with self.assertRaises(Exception):
            runtime.submit_command(
                text="另一个命令",
                operator_id="test-operator",
                correlation_id="trace-3",
            )

    @unittest.skipUnless(HAS_SERVICE, "robot_edge not available")
    def test_runtime_cancel(self) -> None:
        """Cancel must stop an active command."""
        backend = FixtureBackend()
        runtime = RobotEdgeRuntime(backend=backend)

        command_id = runtime.submit_command(
            text="导航到椅子",
            operator_id="test-operator",
            correlation_id="trace-1",
        )

        cancelled = runtime.cancel_command(command_id=command_id, operator_id="test-operator")
        self.assertTrue(cancelled)


class TestRobotEdgeAPI(unittest.TestCase):
    """Test the Robot Edge FastAPI endpoints."""

    @unittest.skipUnless(HAS_SERVICE, "robot_edge not available")
    def setUp(self) -> None:
        """Set up test client."""
        # Create app and manually initialize runtime for testing
        from robot_edge.app import create_app
        import robot_edge.app as app_module
        app = create_app(execution_mode="fixture")

        # Manually initialize runtime
        backend = FixtureBackend()
        runtime = RobotEdgeRuntime(backend=backend)
        app_module._runtime = runtime

        self.client = TestClient(app)
        self.addCleanup(self._cleanup_runtime)

    def _cleanup_runtime(self) -> None:
        """Clean up global runtime after test."""
        import robot_edge.app as app_module
        app_module._runtime = None

    @unittest.skipUnless(HAS_SERVICE, "robot_edge not available")
    def test_health_live(self) -> None:
        """Live endpoint must respond."""
        response = self.client.get("/v1/health/live")
        self.assertEqual(response.status_code, 200)

    @unittest.skipUnless(HAS_SERVICE, "robot_edge not available")
    def test_health_ready(self) -> None:
        """Ready endpoint must return readiness state."""
        response = self.client.get("/v1/health/ready")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["execution_mode"], "fixture")
        self.assertFalse(data["hardware_authority"])

    @unittest.skipUnless(HAS_SERVICE, "robot_edge not available")
    def test_get_capabilities(self) -> None:
        """Capabilities endpoint must return capability inventory."""
        response = self.client.get("/v1/capabilities")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("capabilities", data)
        self.assertIn("navigation", data["capabilities"])

    @unittest.skipUnless(HAS_SERVICE, "robot_edge not available")
    def test_get_telemetry_snapshot(self) -> None:
        """Telemetry endpoint must return snapshot."""
        response = self.client.get("/v1/telemetry/snapshot")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("channels", data)

    @unittest.skipUnless(HAS_SERVICE, "robot_edge not available")
    def test_submit_command(self) -> None:
        """Command submission must work through API."""
        # In fixture mode without auth, this might be disabled
        # or we might need to add auth headers
        pass


class TestEventStream(unittest.TestCase):
    """Test event stream behavior."""

    @unittest.skipUnless(HAS_SERVICE, "robot_edge not available")
    def test_event_history_is_bounded(self) -> None:
        """Event history must be bounded to prevent memory issues."""
        backend = FixtureBackend()
        runtime = RobotEdgeRuntime(backend=backend, max_event_history=10)

        # Submit multiple commands to generate events
        for i in range(20):
            runtime.submit_command(
                text=f"test {i}",
                operator_id="test-operator",
                correlation_id=f"trace-{i}",
            )

        # Get all events
        events = runtime.get_events_since(event_id=0)
        self.assertLessEqual(len(events), 10)  # Should be bounded


class TestServiceExists(unittest.TestCase):
    """Test that the service package exists."""

    def test_service_package_exists(self) -> None:
        """robot_edge must be importable."""
        self.assertTrue(
            HAS_SERVICE,
            "robot_edge package not found - need to create it",
        )


if __name__ == "__main__":
    unittest.main()
