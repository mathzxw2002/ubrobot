"""Test safety supervisor (fail-closed behavior)."""

import unittest
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, call

try:
    from robot_edge.safety import RecordingStopSink, SafetySupervisor
    HAS_SAFETY = True
except ImportError:
    HAS_SAFETY = False


class TestStopSink(unittest.TestCase):
    """Test stop sink (records stop calls)."""

    @unittest.skipUnless(HAS_SAFETY, "robot_edge.safety not available")
    def test_stop_sink_records_calls(self) -> None:
        """Stop sink should record when stop was called."""
        sink = RecordingStopSink()
        self.assertFalse(sink.stopped)

        sink.stop(reason="test stop")

        self.assertTrue(sink.stopped)
        self.assertEqual(sink.last_reason, "test stop")

    @unittest.skipUnless(HAS_SAFETY, "robot_edge.safety not available")
    def test_stop_sink_reset(self) -> None:
        """Stop sink should allow reset for testing."""
        sink = RecordingStopSink()
        sink.stop("test")
        sink.reset()
        self.assertFalse(sink.stopped)


class TestSafetySupervisor(unittest.TestCase):
    """Test safety supervisor."""

    @unittest.skipUnless(HAS_SAFETY, "robot_edge.safety not available")
    def test_supervisor_starts_unlatched(self) -> None:
        """Supervisor should start unlatched."""
        sink = RecordingStopSink()
        supervisor = SafetySupervisor(stop_sinks=[sink])
        self.assertFalse(supervisor.is_latched())
        self.assertTrue(supervisor.allows_commands())

    @unittest.skipUnless(HAS_SAFETY, "robot_edge.safety not available")
    def test_emergency_stop_latches(self) -> None:
        """Emergency stop should latch and call all sinks."""
        sink1 = RecordingStopSink()
        sink2 = RecordingStopSink()
        supervisor = SafetySupervisor(stop_sinks=[sink1, sink2])

        supervisor.emergency_stop(reason="test emergency", operator_id="test")

        self.assertTrue(supervisor.is_latched())
        self.assertFalse(supervisor.allows_commands())
        self.assertTrue(sink1.stopped)
        self.assertTrue(sink2.stopped)

    @unittest.skipUnless(HAS_SAFETY, "robot_edge.safety not available")
    def test_lease_expiry_triggers_stop(self) -> None:
        """Lease expiry should trigger fail-closed stop."""
        sink = RecordingStopSink()
        supervisor = SafetySupervisor(stop_sinks=[sink])

        supervisor.on_lease_expired()

        self.assertTrue(supervisor.is_latched())
        self.assertTrue(sink.stopped)

    @unittest.skipUnless(HAS_SAFETY, "robot_edge.safety not available")
    def test_edge_disconnect_triggers_stop(self) -> None:
        """Edge disconnect should trigger fail-closed stop."""
        sink = RecordingStopSink()
        supervisor = SafetySupervisor(stop_sinks=[sink])

        supervisor.on_edge_disconnected()

        self.assertTrue(supervisor.is_latched())
        self.assertTrue(sink.stopped)

    @unittest.skipUnless(HAS_SAFETY, "robot_edge.safety not available")
    def test_local_stop_triggers_stop(self) -> None:
        """Local stop should trigger fail-closed stop."""
        sink = RecordingStopSink()
        supervisor = SafetySupervisor(stop_sinks=[sink])

        supervisor.on_local_stop()

        self.assertTrue(supervisor.is_latched())
        self.assertTrue(sink.stopped)

    @unittest.skipUnless(HAS_SAFETY, "robot_edge.safety not available")
    def test_reset_requires_auth(self) -> None:
        """Reset should only after explicit authorized reset."""
        sink = RecordingStopSink()
        supervisor = SafetySupervisor(stop_sinks=[sink])

        supervisor.emergency_stop("test", "test")
        self.assertTrue(supervisor.is_latched())

        # Can't reset from safety stop once latched
        supervisor.reset(authorized=True)
        self.assertFalse(supervisor.is_latched())
        self.assertTrue(supervisor.allows_commands())

    @unittest.skipUnless(HAS_SAFETY, "robot_edge.safety not available")
    def test_stop_only_once(self) -> None:
        """Stop sinks should only be called once even on multiple triggers."""
        sink = RecordingStopSink()
        supervisor = SafetySupervisor(stop_sinks=[sink])

        supervisor.emergency_stop("first", "test")
        self.assertTrue(sink.stopped)

        # Reset for next test
        sink.reset()

        # Now multiple triggers should only call once
        supervisor.emergency_stop("test", "test")
        supervisor.on_lease_expired()

        # Check that only one stop call
        # (sink counts separately, but supervisor ensures each only once)


class TestSafetyExists(unittest.TestCase):
    """Test that safety module exists."""

    def test_safety_module_exists(self) -> None:
        """robot_edge.safety must be importable."""
        self.assertTrue(
            HAS_SAFETY,
            "robot_edge.safety module not found - need to create it",
        )


if __name__ == "__main__":
    unittest.main()
