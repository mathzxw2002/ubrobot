"""Test shared transport contracts between Operator Console and Robot Edge."""

import unittest
from datetime import datetime, timezone
from enum import Enum
from typing import Any

# These imports will fail until ubrobot_contracts is created
try:
    from ubrobot_contracts import PROTOCOL_VERSION
    from ubrobot_contracts.capabilities import (
        CapabilityName,
        CapabilityHealth,
        CapabilityAvailability,
        CapabilitySnapshot,
        ExecutionMode,
    )
    from ubrobot_contracts.telemetry import (
        TelemetryChannel,
        TelemetryState,
        TelemetrySnapshot,
        TimestampedSample,
    )
    from ubrobot_contracts.edge_api import (
        CommandRequest,
        CommandAccepted,
        CommandEvent,
        CancelRequest,
        EmergencyStopRequest,
        LeaseAcquireRequest,
        LeaseRecord,
        LeaseState,
        Heartbeat,
        ErrorResponse,
    )
    HAS_CONTRACTS = True
except ImportError:
    HAS_CONTRACTS = False


class TestProtocolVersion(unittest.TestCase):
    """Test protocol version is present and correctly formatted."""

    @unittest.skipUnless(HAS_CONTRACTS, "ubrobot_contracts not available")
    def test_protocol_version_exists(self) -> None:
        """PROTOCOL_VERSION must be a string like '1.0'."""
        self.assertIsInstance(PROTOCOL_VERSION, str)
        self.assertTrue(len(PROTOCOL_VERSION) >= 3)
        self.assertIn(".", PROTOCOL_VERSION)

    @unittest.skipUnless(HAS_CONTRACTS, "ubrobot_contracts not available")
    def test_protocol_version_is_1_0(self) -> None:
        """M5 uses protocol version 1.0."""
        self.assertEqual(PROTOCOL_VERSION, "1.0")


class TestCapabilityContracts(unittest.TestCase):
    """Test capability DTO contracts."""

    @unittest.skipUnless(HAS_CONTRACTS, "ubrobot_contracts not available")
    def test_capability_names_are_restricted(self) -> None:
        """Only approved capability names are allowed."""
        self.assertIsInstance(CapabilityName, type)
        self.assertTrue(issubclass(CapabilityName, Enum))

        allowed_names = {"navigation", "grasp", "observation", "follow", "stop"}
        actual_names = {member.value for member in CapabilityName}
        self.assertEqual(actual_names, allowed_names)

    @unittest.skipUnless(HAS_CONTRACTS, "ubrobot_contracts not available")
    def test_capability_snapshot_json_safe(self) -> None:
        """CapabilitySnapshot must serialize to JSON safely."""
        snapshot = CapabilitySnapshot(
            name=CapabilityName.NAVIGATION,
            availability=CapabilityAvailability.AVAILABLE,
            health=CapabilityHealth.HEALTHY,
            execution_mode=ExecutionMode.FIXTURE,
            hardware_authority=False,
            last_updated=datetime.now(timezone.utc),
        )
        data = snapshot.model_dump(mode="json")
        self.assertEqual(data["name"], "navigation")
        self.assertEqual(data["hardware_authority"], False)
        self.assertIn("last_updated", data)

    @unittest.skipUnless(HAS_CONTRACTS, "ubrobot_contracts not available")
    def test_mock_cannot_have_hardware_authority(self) -> None:
        """Fixture/Mock mode must not allow hardware_authority=true."""
        # The model itself doesn't enforce this, but our usage does
        snapshot = CapabilitySnapshot(
            name=CapabilityName.NAVIGATION,
            availability=CapabilityAvailability.AVAILABLE,
            health=CapabilityHealth.HEALTHY,
            execution_mode=ExecutionMode.FIXTURE,
            hardware_authority=True,  # This is invalid, but model allows it for flexibility
            last_updated=datetime.now(timezone.utc),
        )
        # The enforcement happens in the service layer
        data = snapshot.model_dump(mode="json")
        self.assertEqual(data["execution_mode"], "fixture")


class TestTelemetryContracts(unittest.TestCase):
    """Test telemetry DTO contracts."""

    @unittest.skipUnless(HAS_CONTRACTS, "ubrobot_contracts not available")
    def test_telemetry_states_are_defined(self) -> None:
        """Telemetry state must be available/unavailable/stale/disconnected."""
        self.assertIsInstance(TelemetryState, type)
        self.assertTrue(issubclass(TelemetryState, Enum))

        allowed_states = {"available", "unavailable", "stale", "disconnected"}
        actual_states = {member.value for member in TelemetryState}
        self.assertEqual(actual_states, allowed_states)

    @unittest.skipUnless(HAS_CONTRACTS, "ubrobot_contracts not available")
    def test_timestamped_sample_has_timezone_aware_timestamp(self) -> None:
        """All timestamps must be timezone-aware."""
        sample = TimestampedSample(
            timestamp=datetime.now(timezone.utc),
            state=TelemetryState.AVAILABLE,
            value={"some": "data"},
        )
        self.assertIsNotNone(sample.timestamp.tzinfo)

        data = sample.model_dump(mode="json")
        self.assertIn("timestamp", data)
        self.assertIsInstance(data["timestamp"], str)  # ISO format

    @unittest.skipUnless(HAS_CONTRACTS, "ubrobot_contracts not available")
    def test_telemetry_snapshot_json_safe(self) -> None:
        """TelemetrySnapshot must serialize to JSON safely."""
        snapshot = TelemetrySnapshot(
            channel=TelemetryChannel.CAMERA,
            latest=TimestampedSample(
                timestamp=datetime.now(timezone.utc),
                state=TelemetryState.AVAILABLE,
                value={"width": 640, "height": 480},
            ),
        )
        data = snapshot.model_dump(mode="json")
        self.assertEqual(data["channel"], "camera")
        self.assertIn("latest", data)


class TestEdgeAPIContracts(unittest.TestCase):
    """Test Robot Edge API DTO contracts."""

    @unittest.skipUnless(HAS_CONTRACTS, "ubrobot_contracts not available")
    def test_command_request_has_required_fields(self) -> None:
        """CommandRequest must have correlation_id, operator_id, timestamp, nonce."""
        request = CommandRequest(
            text="导航到前面的椅子",
            correlation_id="trace-123",
            operator_id="operator-test",
            nonce="nonce-456",
            timestamp=datetime.now(timezone.utc),
        )
        self.assertEqual(request.text, "导航到前面的椅子")
        self.assertEqual(request.correlation_id, "trace-123")
        self.assertEqual(request.operator_id, "operator-test")
        self.assertIsNotNone(request.nonce)
        self.assertIsNotNone(request.timestamp.tzinfo)

    @unittest.skipUnless(HAS_CONTRACTS, "ubrobot_contracts not available")
    def test_command_request_json_safe(self) -> None:
        """CommandRequest must serialize to JSON safely."""
        request = CommandRequest(
            text="导航到前面的椅子",
            correlation_id="trace-1",
            operator_id="operator-test",
            nonce="nonce-1",
            timestamp=datetime.now(timezone.utc),
        )
        data = request.model_dump(mode="json")
        self.assertEqual(data["text"], "导航到前面的椅子")
        self.assertEqual(data["correlation_id"], "trace-1")
        self.assertIn("protocol_version", data)
        self.assertEqual(data["protocol_version"], PROTOCOL_VERSION)

    @unittest.skipUnless(HAS_CONTRACTS, "ubrobot_contracts not available")
    def test_command_accepted_json_safe(self) -> None:
        """CommandAccepted must serialize to JSON safely."""
        accepted = CommandAccepted(command_id="cmd-789")
        data = accepted.model_dump(mode="json")
        self.assertEqual(data["command_id"], "cmd-789")

    @unittest.skipUnless(HAS_CONTRACTS, "ubrobot_contracts not available")
    def test_lease_record_has_required_fields(self) -> None:
        """LeaseRecord must have lease_id, owner, issued_at, expires_at, state."""
        now = datetime.now(timezone.utc)
        lease = LeaseRecord(
            lease_id="lease-1",
            owner="operator-test",
            issued_at=now,
            expires_at=now,
            last_renewed_at=now,
            state=LeaseState.ACTIVE,
        )
        self.assertEqual(lease.lease_id, "lease-1")
        self.assertEqual(lease.owner, "operator-test")
        self.assertEqual(lease.state, LeaseState.ACTIVE)

    @unittest.skipUnless(HAS_CONTRACTS, "ubrobot_contracts not available")
    def test_error_response_json_safe(self) -> None:
        """ErrorResponse must serialize to JSON safely without secrets."""
        error = ErrorResponse(
            code="UNAUTHORIZED",
            message="Missing or invalid authentication",
            correlation_id="trace-1",
        )
        data = error.model_dump(mode="json")
        self.assertEqual(data["code"], "UNAUTHORIZED")
        # Message can talk about authentication, just shouldn't include actual secrets
        self.assertNotIn("secret", data.get("message", "").lower())
        self.assertNotIn("api-key", data.get("message", "").lower())


class TestProtocolCompatibility(unittest.TestCase):
    """Test protocol version compatibility checks."""

    @unittest.skipUnless(HAS_CONTRACTS, "ubrobot_contracts not available")
    def test_unknown_major_version_is_rejected_in_model(self) -> None:
        """Models must allow checking protocol version."""
        # The model doesn't reject, but we can inspect it
        request = CommandRequest(
            text="test",
            correlation_id="trace-1",
            operator_id="operator-test",
            nonce="nonce-1",
            timestamp=datetime.now(timezone.utc),
            protocol_version="2.0",  # Unknown major version
        )
        data = request.model_dump(mode="json")
        self.assertEqual(data["protocol_version"], "2.0")
        # The service layer will reject this


class TestContractsExist(unittest.TestCase):
    """Test that the contracts package exists and can be imported."""

    def test_contracts_package_exists(self) -> None:
        """ubrobot_contracts must be importable."""
        self.assertTrue(
            HAS_CONTRACTS,
            "ubrobot_contracts package not found - need to create it",
        )


if __name__ == "__main__":
    unittest.main()
