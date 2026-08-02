from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.chat_ui import app as ui_app
from src.chat_ui.capability_registry import (
    CapabilityAvailability,
    CapabilityDescriptor,
    CapabilityHealth,
    CapabilityRegistry,
    ExecutionMode,
    create_default_registry,
)
from src.chat_ui.pipeline import ChatPipeline


class CapabilityRegistryTest(unittest.TestCase):
    def test_default_registry_contains_required_capabilities_and_resources(self):
        registry = create_default_registry(
            execution_mode=ExecutionMode.MOCK,
            simulated_capabilities=("navigation", "grasp", "follow", "stop"),
        )

        snapshot = registry.snapshot()

        self.assertEqual(
            set(snapshot),
            {"navigation", "grasp", "observation", "follow", "stop"},
        )
        self.assertEqual(snapshot["navigation"]["availability"], "available")
        self.assertIn("navigation_lease", snapshot["navigation"]["required_resources"])
        self.assertEqual(snapshot["observation"]["availability"], "disconnected")
        self.assertTrue(all(not item["hardware_authority"] for item in snapshot.values()))
        json.dumps(snapshot)

    def test_mock_or_fixture_descriptor_cannot_claim_hardware_authority(self):
        with self.assertRaises(ValueError):
            CapabilityDescriptor(
                name="navigation",
                availability=CapabilityAvailability.AVAILABLE,
                health=CapabilityHealth.HEALTHY,
                execution_mode=ExecutionMode.MOCK,
                required_resources=("odometry",),
                hardware_authority=True,
            )

    def test_registry_updates_state_without_storing_callbacks_or_handles(self):
        registry = CapabilityRegistry(
            [
                CapabilityDescriptor(
                    name="stop",
                    availability=CapabilityAvailability.DISCONNECTED,
                    health=CapabilityHealth.UNKNOWN,
                    execution_mode=ExecutionMode.REMOTE,
                    required_resources=("safety_control",),
                )
            ]
        )

        updated = registry.update(
            "stop",
            availability=CapabilityAvailability.AVAILABLE,
            health=CapabilityHealth.HEALTHY,
            detail="robot edge heartbeat received",
        )

        self.assertEqual(updated.availability, CapabilityAvailability.AVAILABLE)
        self.assertEqual(registry.snapshot()["stop"]["health"], "healthy")

    def test_mock_pipeline_exposes_only_serialized_capability_state(self):
        with patch.dict(
            "os.environ",
            {"UBROBOT_CHAT_BACKEND": "cortex-mock", "UBROBOT_CHAT_MEDIA": "off"},
            clear=False,
        ):
            pipeline = ChatPipeline(initialize_media=False)
            snapshot = pipeline.operator_snapshot()

        json.dumps(snapshot)
        self.assertEqual(snapshot["capabilities"]["navigation"]["execution_mode"], "mock")
        self.assertNotIn("backend", snapshot)
        self.assertNotIn("telemetry_adapter", snapshot)

    def test_capability_endpoint_is_serialized_and_has_no_hardware_authority(self):
        with patch.dict(
            "os.environ",
            {"UBROBOT_CHAT_BACKEND": "cortex-mock", "UBROBOT_CHAT_MEDIA": "off"},
            clear=False,
        ):
            ui_app.chat_pipeline = ChatPipeline(initialize_media=False)
            with TestClient(ui_app.create_fastapi()) as client:
                response = client.get("/api/operator/capabilities")

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["hardware_authority"])
        self.assertEqual(
            response.json()["capabilities"]["observation"]["availability"],
            "disconnected",
        )


if __name__ == "__main__":
    unittest.main()
