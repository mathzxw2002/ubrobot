from __future__ import annotations

from datetime import datetime, timedelta, timezone
import inspect
import json
import time
import unittest

from src.chat_ui.adapters import cortex as cortex_adapters
from src.chat_ui.adapters import telemetry as telemetry_adapters
from src.chat_ui.adapters.cortex import FixtureCortexAdapter
from src.chat_ui.adapters.telemetry import (
    CameraTelemetry,
    FixtureTelemetryAdapter,
    JointStatesTelemetry,
    NavigationLeaseTelemetry,
    OdometryTelemetry,
    TelemetryState,
)
from src.chat_ui.telemetry import TelemetryHub


class TelemetryDTOTest(unittest.TestCase):
    def test_all_dtos_serialize_to_json_primitives(self):
        values = [
            CameraTelemetry(
                state=TelemetryState.AVAILABLE,
                source="fixture",
                width=640,
                height=480,
                encoding="rgb8",
            ),
            OdometryTelemetry(
                state=TelemetryState.AVAILABLE,
                source="fixture",
                x=1.0,
                y=2.0,
                yaw=0.5,
            ),
            JointStatesTelemetry(
                state=TelemetryState.AVAILABLE,
                source="fixture",
                names=("joint_1", "joint_2"),
                positions=(0.1, 0.2),
            ),
            NavigationLeaseTelemetry(
                state=TelemetryState.AVAILABLE,
                source="fixture",
                owner="operator-a",
                lease_id="lease-1",
                expires_at=datetime.now(timezone.utc) + timedelta(seconds=10),
            ),
        ]

        for value in values:
            encoded = json.dumps(value.to_dict())
            self.assertIn(value.channel, encoded)

    def test_joint_arrays_must_match_names(self):
        with self.assertRaises(ValueError):
            JointStatesTelemetry(
                state=TelemetryState.AVAILABLE,
                source="fixture",
                names=("joint_1",),
                positions=(0.1, 0.2),
            )


class FixtureTelemetryAdapterTest(unittest.TestCase):
    def test_missing_channels_are_explicitly_disconnected(self):
        adapter = FixtureTelemetryAdapter(
            {
                "odometry": OdometryTelemetry(
                    state=TelemetryState.AVAILABLE,
                    source="fixture",
                    x=1.0,
                )
            }
        )

        snapshot = adapter.snapshot()

        self.assertEqual(snapshot["odometry"]["state"], "available")
        self.assertEqual(snapshot["camera"]["state"], "disconnected")
        self.assertIn("no fixture", snapshot["camera"]["detail"])

    def test_fixture_publishes_explicit_states_into_hub(self):
        hub = TelemetryHub()
        FixtureTelemetryAdapter().publish_all(hub)

        snapshot = hub.snapshot()

        self.assertEqual(snapshot["camera"]["state"], "disconnected")
        self.assertFalse(snapshot["camera"]["available"])
        self.assertTrue(snapshot["camera"]["disconnected"])


class TelemetryBoundaryTest(unittest.TestCase):
    def test_hub_rejects_sdk_like_objects_and_binary_buffers(self):
        class CameraHandle:
            pass

        hub = TelemetryHub()
        for value in (CameraHandle(), b"raw-frame"):
            with self.assertRaises(TypeError):
                hub.publish("camera", value)

    def test_stale_state_is_explicit_and_does_not_fabricate_live_data(self):
        hub = TelemetryHub(stale_after_sec=0.01)
        hub.publish(
            "odometry",
            OdometryTelemetry(
                state=TelemetryState.AVAILABLE,
                source="fixture",
                x=1.0,
            ),
        )
        time.sleep(0.02)

        sample = hub.snapshot()["odometry"]

        self.assertEqual(sample["state"], "stale")
        self.assertTrue(sample["stale"])
        self.assertFalse(sample["available"])

    def test_declared_unavailable_state_is_not_promoted_to_available(self):
        hub = TelemetryHub()
        hub.publish(
            "camera",
            CameraTelemetry(
                state=TelemetryState.UNAVAILABLE,
                source="fixture",
                detail="fixture says camera is unavailable",
            ),
        )

        sample = hub.snapshot()["camera"]

        self.assertEqual(sample["state"], "unavailable")
        self.assertFalse(sample["available"])

    def test_workstation_adapters_have_no_hardware_or_ros_imports(self):
        source = (
            inspect.getsource(telemetry_adapters)
            + inspect.getsource(cortex_adapters)
        ).lower()
        forbidden_imports = (
            "import rclpy",
            "import pyrealsense",
            "import unitree",
            "import piper",
            "from rclpy",
        )
        for forbidden in forbidden_imports:
            self.assertNotIn(forbidden, source)


class FixtureCortexAdapterTest(unittest.TestCase):
    def test_fixture_backend_matches_task_runtime_contract_without_authority(self):
        adapter = FixtureCortexAdapter(
            {"导航到椅子": (("planning", "running", "complete"), "done")}
        )
        feedback = []

        reply = adapter.execute("导航到椅子", on_feedback=feedback.append)

        self.assertEqual(reply, "done")
        self.assertEqual(feedback, ["planning", "running", "complete"])
        self.assertFalse(adapter.hardware_authority)


if __name__ == "__main__":
    unittest.main()
