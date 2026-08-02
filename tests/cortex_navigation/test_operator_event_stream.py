from __future__ import annotations

import os
import queue
import time
import unittest
from unittest.mock import patch
from pathlib import Path

from fastapi.testclient import TestClient

from src.chat_ui import app as ui_app
from src.chat_ui.event_stream import EventStream
from src.chat_ui.pipeline import ChatPipeline
from src.chat_ui.voice_runtime import MockVoiceProvider, VoiceEvent, VoiceEventType


class EventStreamTest(unittest.TestCase):
    def test_history_is_ordered_bounded_and_serializable(self):
        stream = EventStream(max_history=3)
        for index in range(4):
            stream.publish(
                kind="task.feedback",
                source="task_runtime",
                correlation_id="turn-1",
                task_id="task-1",
                payload={"index": index},
            )

        events = stream.history()

        self.assertEqual([event.event_id for event in events], [2, 3, 4])
        self.assertEqual(events[-1].to_dict()["payload"], {"index": 3})
        self.assertEqual(events[-1].correlation_id, "turn-1")

    def test_subscription_replays_events_after_cursor_without_race(self):
        stream = EventStream(max_history=5)
        stream.publish(kind="one", source="test", payload={})
        stream.publish(kind="two", source="test", payload={})

        subscription = stream.subscribe(after_event_id=1, queue_size=2)
        stream.publish(kind="three", source="test", payload={})

        self.assertEqual([event.kind for event in subscription.replay], ["two"])
        self.assertEqual(subscription.get(timeout=0.1).kind, "three")
        subscription.close()

    def test_pipeline_task_events_share_interaction_correlation_id(self):
        class ImmediateBackend:
            def execute(self, task, *, on_feedback):
                on_feedback("running")
                return "done"

            def cancel_active(self):
                return False

        pipeline = ChatPipeline(
            backend=ImmediateBackend(),
            initialize_media=False,
            voice_provider=MockVoiceProvider(),
        )

        pipeline.request_text("导航到前面的椅子")

        events = [
            event
            for event in pipeline.event_stream.history()
            if event.kind.startswith(("interaction.", "task."))
        ]
        correlations = {event.correlation_id for event in events}
        self.assertEqual(len(correlations), 1)
        self.assertNotIn(None, correlations)
        self.assertTrue(all(event.source for event in events))

    def test_browser_client_contains_realtime_recovery_contract(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "chat_ui"
            / "voice_client.js"
        ).read_text(encoding="utf-8")

        self.assertIn("/api/operator/events", source)
        self.assertIn("microphone.level", source)
        self.assertIn("playback.done", source)
        self.assertIn("voiceReconnectAttempts >= 3", source)

    def test_voice_request_correlation_flows_into_interaction_and_task(self):
        class ImmediateBackend:
            def execute(self, task, *, on_feedback):
                on_feedback("running")
                return "done"

            def cancel_active(self):
                return False

        provider = MockVoiceProvider()
        pipeline = ChatPipeline(
            backend=ImmediateBackend(),
            initialize_media=False,
            voice_provider=provider,
        )
        pipeline.voice_runtime.start()
        provider.emit(
            VoiceEvent(
                VoiceEventType.INTERACTION_REQUEST,
                text="导航到前面的椅子",
                request_id="call-correlation",
            )
        )
        deadline = time.monotonic() + 1.0
        while not provider.completed and time.monotonic() < deadline:
            time.sleep(0.005)

        relevant = [
            event
            for event in pipeline.event_stream.history()
            if event.kind == "voice.interaction.request"
            or event.kind.startswith("interaction.")
            or event.kind.startswith("task.")
        ]
        correlations = {event.correlation_id for event in relevant}
        self.assertEqual(len(correlations), 1)
        self.assertIn("call-correlation", next(iter(correlations)))

    def test_slow_subscriber_drops_oldest_and_reports_gap(self):
        stream = EventStream(max_history=10)
        subscription = stream.subscribe(queue_size=2)

        for index in range(3):
            stream.publish(kind=f"event-{index}", source="test", payload={})

        self.assertEqual(subscription.dropped_count(), 1)
        self.assertEqual(subscription.get(timeout=0.1).kind, "event-1")
        self.assertEqual(subscription.get(timeout=0.1).kind, "event-2")
        with self.assertRaises(queue.Empty):
            subscription.get(timeout=0.01)
        subscription.close()


class OperatorEventApiTest(unittest.TestCase):
    def tearDown(self):
        ui_app.chat_pipeline = None

    def test_snapshot_and_websocket_replay_runtime_events(self):
        with patch.dict(
            os.environ,
            {
                "UBROBOT_CHAT_BACKEND": "cortex-mock",
                "UBROBOT_CHAT_MEDIA": "off",
                "UBROBOT_VOICE_PROVIDER": "mock",
            },
            clear=False,
        ):
            pipeline = ChatPipeline(initialize_media=False)
            ui_app.chat_pipeline = pipeline
            pipeline.telemetry_hub.publish("odometry", {"x": 1.0})
            application = ui_app.create_fastapi()
            with TestClient(application) as client:
                snapshot = client.get("/api/operator/snapshot")
                self.assertEqual(snapshot.status_code, 200)
                cursor = snapshot.json()["latest_event_id"]

                pipeline.telemetry_hub.publish("odometry", {"x": 2.0})
                with client.websocket_connect(
                    f"/api/operator/events?after={cursor}"
                ) as websocket:
                    initial = websocket.receive_json()
                    replay = websocket.receive_json()

        self.assertEqual(initial["type"], "snapshot")
        self.assertEqual(replay["type"], "event")
        self.assertEqual(replay["event"]["kind"], "telemetry.updated")
        self.assertEqual(replay["event"]["payload"]["channel"], "odometry")


if __name__ == "__main__":
    unittest.main()
