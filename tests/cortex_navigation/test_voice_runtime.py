from __future__ import annotations

import json
import os
import threading
import time
import unittest
from unittest.mock import patch

from src.chat_ui.qwen_realtime import QwenOmniRealtimeProvider, QwenRealtimeConfig
from src.chat_ui.interaction_runtime import InteractionCategory, InteractionRuntime
from src.chat_ui.task_runtime import TaskRuntime, TaskStatus
from src.chat_ui.voice_runtime import (
    MockVoiceProvider,
    VoiceEvent,
    VoiceEventType,
    VoiceSessionManager,
    VoiceState,
)
from src.chat_ui.event_stream import EventStream


class SafetyBackend:
    def __init__(self):
        self.entered = threading.Event()
        self.cancelled = threading.Event()
        self.emergency_calls = 0

    def execute(self, task, *, on_feedback):
        on_feedback("moving")
        self.entered.set()
        self.cancelled.wait(1.0)
        return "stopped"

    def cancel_active(self):
        self.cancelled.set()
        return True

    def emergency_stop(self):
        self.emergency_calls += 1
        self.cancelled.set()
        return True


class VoiceSessionManagerTest(unittest.TestCase):
    @staticmethod
    def wait_for_state(manager, state, timeout=1.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if manager.snapshot().state == state:
                return
            time.sleep(0.005)
        raise AssertionError(f"voice state did not become {state}")

    def test_half_duplex_rejects_input_while_speaking(self):
        provider = MockVoiceProvider()
        manager = VoiceSessionManager(
            provider,
            interaction_handler=lambda text: f"accepted: {text}",
            emergency_stop_handler=lambda source: True,
        )
        manager.start()
        self.assertEqual(manager.snapshot().state, VoiceState.LISTENING)
        self.assertTrue(manager.push_audio(b"pcm"))

        provider.emit(
            VoiceEvent(
                VoiceEventType.INTERACTION_REQUEST,
                text="导航到前面的椅子",
                request_id="call-1",
            )
        )

        self.wait_for_state(manager, VoiceState.SPEAKING)
        self.assertEqual(manager.snapshot().state, VoiceState.SPEAKING)
        self.assertFalse(manager.push_audio(b"ignored while speaking"))
        self.assertEqual(provider.completed, [("call-1", "accepted: 导航到前面的椅子")])
        provider.emit(VoiceEvent(VoiceEventType.SPEECH_DONE))
        self.assertEqual(manager.snapshot().state, VoiceState.LISTENING)

    def test_emergency_stop_preempts_speaking(self):
        provider = MockVoiceProvider()
        sources = []
        manager = VoiceSessionManager(
            provider,
            interaction_handler=lambda text: "moving",
            emergency_stop_handler=lambda source: sources.append(source) or True,
        )
        manager.start()
        provider.emit(
            VoiceEvent(
                VoiceEventType.INTERACTION_REQUEST,
                text="向前走",
                request_id="call-2",
            )
        )

        self.wait_for_state(manager, VoiceState.SPEAKING)
        self.assertTrue(manager.emergency_stop("local-keyword"))
        self.assertEqual(manager.snapshot().state, VoiceState.EMERGENCY_STOPPED)
        self.assertEqual(sources, ["local-keyword"])
        self.assertGreaterEqual(provider.cancel_count, 1)

    def test_new_session_clears_stale_transcript_and_error(self):
        provider = MockVoiceProvider()
        manager = VoiceSessionManager(
            provider,
            interaction_handler=lambda text: text,
            emergency_stop_handler=lambda source: True,
        )
        manager.handle_provider_event(
            VoiceEvent(VoiceEventType.ERROR, error="old failure")
        )
        manager._final = "old transcript"

        manager.start()

        snapshot = manager.snapshot()
        self.assertEqual(snapshot.state, VoiceState.LISTENING)
        self.assertEqual(snapshot.transcript_final, "")
        self.assertIsNone(snapshot.last_error)

    def test_audio_playback_ack_controls_return_to_listening(self):
        provider = MockVoiceProvider()
        manager = VoiceSessionManager(
            provider,
            interaction_handler=lambda text: "accepted",
            emergency_stop_handler=lambda source: True,
            audio_sink=lambda chunk: None,
        )
        manager.start()
        provider.emit(
            VoiceEvent(
                VoiceEventType.INTERACTION_REQUEST,
                text="导航到椅子",
                request_id="call-playback",
            )
        )
        self.wait_for_state(manager, VoiceState.SPEAKING)

        provider.emit(VoiceEvent(VoiceEventType.AUDIO_CHUNK, audio=b"pcm"))
        provider.emit(VoiceEvent(VoiceEventType.SPEECH_DONE))

        self.assertEqual(manager.snapshot().state, VoiceState.SPEAKING)
        self.assertTrue(manager.snapshot().playback_pending)
        self.assertTrue(manager.playback_finished())
        self.assertEqual(manager.snapshot().state, VoiceState.LISTENING)

    def test_stale_provider_events_from_previous_session_are_ignored(self):
        provider = MockVoiceProvider()
        manager = VoiceSessionManager(
            provider,
            interaction_handler=lambda text: text,
            emergency_stop_handler=lambda source: True,
        )
        manager.start()
        stale_sink = provider.event_sink
        manager.stop()
        manager.start()

        stale_sink(VoiceEvent(VoiceEventType.ERROR, error="stale failure"))

        self.assertEqual(manager.snapshot().state, VoiceState.LISTENING)
        self.assertIsNone(manager.snapshot().last_error)

    def test_partial_transcript_vad_and_microphone_level_publish_events(self):
        provider = MockVoiceProvider()
        stream = EventStream()
        manager = VoiceSessionManager(
            provider,
            interaction_handler=lambda text: text,
            emergency_stop_handler=lambda source: True,
            event_publisher=stream.publish,
        )
        manager.start()

        provider.emit(VoiceEvent(VoiceEventType.VAD_STARTED))
        provider.emit(
            VoiceEvent(VoiceEventType.TRANSCRIPT_PARTIAL, text="导航到")
        )
        manager.update_microphone_level(0.25)

        snapshot = manager.snapshot()
        self.assertTrue(snapshot.vad_active)
        self.assertEqual(snapshot.transcript_partial, "导航到")
        self.assertEqual(snapshot.microphone_level, 0.25)
        kinds = [event.kind for event in stream.history()]
        self.assertIn("voice.vad", kinds)
        self.assertIn("voice.transcript.partial", kinds)
        self.assertIn("voice.microphone_level", kinds)

    def test_disconnect_does_not_hide_provider_error(self):
        provider = MockVoiceProvider()
        manager = VoiceSessionManager(
            provider,
            interaction_handler=lambda text: text,
            emergency_stop_handler=lambda source: True,
        )
        manager.start()
        manager.handle_provider_event(
            VoiceEvent(VoiceEventType.ERROR, error="connection failed")
        )
        manager.handle_provider_event(VoiceEvent(VoiceEventType.DISCONNECTED))

        snapshot = manager.snapshot()
        self.assertEqual(snapshot.state, VoiceState.ERROR)
        self.assertEqual(snapshot.last_error, "connection failed")

    def test_provider_error_notifies_browser_control_channel(self):
        provider = MockVoiceProvider()
        controls = []
        manager = VoiceSessionManager(
            provider,
            interaction_handler=lambda text: text,
            emergency_stop_handler=lambda source: True,
            control_sink=controls.append,
        )
        manager.start()

        provider.emit(VoiceEvent(VoiceEventType.ERROR, error="network lost"))

        self.assertEqual(controls, ["provider.error"])


class EmergencyStopRuntimeTest(unittest.TestCase):
    def test_emergency_stop_uses_stronger_backend_primitive_and_records_source(self):
        backend = SafetyBackend()
        runtime = TaskRuntime(backend)
        worker = threading.Thread(target=lambda: runtime.execute("向前导航"))
        worker.start()
        self.assertTrue(backend.entered.wait(1.0))

        self.assertTrue(runtime.emergency_stop(source="ui"))
        worker.join(1.0)

        self.assertEqual(backend.emergency_calls, 1)
        self.assertEqual(runtime.tasks()[0].status, TaskStatus.CANCELLED)
        event = next(e for e in runtime.events() if e.event_type == "safety.emergency_stop")
        self.assertEqual(event.data["source"], "ui")

    def test_spoken_emergency_phrase_uses_safety_path(self):
        backend = SafetyBackend()
        tasks = TaskRuntime(backend)
        interactions = InteractionRuntime(tasks)

        result = interactions.handle("紧急叫停机器人", source="voice")

        self.assertEqual(result.turn.category, InteractionCategory.EMERGENCY_STOP)
        self.assertEqual(backend.emergency_calls, 1)
        event = next(e for e in tasks.events() if e.event_type == "safety.emergency_stop")
        self.assertEqual(event.data["source"], "voice")


class QwenProviderContractTest(unittest.TestCase):
    def test_environment_defaults_to_direct_connection(self):
        with patch.dict(
            os.environ,
            {
                "DASHSCOPE_API_KEY": "secret",
                "DASHSCOPE_WORKSPACE_ID": "ws123",
            },
            clear=True,
        ):
            config = QwenRealtimeConfig.from_env()
        self.assertIsNone(config.proxy)

    def test_environment_can_explicitly_enable_proxy_auto_detection(self):
        with patch.dict(
            os.environ,
            {
                "DASHSCOPE_API_KEY": "secret",
                "DASHSCOPE_WORKSPACE_ID": "ws123",
                "UBROBOT_QWEN_REALTIME_PROXY": "auto",
            },
            clear=True,
        ):
            config = QwenRealtimeConfig.from_env()
        self.assertIs(config.proxy, True)

    def test_environment_configures_bounded_session_timeout(self):
        with patch.dict(
            os.environ,
            {
                "DASHSCOPE_API_KEY": "secret",
                "DASHSCOPE_WORKSPACE_ID": "ws123",
                "UBROBOT_QWEN_REALTIME_SESSION_TIMEOUT_SEC": "900",
            },
            clear=True,
        ):
            config = QwenRealtimeConfig.from_env()

        self.assertEqual(config.session_timeout_sec, 900.0)

    def test_url_and_session_expose_only_interaction_tool(self):
        provider = QwenOmniRealtimeProvider(
            QwenRealtimeConfig(api_key="secret", workspace_id="ws123")
        )
        self.assertEqual(
            provider.config.websocket_url,
            "wss://ws123.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime"
            "?model=qwen3.5-omni-plus-realtime",
        )
        session = provider._session_update()["session"]
        self.assertEqual(session["turn_detection"]["type"], "server_vad")
        self.assertEqual(
            [tool["function"]["name"] for tool in session["tools"]],
            ["submit_interaction"],
        )

    def test_native_tool_event_maps_to_neutral_interaction_request(self):
        provider = QwenOmniRealtimeProvider(
            QwenRealtimeConfig(api_key="secret", workspace_id="ws123")
        )
        events = []
        provider._sink = events.append
        provider._handle_server_event(
            {
                "type": "response.function_call_arguments.done",
                "call_id": "call-3",
                "name": "submit_interaction",
                "arguments": json.dumps({"text": "导航到前面的椅子"}),
            }
        )
        self.assertEqual(events[0].event_type, VoiceEventType.INTERACTION_REQUEST)
        self.assertEqual(events[0].request_id, "call-3")
        self.assertEqual(events[0].text, "导航到前面的椅子")

    def test_session_updated_marks_provider_connected(self):
        provider = QwenOmniRealtimeProvider(
            QwenRealtimeConfig(api_key="secret", workspace_id="ws123")
        )
        events = []
        provider._sink = events.append

        provider._handle_server_event({"type": "session.updated"})

        self.assertTrue(provider._connected.is_set())
        self.assertEqual(events[0].event_type, VoiceEventType.CONNECTED)

    def test_unapproved_provider_audio_is_not_forwarded(self):
        provider = QwenOmniRealtimeProvider(
            QwenRealtimeConfig(api_key="secret", workspace_id="ws123")
        )
        events = []
        provider._sink = events.append
        provider._handle_server_event(
            {"type": "response.audio.delta", "delta": "bm90LWF1dGhvcml6ZWQ="}
        )
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
