from __future__ import annotations

import json
from pathlib import Path
import unittest

from src.chat_ui.qwen_realtime import QwenOmniRealtimeProvider, QwenRealtimeConfig
from src.chat_ui.voice_runtime import VoiceEventType


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "qwen_realtime_events.json"
)


class QwenRealtimeFixtureTest(unittest.TestCase):
    def setUp(self):
        self.provider = QwenOmniRealtimeProvider(
            QwenRealtimeConfig(api_key="secret", workspace_id="workspace")
        )
        self.events = []
        self.provider._sink = self.events.append
        self.fixture = {
            item["name"]: item["event"]
            for item in json.loads(FIXTURE.read_text(encoding="utf-8"))
        }

    def test_transcript_and_vad_events_are_independent_of_tool_completion(self):
        for name in ("connected", "vad_started", "partial", "final", "vad_stopped"):
            self.provider._handle_server_event(self.fixture[name])

        self.assertEqual(
            [event.event_type for event in self.events],
            [
                VoiceEventType.CONNECTED,
                VoiceEventType.VAD_STARTED,
                VoiceEventType.TRANSCRIPT_PARTIAL,
                VoiceEventType.TRANSCRIPT_FINAL,
                VoiceEventType.VAD_STOPPED,
            ],
        )
        self.assertEqual(self.events[2].text, "导航到前面")
        self.assertEqual(self.events[3].text, "导航到前面的椅子")

    def test_authorized_tool_result_audio_and_speech_done_are_forwarded(self):
        self.provider._handle_server_event(self.fixture["tool_call"])
        self.provider.complete_interaction("call-1", "执行完成")
        self.provider._waiting_for_tool_phase_done = False
        self.provider._handle_server_event(self.fixture["audio"])
        self.provider._handle_server_event(self.fixture["speech_done"])

        self.assertEqual(self.events[0].event_type, VoiceEventType.INTERACTION_REQUEST)
        self.assertEqual(self.events[1].event_type, VoiceEventType.AUDIO_CHUNK)
        self.assertEqual(self.events[1].audio, b"pcm")
        self.assertEqual(self.events[2].event_type, VoiceEventType.SPEECH_DONE)

    def test_error_and_disconnect_map_without_exposing_credentials(self):
        self.provider._handle_server_event(self.fixture["error"])
        self.provider._handle_server_event(self.fixture["disconnect"])

        self.assertEqual(self.events[0].event_type, VoiceEventType.ERROR)
        self.assertEqual(self.events[0].error, "sanitized failure")
        self.assertEqual(self.events[1].event_type, VoiceEventType.DISCONNECTED)
        self.assertNotIn("secret", repr(self.events))


if __name__ == "__main__":
    unittest.main()
