"""Hardware-free HTTP smoke test for the Gradio UI mount."""

from __future__ import annotations

from pathlib import Path
import os
import sys
import unittest
import time
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
import gradio as gr


ROOT = Path(__file__).resolve().parents[2]
CHAT_UI = ROOT / "src" / "chat_ui"
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chat_ui import app as ui_app  # noqa: E402
from chat_ui.pipeline import ChatPipeline  # noqa: E402
from chat_ui.voice_runtime import (  # noqa: E402
    MockVoiceProvider,
    VoiceEvent,
    VoiceEventType,
)


class UiHttpSmokeTest(unittest.TestCase):
    def test_voice_websocket_streams_pcm_into_provider(self):
        provider = MockVoiceProvider()
        with patch.dict(
            os.environ,
            {"UBROBOT_CHAT_BACKEND": "cortex-mock", "UBROBOT_CHAT_MEDIA": "off"},
            clear=False,
        ):
            ui_app.chat_pipeline = ChatPipeline(
                initialize_media=False,
                voice_provider=provider,
            )
            mounted = ui_app.create_fastapi()
            with TestClient(mounted) as client:
                with client.websocket_connect("/api/voice/stream") as websocket:
                    websocket.send_bytes(b"pcm16")

        self.assertEqual(provider.audio_inputs, [b"pcm16"])

    def test_voice_websocket_waits_for_browser_playback_ack(self):
        provider = MockVoiceProvider()
        with patch.dict(
            os.environ,
            {"UBROBOT_CHAT_BACKEND": "cortex-mock", "UBROBOT_CHAT_MEDIA": "off"},
            clear=False,
        ):
            ui_app.chat_pipeline = ChatPipeline(
                initialize_media=False,
                voice_provider=provider,
            )
            mounted = ui_app.create_fastapi()
            with TestClient(mounted) as client:
                with client.websocket_connect("/api/voice/stream") as websocket:
                    deadline = time.monotonic() + 1.0
                    while provider.event_sink is None and time.monotonic() < deadline:
                        time.sleep(0.005)
                    self.assertIsNotNone(provider.event_sink)
                    provider.emit(
                        VoiceEvent(
                            VoiceEventType.INTERACTION_REQUEST,
                            text="状态怎么样",
                            request_id="call-ui",
                        )
                    )
                    deadline = time.monotonic() + 1.0
                    while (
                        ui_app.chat_pipeline.voice_runtime.snapshot().state.value
                        != "speaking"
                        and time.monotonic() < deadline
                    ):
                        time.sleep(0.005)
                    provider.emit(
                        VoiceEvent(
                            VoiceEventType.AUDIO_CHUNK,
                            audio=b"pcm-output",
                        )
                    )
                    provider.emit(
                        VoiceEvent(VoiceEventType.SPEECH_DONE)
                    )

                    self.assertEqual(websocket.receive_bytes(), b"pcm-output")
                    self.assertEqual(
                        websocket.receive_json(),
                        {"type": "provider.speech_done"},
                    )
                    self.assertEqual(
                        ui_app.chat_pipeline.voice_runtime.snapshot().state.value,
                        "speaking",
                    )
                    websocket.send_text('{"type":"playback.done"}')
                    deadline = time.monotonic() + 1.0
                    while (
                        ui_app.chat_pipeline.voice_runtime.snapshot().state.value
                        != "listening"
                        and time.monotonic() < deadline
                    ):
                        time.sleep(0.005)
                    self.assertEqual(
                        ui_app.chat_pipeline.voice_runtime.snapshot().state.value,
                        "listening",
                    )

        self.assertEqual(
            ui_app.chat_pipeline.voice_runtime.snapshot().state.value,
            "idle",
        )

    def test_gradio_ui_mounts_with_offline_cortex_backend(self):
        with patch.dict(
            os.environ,
            {
                "UBROBOT_CHAT_BACKEND": "cortex-mock",
                "UBROBOT_CHAT_MEDIA": "off",
            },
            clear=False,
        ):
            ui_app.chat_pipeline = ChatPipeline(initialize_media=False)
            mounted = gr.mount_gradio_app(
                FastAPI(),
                ui_app.create_gradio(),
                path="/",
            )
            response = TestClient(mounted).get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("UBRobot ChatUI", response.text)
        self.assertEqual(ui_app.chat_pipeline.backend_name, "cortex-mock")

    def test_operator_input_is_native_visible_and_interactive(self):
        with patch.dict(
            os.environ,
            {
                "UBROBOT_CHAT_BACKEND": "cortex-mock",
                "UBROBOT_CHAT_MEDIA": "off",
            },
            clear=False,
        ):
            ui_app.chat_pipeline = ChatPipeline(initialize_media=False)
            demo = ui_app.create_gradio()
            config = demo.get_config_file()

        command_inputs = [
            component
            for component in config["components"]
            if component["type"] == "multimodaltextbox"
            and component["props"].get("elem_id") == "operator-command-input"
        ]
        self.assertEqual(len(command_inputs), 1)
        self.assertTrue(command_inputs[0]["props"]["interactive"])
        self.assertTrue(command_inputs[0]["props"]["visible"])
        self.assertNotIn(
            "modelscopelegacymultimodalinput",
            {component["type"] for component in config["components"]},
        )
        self.assertTrue(
            any(
                component["type"] == "markdown"
                and "Codex 内置预览可能不提供物理麦克风"
                in component["props"].get("value", "")
                and "当前文件上传 ASR：**已关闭**"
                in component["props"].get("value", "")
                for component in config["components"]
            )
        )

        element_ids = {
            component["props"].get("elem_id")
            for component in config["components"]
        }
        self.assertIn("operator-voice-start", element_ids)
        self.assertIn("operator-voice-stop", element_ids)
        self.assertIn("operator-voice-retry", element_ids)
        self.assertIn("operator-emergency-stop", element_ids)

        timer_ids = {
            component["id"]
            for component in config["components"]
            if component["type"] == "timer"
        }
        self.assertTrue(timer_ids)
        self.assertTrue(
            any(
                dependency["targets"]
                and dependency["targets"][0][0] in timer_ids
                and dependency["targets"][0][1] == "tick"
                for dependency in config["dependencies"]
            )
        )


if __name__ == "__main__":
    unittest.main()
