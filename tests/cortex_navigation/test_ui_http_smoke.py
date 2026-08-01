"""Hardware-free HTTP smoke test for the Gradio UI mount."""

from __future__ import annotations

from pathlib import Path
import os
import sys
import unittest
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


class UiHttpSmokeTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
