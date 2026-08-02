from __future__ import annotations

from importlib import metadata
import os
import unittest
from unittest.mock import patch

from packaging.specifiers import SpecifierSet

from src.chat_ui import app as ui_app
from src.chat_ui.pipeline import ChatPipeline


SUPPORTED_RUNTIME = {
    "fastapi": "==0.124.2",
    "gradio": "==5.50.0",
    "gradio-client": "==1.14.0",
    "starlette": "==0.47.2",
    "uvicorn": "==0.35.0",
    "websockets": "==15.0.1",
}


class DependencyContractTest(unittest.TestCase):
    def tearDown(self):
        ui_app.chat_pipeline = None

    def test_operator_console_runtime_versions_are_supported(self):
        for package, specifier in SUPPORTED_RUNTIME.items():
            installed = metadata.version(package)
            self.assertIn(
                installed,
                SpecifierSet(specifier),
                f"{package} {installed} is outside supported range {specifier}",
            )

    def test_mock_fastapi_application_can_be_constructed_without_media(self):
        with patch.dict(
            os.environ,
            {
                "UBROBOT_CHAT_BACKEND": "cortex-mock",
                "UBROBOT_CHAT_MEDIA": "off",
                "UBROBOT_VOICE_PROVIDER": "off",
            },
            clear=False,
        ):
            ui_app.chat_pipeline = ChatPipeline(initialize_media=False)
            application = ui_app.create_fastapi()

        self.assertIsNotNone(application)
        self.assertEqual(ui_app.chat_pipeline.backend_name, "cortex-mock")


if __name__ == "__main__":
    unittest.main()
