"""Tests for the offline Windows dev-mode backend and media-off pipeline."""

from __future__ import annotations

from pathlib import Path
import queue
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
CHAT_UI = ROOT / "src" / "chat_ui"
sys.path.insert(0, str(CHAT_UI))

from cortex_client import CortexBusyError, CortexRequestError  # noqa: E402
from mock_backend import MockCortexBackend  # noqa: E402
from tests.cortex_navigation.test_chat_pipeline_routing import (  # noqa: E402
    load_pipeline_module,
)


class MockCortexBackendTest(unittest.TestCase):
    def test_navigation_prompt_streams_feedback_and_completes(self):
        backend = MockCortexBackend(nav_duration_sec=0.3)
        feedback = []

        reply = backend.execute("请走到椅子旁边", on_feedback=feedback.append)

        self.assertEqual(reply, "All 1 steps completed.")
        self.assertEqual(feedback[0], "Received task. Creating a plan for: 请走到椅子旁边")
        self.assertTrue(any("EXECUTE" in text for text in feedback))
        self.assertTrue(any("waiting for async actions" in text for text in feedback))
        self.assertEqual(feedback[-1], "All 1 steps completed.")

    def test_text_prompt_returns_no_actions_reply(self):
        backend = MockCortexBackend()
        feedback = []

        reply = backend.execute("报告系统状态", on_feedback=feedback.append)

        self.assertIn("[No actions needed].", reply)
        self.assertIn("报告系统状态", reply)

    def test_cancel_mid_navigation_raises_and_returns_fast(self):
        backend = MockCortexBackend(nav_duration_sec=30.0)
        feedback = []
        outcome = {}

        def execute():
            try:
                outcome["reply"] = backend.execute(
                    "请走到椅子旁边", on_feedback=feedback.append
                )
            except CortexRequestError as exc:
                outcome["error"] = str(exc)

        worker = threading.Thread(target=execute)
        worker.start()
        time.sleep(0.3)
        started = time.monotonic()
        self.assertTrue(backend.cancel_active())
        worker.join(5.0)

        self.assertFalse(worker.is_alive())
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertIn("Plan aborted", outcome.get("error", ""))
        self.assertTrue(any("Plan aborted" in text for text in feedback))

    def test_second_request_is_rejected_while_active(self):
        backend = MockCortexBackend(nav_duration_sec=5.0)
        worker = threading.Thread(
            target=lambda: backend.execute("走到椅子", on_feedback=lambda _t: None)
        )
        worker.start()
        time.sleep(0.2)
        try:
            with self.assertRaises(CortexBusyError):
                backend.execute("另一个请求", on_feedback=lambda _t: None)
        finally:
            backend.cancel_active()
            worker.join(5.0)

    def test_cancel_without_active_goal_returns_false(self):
        self.assertFalse(MockCortexBackend().cancel_active())


class MediaOffPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline_module = load_pipeline_module()

    def test_cortex_mock_backend_selected_by_env(self):
        with patch.dict(
            __import__("os").environ, {"UBROBOT_CHAT_BACKEND": "cortex-mock"}
        ):
            pipeline = self.pipeline_module.ChatPipeline(initialize_media=False)
        self.assertIsInstance(pipeline.backend, MockCortexBackend)
        self.assertEqual(pipeline.backend_name, "cortex-mock")

    def test_invalid_backend_still_rejected(self):
        with patch.dict(
            __import__("os").environ, {"UBROBOT_CHAT_BACKEND": "bogus"}
        ):
            with self.assertRaises(ValueError):
                self.pipeline_module.ChatPipeline(initialize_media=False)

    def test_run_pipeline_media_off_completes_without_workers(self):
        backend = MockCortexBackend(nav_duration_sec=0.2)
        pipeline = self.pipeline_module.ChatPipeline(
            backend=backend,
            initialize_media=False,
        )
        user_input = SimpleNamespace(text="请走到椅子旁边", files=[])
        messages = [{"role": "system", "content": None}]

        result = pipeline.run_pipeline(user_input, messages)

        self.assertIsNotNone(result)
        self.assertEqual(result[-1]["role"], "assistant")
        self.assertEqual(result[-1]["content"], "All 1 steps completed.")
        # media-off path closed the video queue for yield_results
        self.assertIsNone(pipeline.video_queue.get_nowait())
        # feedback streamed through the cortex queue
        self.assertFalse(pipeline.cortex_feedback_queue.empty())
        # no media workers were started
        self.assertIsNone(getattr(pipeline, "tts_thread", None))
        self.assertIsNone(getattr(pipeline, "ffmpeg_thread", None))

    def test_run_pipeline_media_off_marks_audio_input(self):
        # The text-only reply echoes the task, so the ASR-disabled marker is
        # observable in the assistant content.
        backend = MockCortexBackend(nav_duration_sec=0.2)
        pipeline = self.pipeline_module.ChatPipeline(
            backend=backend,
            initialize_media=False,
        )
        user_input = SimpleNamespace(
            text="你好", files=[SimpleNamespace(path="voice.wav")]
        )

        result = pipeline.run_pipeline(
            user_input, [{"role": "system", "content": None}]
        )

        self.assertIsNotNone(result)
        self.assertIn("[ASR disabled: media off]", result[-1]["content"])


if __name__ == "__main__":
    unittest.main()
