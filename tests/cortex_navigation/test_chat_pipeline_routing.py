import importlib.util
import os
from pathlib import Path
import queue
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
CHAT_UI = ROOT / "src" / "chat_ui"


class FakeBackend:
    def __init__(self, events=None):
        self.tasks = []
        self.cancel_calls = 0
        self.events = events

    def execute(self, task, *, on_feedback):
        self.tasks.append(task)
        on_feedback("planning")
        on_feedback("ready")
        return "final answer"

    def cancel_active(self):
        self.cancel_calls += 1
        if self.events is not None:
            self.events.append("cancel")
        return True


class FakeWorker:
    def __init__(self, name, events):
        self.name = name
        self.events = events
        self.timeouts = []

    def join(self, timeout=None):
        self.timeouts.append(timeout)
        self.events.append(f"join:{self.name}")


def load_pipeline_module():
    torch = ModuleType("torch")
    torch.no_grad = lambda: (lambda value: value)
    gradio = ModuleType("gradio")
    gradio.update = lambda **kwargs: kwargs
    gradio.Info = lambda *_args, **_kwargs: None
    gradio.Error = lambda *_args, **_kwargs: None
    utils = ModuleType("utils")
    utils.get_timestamp_str = lambda: "timestamp"
    utils.merge_audios = lambda path: path
    utils.merge_frames_with_audio = lambda audio: audio
    tts = ModuleType("ubrobot.robots.tts")
    tts.CosyVoice_API = object
    asr = ModuleType("ubrobot.robots.asr")
    asr.Fun_ASR = object

    if str(CHAT_UI) not in sys.path:
        sys.path.insert(0, str(CHAT_UI))
    module_name = "chat_pipeline_under_test"
    sys.modules.pop(module_name, None)
    with patch.dict(
        sys.modules,
        {
            "torch": torch,
            "gradio": gradio,
            "utils": utils,
            "ubrobot.robots.tts": tts,
            "ubrobot.robots.asr": asr,
        },
    ):
        spec = importlib.util.spec_from_file_location(module_name, CHAT_UI / "pipeline.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module


class ChatPipelineRoutingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline_module = load_pipeline_module()

    def test_plain_text_routes_unchanged_and_final_text_is_queued_once(self):
        backend = FakeBackend()
        pipeline = self.pipeline_module.ChatPipeline(
            backend=backend,
            initialize_media=False,
        )

        reply = pipeline.request_text("please approach the chair")

        self.assertEqual(backend.tasks, ["please approach the chair"])
        self.assertEqual(reply, "final answer")
        self.assertEqual(pipeline.vlm_queue.get_nowait(), "final answer")
        with self.assertRaises(queue.Empty):
            pipeline.vlm_queue.get_nowait()
        self.assertEqual(
            [pipeline.cortex_feedback_queue.get_nowait() for _ in range(2)],
            ["planning", "ready"],
        )

    def test_nav_prefix_is_not_required_or_removed(self):
        backend = FakeBackend()
        pipeline = self.pipeline_module.ChatPipeline(
            backend=backend,
            initialize_media=False,
        )

        pipeline.request_text("nav: chair")

        self.assertEqual(backend.tasks, ["nav: chair"])

    def test_default_backend_is_cortex_without_go2_manager(self):
        backend = FakeBackend()
        with patch.dict(os.environ, {"UBROBOT_CHAT_BACKEND": "cortex"}):
            with patch.object(
                self.pipeline_module,
                "create_ros_cortex_client",
                return_value=backend,
            ) as factory:
                pipeline = self.pipeline_module.ChatPipeline(initialize_media=False)

        factory.assert_called_once_with()
        self.assertIs(pipeline.backend, backend)
        self.assertNotIn("Go2Manager", self.pipeline_module.__dict__)

    def test_stop_cancels_action_before_bounded_worker_joins(self):
        events = []
        backend = FakeBackend(events)
        pipeline = self.pipeline_module.ChatPipeline(
            backend=backend,
            initialize_media=False,
        )
        pipeline.tts_thread = FakeWorker("tts", events)
        pipeline.ffmpeg_thread = FakeWorker("ffmpeg", events)

        active = pipeline.stop_pipeline(True)

        self.assertFalse(active)
        self.assertEqual(events, ["cancel", "join:tts", "join:ffmpeg"])
        self.assertTrue(all(pipeline_worker.timeouts[0] is not None for pipeline_worker in (
            pipeline.tts_thread,
            pipeline.ffmpeg_thread,
        )))

    def test_cortex_path_has_no_local_robot_observation(self):
        pipeline = self.pipeline_module.ChatPipeline(
            backend=FakeBackend(),
            initialize_media=False,
        )

        self.assertEqual(pipeline.get_robot_observation(), (None, None))


if __name__ == "__main__":
    unittest.main()
