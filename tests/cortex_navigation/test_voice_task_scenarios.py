"""Deterministic M3 safety scenarios with all robot actions mocked."""

from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

from src.chat_ui import app as ui_app
from src.chat_ui.event_stream import EventStream
from src.chat_ui.interaction_runtime import InteractionCategory, InteractionRuntime
from src.chat_ui.pipeline import ChatPipeline
from src.chat_ui.task_runtime import TaskRuntime, TaskStatus


class SafetyScenarioBackend:
    hardware_authority = False

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.cancelled = threading.Event()
        self.requests: list[str] = []
        self.cancel_calls = 0
        self.emergency_calls = 0

    def execute(self, task, *, on_feedback):
        self.requests.append(task)
        on_feedback("mock planning")
        self.entered.set()
        while not self.release.wait(0.01):
            if self.cancelled.is_set():
                raise RuntimeError("mock execution cancelled")
        on_feedback("mock complete")
        return "mock done"

    def cancel_active(self):
        self.cancel_calls += 1
        self.cancelled.set()
        return self.entered.is_set()

    def emergency_stop(self):
        self.emergency_calls += 1
        self.cancelled.set()
        return True

    def close(self):
        self.cancelled.set()
        self.release.set()


class VoiceTaskSafetyScenariosTest(unittest.TestCase):
    def setUp(self):
        self.backend = SafetyScenarioBackend()
        self.events = EventStream()
        self.tasks = TaskRuntime(
            self.backend,
            event_publisher=self.events.publish,
        )
        self.interactions = InteractionRuntime(
            self.tasks,
            event_publisher=self.events.publish,
        )
        self.worker_errors: list[Exception] = []

    def tearDown(self):
        self.backend.release.set()

    def _start_navigation(self):
        def run():
            try:
                self.interactions.handle("导航到前面的椅子")
            except Exception as exc:  # Cancellation is the expected worker exit.
                self.worker_errors.append(exc)

        worker = threading.Thread(target=run)
        worker.start()
        self.assertTrue(self.backend.entered.wait(1.0))
        return worker

    def test_status_query_during_navigation_never_dispatches_second_command(self):
        worker = self._start_navigation()

        result = self.interactions.handle("任务进度怎么样？", source="voice")

        self.assertEqual(result.turn.category, InteractionCategory.QUERY)
        self.assertFalse(result.dispatched)
        self.assertEqual(self.backend.requests, ["导航到前面的椅子"])
        self.assertEqual(len(self.tasks.tasks()), 1)
        self.backend.release.set()
        worker.join(1.0)
        self.assertFalse(worker.is_alive())

    def test_normal_voice_cancel_controls_active_task_without_dispatch(self):
        worker = self._start_navigation()

        result = self.interactions.handle("停一下", source="voice")
        worker.join(1.0)

        self.assertEqual(result.turn.category, InteractionCategory.CONTROL)
        self.assertFalse(result.dispatched)
        self.assertEqual(self.backend.requests, ["导航到前面的椅子"])
        self.assertEqual(self.backend.cancel_calls, 1)
        self.assertEqual(self.tasks.tasks()[0].status, TaskStatus.CANCELLED)

    def test_spoken_emergency_stop_bypasses_and_supersedes_queue(self):
        worker = self._start_navigation()
        queued = self.interactions.handle("导航到门口", source="text")
        self.assertFalse(queued.dispatched)
        self.assertEqual(len(self.tasks.pending_tasks()), 1)

        result = self.interactions.handle("紧急叫停机器人", source="voice")
        worker.join(1.0)

        self.assertEqual(result.turn.category, InteractionCategory.EMERGENCY_STOP)
        self.assertFalse(result.dispatched)
        self.assertEqual(self.backend.emergency_calls, 1)
        self.assertEqual(self.tasks.pending_tasks(), [])
        self.assertEqual(
            self.tasks.task(queued.task_id).status,
            TaskStatus.SUPERSEDED,
        )
        safety = [
            event
            for event in self.events.history()
            if event.kind == "safety.emergency_stop"
        ]
        self.assertEqual(len(safety), 1)
        self.assertEqual(safety[0].correlation_id, result.turn.turn_id)
        self.assertEqual(safety[0].payload["priority"], "critical")
        self.assertTrue(safety[0].payload["bypass_queue"])
        self.assertEqual(safety[0].payload["superseded_task_ids"], [queued.task_id])

    def test_ui_emergency_stop_uses_independent_safety_path(self):
        pipeline = ChatPipeline(backend=self.backend, initialize_media=False)
        pipeline.backend_name = "cortex-mock"
        ui_app.chat_pipeline = pipeline
        worker_errors = []

        def run():
            try:
                pipeline.request_text("导航到前面的椅子")
            except Exception as exc:
                worker_errors.append(exc)

        worker = threading.Thread(target=run)
        worker.start()
        self.assertTrue(self.backend.entered.wait(1.0))
        turns_before_stop = pipeline.interaction_runtime.turns()

        _voice_status, notice = ui_app.emergency_stop_operator()
        worker.join(1.0)

        self.assertEqual(notice, "紧急停止已确认。")
        self.assertEqual(self.backend.emergency_calls, 1)
        self.assertEqual(pipeline.interaction_runtime.turns(), turns_before_stop)
        safety_events = [
            event
            for event in pipeline.event_stream.history()
            if event.kind == "safety.emergency_stop"
        ]
        self.assertEqual(safety_events[0].payload["source"], "operator-console")

    def test_mock_ui_has_prominent_no_hardware_authority_banner(self):
        with patch.dict(
            "os.environ",
            {"UBROBOT_CHAT_BACKEND": "cortex-mock", "UBROBOT_CHAT_MEDIA": "off"},
            clear=False,
        ):
            ui_app.chat_pipeline = ChatPipeline(initialize_media=False)
            config = ui_app.create_gradio().get_config_file()

        banners = [
            component
            for component in config["components"]
            if component["props"].get("elem_id") == "operator-mock-safety-banner"
        ]
        self.assertEqual(len(banners), 1)
        self.assertTrue(banners[0]["props"]["visible"])
        self.assertIn(
            "MOCK / NO HARDWARE AUTHORITY",
            banners[0]["props"]["value"],
        )


if __name__ == "__main__":
    unittest.main()
