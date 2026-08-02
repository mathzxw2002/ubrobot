from __future__ import annotations

import threading
import time
import unittest

from src.chat_ui.interaction_runtime import InteractionCategory, InteractionRuntime
from src.chat_ui.task_runtime import TaskRuntime, TaskStatus
from src.chat_ui.telemetry import TelemetryHub
from tests.cortex_navigation.test_chat_pipeline_routing import load_pipeline_module


class BlockingBackend:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.cancelled = threading.Event()
        self.tasks = []

    def execute(self, task, *, on_feedback):
        self.tasks.append(task)
        on_feedback("planning")
        self.entered.set()
        while not self.release.wait(0.01):
            if self.cancelled.is_set():
                raise RuntimeError("cancelled")
        on_feedback("complete")
        return "done"

    def cancel_active(self):
        self.cancelled.set()
        return True


class TaskRuntimeTest(unittest.TestCase):
    def test_single_active_task_and_pending_queue(self):
        backend = BlockingBackend()
        runtime = TaskRuntime(backend)
        outcome = {}
        worker = threading.Thread(
            target=lambda: outcome.setdefault("first", runtime.execute("去椅子旁边"))
        )
        worker.start()
        self.assertTrue(backend.entered.wait(1.0))

        second = runtime.execute("去门口")

        self.assertFalse(second.dispatched)
        self.assertEqual(second.task.status, TaskStatus.QUEUED)
        self.assertEqual(len(runtime.pending_tasks()), 1)
        self.assertEqual(runtime.active_task().intent, "去椅子旁边")
        backend.release.set()
        worker.join(1.0)
        self.assertEqual(outcome["first"].task.status, TaskStatus.SUCCEEDED)

    def test_parent_and_root_ids_are_retained(self):
        backend = BlockingBackend()
        runtime = TaskRuntime(backend)
        parent = runtime.submit("取水瓶")
        child = runtime.submit("导航到椅子", parent_task_id=parent.task_id, sequence_no=1)

        self.assertEqual(child.parent_task_id, parent.task_id)
        self.assertEqual(child.root_task_id, parent.task_id)
        self.assertEqual(child.sequence_no, 1)

    def test_cancel_marks_running_task_cancelled(self):
        backend = BlockingBackend()
        runtime = TaskRuntime(backend)
        worker = threading.Thread(target=lambda: self._ignore_error(runtime))
        worker.start()
        self.assertTrue(backend.entered.wait(1.0))
        task_id = runtime.active_task().task_id

        self.assertTrue(runtime.cancel_active())
        worker.join(1.0)

        self.assertEqual(runtime.task(task_id).status, TaskStatus.CANCELLED)
        self.assertTrue(any(e.event_type == "task.cancelled" for e in runtime.events()))

    @staticmethod
    def _ignore_error(runtime):
        try:
            runtime.execute("导航")
        except RuntimeError:
            pass


class InteractionRuntimeTest(unittest.TestCase):
    def test_status_query_does_not_dispatch_second_cortex_request(self):
        backend = BlockingBackend()
        tasks = TaskRuntime(backend)
        interactions = InteractionRuntime(tasks)
        worker = threading.Thread(target=lambda: interactions.handle("导航到椅子"))
        worker.start()
        self.assertTrue(backend.entered.wait(1.0))

        result = interactions.handle("任务进度怎么样？", source="voice")

        self.assertEqual(result.turn.category, InteractionCategory.QUERY)
        self.assertEqual(result.turn.source, "voice")
        self.assertFalse(result.dispatched)
        self.assertEqual(backend.tasks, ["导航到椅子"])
        backend.release.set()
        worker.join(1.0)

    def test_voice_cancel_controls_active_task_without_new_task(self):
        backend = BlockingBackend()
        tasks = TaskRuntime(backend)
        interactions = InteractionRuntime(tasks)
        worker = threading.Thread(target=lambda: self._ignore_cancel(interactions))
        worker.start()
        self.assertTrue(backend.entered.wait(1.0))

        result = interactions.handle("停一下", source="voice")
        worker.join(1.0)

        self.assertEqual(result.turn.category, InteractionCategory.CONTROL)
        self.assertFalse(result.dispatched)
        self.assertEqual(backend.tasks, ["导航到椅子"])

    @staticmethod
    def _ignore_cancel(interactions):
        try:
            interactions.handle("导航到椅子")
        except RuntimeError:
            pass


class TelemetryHubTest(unittest.TestCase):
    def test_snapshot_reports_missing_and_stale_channels(self):
        hub = TelemetryHub(history_size=2, stale_after_sec=0.01)
        hub.publish("odometry", {"x": 1.0})
        hub.publish("odometry", {"x": 2.0})
        hub.publish("odometry", {"x": 3.0})

        self.assertEqual([s.sequence for s in hub.history("odometry")], [2, 3])
        self.assertFalse(hub.snapshot()["camera"]["available"])
        time.sleep(0.02)
        self.assertTrue(hub.snapshot()["odometry"]["stale"])


class PipelineInteractionIntegrationTest(unittest.TestCase):
    def test_voice_status_query_remains_available_during_motion_task(self):
        pipeline_module = load_pipeline_module()
        backend = BlockingBackend()
        pipeline = pipeline_module.ChatPipeline(backend=backend, initialize_media=False)
        outcome = {}
        worker = threading.Thread(
            target=lambda: outcome.setdefault(
                "navigation", pipeline.request_text("导航到前面的椅子")
            )
        )
        worker.start()
        self.assertTrue(backend.entered.wait(1.0))

        status = pipeline.request_text("任务进度怎么样？", source="voice")
        snapshot = pipeline.operator_snapshot()

        self.assertIn("导航到前面的椅子", status)
        self.assertEqual(backend.tasks, ["导航到前面的椅子"])
        self.assertEqual(snapshot["interactions"][-1]["source"], "voice")
        self.assertEqual(snapshot["interactions"][-1]["category"], "query")
        self.assertEqual(snapshot["tasks"]["active_task"]["status"], "running")
        backend.release.set()
        worker.join(1.0)
        self.assertEqual(outcome["navigation"], "done")


if __name__ == "__main__":
    unittest.main()
