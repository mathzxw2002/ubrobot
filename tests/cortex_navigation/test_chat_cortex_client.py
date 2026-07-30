from pathlib import Path
from types import SimpleNamespace
import sys
import threading
import unittest


ROOT = Path(__file__).resolve().parents[2]
CHAT_UI = ROOT / "src" / "chat_ui"
if str(CHAT_UI) not in sys.path:
    sys.path.insert(0, str(CHAT_UI))

from cortex_client import (  # noqa: E402
    CortexBusyError,
    CortexClient,
    CortexRequestError,
    CortexResult,
    CortexUnavailableError,
    RosCortexTransport,
)


class FakeGoal:
    def __init__(self, feedback, result, on_feedback, *, wait_error=None):
        self.feedback = list(feedback)
        self.result = result
        self.on_feedback = on_feedback
        self.wait_error = wait_error
        self.wait_timeouts = []
        self.cancel_timeouts = []

    def wait(self, timeout_sec):
        self.wait_timeouts.append(timeout_sec)
        for item in self.feedback:
            self.on_feedback(item)
        if self.wait_error:
            raise self.wait_error
        return self.result

    def cancel(self, timeout_sec):
        self.cancel_timeouts.append(timeout_sec)
        return True


class FakeTransport:
    def __init__(self, *, feedback=(), result=None, wait_error=None):
        self.feedback = list(feedback)
        self.result = result or CortexResult(success=True, text="done")
        self.wait_error = wait_error
        self.tasks = []
        self.goals = []

    def send(self, task, on_feedback):
        self.tasks.append(task)
        goal = FakeGoal(
            self.feedback,
            self.result,
            on_feedback,
            wait_error=self.wait_error,
        )
        self.goals.append(goal)
        return goal


class BlockingGoal:
    def __init__(self, on_feedback):
        self.on_feedback = on_feedback
        self.wait_started = threading.Event()
        self.released = threading.Event()
        self.cancel_timeouts = []

    def wait(self, _timeout_sec):
        self.wait_started.set()
        self.released.wait(2.0)
        return CortexResult(success=False, text="cancelled")

    def cancel(self, timeout_sec):
        self.cancel_timeouts.append(timeout_sec)
        self.released.set()
        return True


class BlockingTransport:
    def __init__(self):
        self.goal = None

    def send(self, _task, on_feedback):
        self.goal = BlockingGoal(on_feedback)
        return self.goal


class ImmediateFuture:
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error

    def done(self):
        return True

    def result(self):
        if self.error:
            raise self.error
        return self.value

    def exception(self):
        return self.error


class FakeRosGoalHandle:
    def __init__(self, result_success=True):
        self.accepted = True
        self.result_success = result_success
        self.cancel_calls = 0

    def get_result_async(self):
        result = SimpleNamespace(success=self.result_success)
        return ImmediateFuture(SimpleNamespace(result=result))

    def cancel_goal_async(self):
        self.cancel_calls += 1
        return ImmediateFuture(SimpleNamespace(goals_canceling=[object()]))


class FakeActionClient:
    available = True
    feedback = [
        ("planning", False),
        ("orchestration ready", True),
    ]
    instances = []

    def __init__(self, node, action_type, action_name):
        self.node = node
        self.action_type = action_type
        self.action_name = action_name
        self.sent_goals = []
        self.goal_handle = FakeRosGoalHandle()
        self.destroyed = False
        type(self).instances.append(self)

    def wait_for_server(self, timeout_sec):
        self.server_timeout = timeout_sec
        return type(self).available

    def send_goal_async(self, goal, feedback_callback):
        self.sent_goals.append(goal)
        for text, completed in type(self).feedback:
            feedback = SimpleNamespace(feedback=text, completed=completed)
            feedback_callback(SimpleNamespace(feedback=feedback))
        return ImmediateFuture(self.goal_handle)

    def destroy(self):
        self.destroyed = True


class FakeExecutor:
    def __init__(self, context):
        self.context = context
        self.nodes = []
        self.stopped = threading.Event()
        self.shutdown_calls = []

    def add_node(self, node):
        self.nodes.append(node)

    def spin(self):
        self.stopped.wait(2.0)

    def shutdown(self, timeout_sec):
        self.shutdown_calls.append(timeout_sec)
        self.stopped.set()
        return True


class FakeNode:
    def __init__(self, name, context):
        self.name = name
        self.context = context
        self.destroyed = False

    def destroy_node(self):
        self.destroyed = True


class FakeVisionLanguageAction:
    class Goal:
        def __init__(self):
            self.task = ""


class FakeRosBindings:
    def __init__(self):
        self.initialized = []
        self.shutdown_contexts = []
        self.context = object()
        self.node = None
        self.executor = None

    def Context(self):
        return self.context

    def Node(self, name, context):
        self.node = FakeNode(name, context)
        return self.node

    def Executor(self, context):
        self.executor = FakeExecutor(context)
        return self.executor

    ActionClient = FakeActionClient
    ActionType = FakeVisionLanguageAction

    def init(self, *, context):
        self.initialized.append(context)

    def shutdown(self, *, context):
        self.shutdown_contexts.append(context)


class CortexClientTest(unittest.TestCase):
    def test_plain_text_feedback_and_final_text_are_unchanged(self):
        transport = FakeTransport(
            feedback=["planning", "moving", "arrived"],
            result=CortexResult(success=True, text="arrived"),
        )
        client = CortexClient(transport, result_timeout_sec=30.0)
        seen = []

        reply = client.execute("请走到椅子旁边", on_feedback=seen.append)

        self.assertEqual(transport.tasks, ["请走到椅子旁边"])
        self.assertEqual(seen, ["planning", "moving", "arrived"])
        self.assertEqual(reply, "arrived")
        self.assertEqual(transport.goals[0].wait_timeouts, [30.0])

    def test_nonempty_input_is_not_trimmed_or_prefix_routed(self):
        transport = FakeTransport(result=CortexResult(True, "ok"))
        client = CortexClient(transport)

        client.execute("  ordinary question without nav prefix  ")

        self.assertEqual(
            transport.tasks,
            ["  ordinary question without nav prefix  "],
        )

    def test_empty_input_is_rejected_before_transport(self):
        transport = FakeTransport()
        client = CortexClient(transport)

        with self.assertRaises(ValueError):
            client.execute("   ")

        self.assertEqual(transport.tasks, [])

    def test_unsuccessful_result_reports_last_feedback(self):
        transport = FakeTransport(
            feedback=["planning", "navigation unavailable"],
            result=CortexResult(success=False, text=""),
        )
        client = CortexClient(transport)

        with self.assertRaisesRegex(CortexRequestError, "navigation unavailable"):
            client.execute("find the chair")

    def test_only_one_request_can_be_active_and_cancel_waits(self):
        transport = BlockingTransport()
        client = CortexClient(transport, cancel_timeout_sec=1.5)
        errors = []

        def run_request():
            try:
                client.execute("first request")
            except CortexRequestError as exc:
                errors.append(exc)

        worker = threading.Thread(target=run_request)
        worker.start()
        self.assertTrue(transport.goal.wait_started.wait(1.0))

        with self.assertRaises(CortexBusyError):
            client.execute("second request")

        self.assertTrue(client.cancel_active())
        worker.join(2.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual(transport.goal.cancel_timeouts, [1.5])
        self.assertEqual(len(errors), 1)
        self.assertFalse(client.cancel_active())


class RosCortexTransportTest(unittest.TestCase):
    def setUp(self):
        FakeActionClient.instances.clear()
        FakeActionClient.available = True
        FakeActionClient.feedback = [
            ("planning", False),
            ("orchestration ready", True),
        ]

    def test_sends_exact_task_and_translates_completed_feedback(self):
        bindings = FakeRosBindings()
        transport = RosCortexTransport(bindings=bindings)
        seen = []
        try:
            goal = transport.send("ordinary request", seen.append)
            result = goal.wait(3.0)

            action_client = FakeActionClient.instances[-1]
            self.assertEqual(action_client.action_name, "/cortex_input_command")
            self.assertEqual(action_client.sent_goals[0].task, "ordinary request")
            self.assertEqual(seen, ["planning", "orchestration ready"])
            self.assertEqual(result, CortexResult(True, "orchestration ready"))
        finally:
            transport.close()

    def test_unavailable_server_fails_after_bounded_wait(self):
        bindings = FakeRosBindings()
        FakeActionClient.available = False
        transport = RosCortexTransport(bindings=bindings, server_timeout_sec=1.25)
        try:
            with self.assertRaises(CortexUnavailableError):
                transport.send("request", lambda _text: None)
            self.assertEqual(FakeActionClient.instances[-1].server_timeout, 1.25)
        finally:
            transport.close()

    def test_cancel_inspects_server_acknowledgement(self):
        bindings = FakeRosBindings()
        transport = RosCortexTransport(bindings=bindings)
        try:
            goal = transport.send("request", lambda _text: None)
            self.assertTrue(goal.cancel(2.0))
            self.assertEqual(FakeActionClient.instances[-1].goal_handle.cancel_calls, 1)
        finally:
            transport.close()

    def test_close_stops_executor_and_releases_ros_context(self):
        bindings = FakeRosBindings()
        transport = RosCortexTransport(bindings=bindings)

        transport.close()

        action_client = FakeActionClient.instances[-1]
        self.assertEqual(bindings.executor.shutdown_calls, [2.0])
        self.assertTrue(action_client.destroyed)
        self.assertTrue(bindings.node.destroyed)
        self.assertEqual(bindings.shutdown_contexts, [bindings.context])

    def test_wait_exception_clears_active_goal(self):
        failing = FakeTransport(wait_error=RuntimeError("transport failed"))
        client = CortexClient(failing)

        with self.assertRaisesRegex(RuntimeError, "transport failed"):
            client.execute("first")

        self.assertFalse(client.cancel_active())


if __name__ == "__main__":
    unittest.main()
