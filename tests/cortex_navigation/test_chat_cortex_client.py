from pathlib import Path
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

    def test_wait_exception_clears_active_goal(self):
        failing = FakeTransport(wait_error=RuntimeError("transport failed"))
        client = CortexClient(failing)

        with self.assertRaisesRegex(RuntimeError, "transport failed"):
            client.execute("first")

        self.assertFalse(client.cancel_active())


if __name__ == "__main__":
    unittest.main()
