#!/usr/bin/env python3
"""Real-planner mock validation: UI client -> real LLM Cortex -> mock wheels.

Same topology as ``end_to_end_mock_test.py`` but the planner is a real
OpenAI-compatible LLM reached through ``planner_relay.py``. Assertions are
behavioral (the LLM is non-deterministic):

1. navigation prompt -> the planner must select the NavigateToObject tool:
   a command lease appears, mock wheels show the forward signature, motion
   stops within the 300 ms deadline after the goal ends;
2. non-motion prompt -> no lease, no non-zero /cmd_vel, reply returned;
3. cancel mid-execution -> acknowledged within two seconds, lease empties,
   motion stops.

Prompt delivery is verified through Cortex's own feedback echo
("Received task. Creating a plan for: <prompt>").
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor

from cortex_client import create_ros_cortex_client
from cortex_navigation_mock_test import (
    DeterministicTrackingFixture,
    HarnessNode,
    POST_STOP_OBSERVATION_SEC,
    STOP_DEADLINE_SEC,
)
from end_to_end_mock_test import (
    assert_forward_signature,
    assert_stopped,
    forward_signature,
    wait_for,
)

NAVIGATION_PROMPT = "请走到椅子旁边"
NON_MOTION_PROMPT = "用一句话报告系统当前状态。不要移动机器人，也不要调用任何工具。"
CANCEL_PROMPT = "请走到椅子旁边（取消用例）"


class RealPlannerMockTest:
    def __init__(self, args):
        self.args = args
        self.fixture = DeterministicTrackingFixture()
        self.fixture.configure(complete_after_sec=args.fixture_goal_sec)
        self.harness = HarnessNode()
        self.executor = MultiThreadedExecutor(num_threads=8)
        self.executor.add_node(self.fixture)
        self.executor.add_node(self.harness)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()

    def close(self):
        self.fixture.stop_active_goals()
        time.sleep(0.1)
        self.executor.shutdown(timeout_sec=2.0)
        self.spin_thread.join(2.0)
        self.fixture.destroy_node()
        self.harness.destroy_node()

    def wait_ready(self):
        wait_for(
            lambda: bool(self.harness.snapshot()[0]),
            10.0,
            "guard produced no /cmd_vel samples",
        )

    def baseline(self):
        start = time.monotonic()
        time.sleep(3.0)
        cmd_samples, _raw, _joints, lease_samples, _feedback = self.harness.snapshot()
        nonzero = [s for s in cmd_samples if s.timestamp >= start and s.nonzero]
        active = [v for ts, v in lease_samples if ts >= start and v]
        if nonzero or active:
            raise AssertionError("baseline observed motion or an active lease")
        return {"cmd_samples": len(cmd_samples)}

    def run_navigation(self, client):
        feedback: list[str] = []
        started_at = time.monotonic()
        try:
            reply = client.execute(NAVIGATION_PROMPT, on_feedback=feedback.append)
        finally:
            client.cancel_active()
        completed_at = time.monotonic()
        if not reply:
            raise AssertionError("Cortex returned no final reply")
        if not feedback:
            raise AssertionError("Cortex returned no Action feedback")
        if not any(NAVIGATION_PROMPT in text for text in feedback):
            raise AssertionError(
                "no feedback echoed the prompt; delivery cannot be verified"
            )

        time.sleep(STOP_DEADLINE_SEC + POST_STOP_OBSERVATION_SEC)
        zero_after = assert_stopped(self.harness, completed_at)

        cmd_samples, _raw, joint_samples, lease_samples, _ = self.harness.snapshot()
        windowed_leases = [v for ts, v in lease_samples if started_at <= ts and v]
        if not windowed_leases:
            raise AssertionError(
                "real planner never issued the NavigateToObject goal "
                "(no command lease appeared)"
            )
        signature = assert_forward_signature(forward_signature(joint_samples))
        return {
            "prompt": NAVIGATION_PROMPT,
            "reply": reply,
            "feedback_samples": len(feedback),
            "duration_sec": completed_at - started_at,
            "wheel_signature": signature,
            "zero_cmd_samples_after_deadline": zero_after,
        }

    def run_non_motion(self, client):
        feedback: list[str] = []
        started_at = time.monotonic()
        cmd_before = len(self.harness.snapshot()[0])
        lease_before = len(self.harness.snapshot()[3])
        try:
            reply = client.execute(NON_MOTION_PROMPT, on_feedback=feedback.append)
        finally:
            client.cancel_active()
        time.sleep(STOP_DEADLINE_SEC + POST_STOP_OBSERVATION_SEC)
        if not reply:
            raise AssertionError("non-motion prompt returned no reply")
        cmd_samples, _raw, _joints, lease_samples, _ = self.harness.snapshot()
        nonzero = [
            s for s in cmd_samples[cmd_before:] if s.nonzero
        ]
        active = [v for _ts, v in lease_samples[lease_before:] if v]
        if nonzero or active:
            raise AssertionError(
                "non-motion prompt produced motion or a lease "
                f"(nonzero={len(nonzero)}, leases={len(active)})"
            )
        return {
            "prompt": NON_MOTION_PROMPT,
            "reply": reply,
            "duration_sec": time.monotonic() - started_at,
        }

    def run_cancel(self, client):
        feedback: list[str] = []
        outcome = {"reply": None, "error": None}

        def execute():
            try:
                outcome["reply"] = client.execute(
                    CANCEL_PROMPT, on_feedback=feedback.append
                )
            except Exception as exc:  # noqa: BLE001 - recorded as evidence
                outcome["error"] = f"{type(exc).__name__}: {exc}"

        worker = threading.Thread(target=execute, name="real-planner-cancel")
        worker_started = time.monotonic()
        worker.start()
        try:
            wait_for(
                lambda: any(
                    value
                    for ts, value in self.harness.snapshot()[3]
                    if ts > worker_started and value
                ),
                120.0,
                "cancel probe never acquired a command lease",
            )
            cancel_started = time.monotonic()
            acknowledged = client.cancel_active()
            cancel_latency = time.monotonic() - cancel_started
            if not acknowledged:
                raise AssertionError("Cortex cancellation was not acknowledged")
            if not math.isfinite(cancel_latency) or cancel_latency > 2.0:
                raise AssertionError(
                    f"cancellation acknowledgement took {cancel_latency:.3f}s"
                )
            worker.join(60.0)
            if worker.is_alive():
                raise TimeoutError("cancelled request did not finish")
            wait_for(
                lambda: any(
                    ts > cancel_started and not value
                    for ts, value in self.harness.snapshot()[3]
                ),
                5.0,
                "lease never emptied after cancellation",
            )
            lease_empty_at = min(
                ts
                for ts, value in self.harness.snapshot()[3]
                if ts > cancel_started and not value
            )
            time.sleep(STOP_DEADLINE_SEC + POST_STOP_OBSERVATION_SEC)
            zero_after = assert_stopped(self.harness, lease_empty_at)
            return {
                "cancel_acknowledged": acknowledged,
                "cancel_latency_sec": cancel_latency,
                "request_reply": outcome["reply"],
                "request_error": outcome["error"],
                "zero_cmd_samples_after_lease_end": zero_after,
            }
        finally:
            client.cancel_active()
            worker.join(10.0)

    def run(self):
        self.wait_ready()
        client = create_ros_cortex_client()
        try:
            return {
                "passed": True,
                "baseline": self.baseline(),
                "navigation": self.run_navigation(client),
                "non_motion": self.run_non_motion(client),
                "cancel": self.run_cancel(client),
            }
        finally:
            client.cancel_active()
            client.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--fixture-goal-sec",
        type=float,
        default=6.0,
        help="seconds until the downstream fixture completes a goal",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    test = RealPlannerMockTest(args)
    try:
        result = test.run()
        rendered = json.dumps(result, indent=2, sort_keys=True, default=str)
        print(rendered, flush=True)
        if args.output:
            args.output.write_text(rendered + "\n", encoding="utf-8")
        return 0
    except Exception as exc:  # noqa: BLE001 - top-level test result
        result = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        rendered = json.dumps(result, indent=2, sort_keys=True)
        print(rendered, flush=True)
        if args.output:
            args.output.write_text(rendered + "\n", encoding="utf-8")
        return 1
    finally:
        test.close()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
