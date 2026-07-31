#!/usr/bin/env python3
"""End-to-end mock validation: Chat UI client -> Cortex -> guarded wheels.

Closes the gap between the 2026-07-30 validations:

- ``chat_cortex_smoke_test.py`` proved UI -> Cortex with no robot stack;
- ``cortex_navigation_mock_test.py`` proved NavigateToObject -> guard ->
  mock wheels with no Cortex and no UI.

This test runs one session with everything connected: the production
``RosCortexTransport`` client, the real Cortex component driven by the
deterministic planner fixture (``mock_planner_server.py``), the capability
server, the lease guard, and the mock LeKiwi driver. The deterministic
TrackVisionTarget fixture still replaces Kompass perception (that is M2).

Assertions per navigation request:

1. the prompt reaches the planner unchanged and the offered tool list is
   exactly the semantic navigation capability plus inspection tools;
2. Cortex issues the NavigateToObject goal (a command lease appears);
3. raw and guarded forward commands flow while the lease is active;
4. mock wheels show the forward signature [one ~0, one negative, one positive];
5. UI feedback streams and a final text reply returns;
6. after completion, /cmd_vel is zero within the 300 ms stop deadline and
   stays zero.

A second request is cancelled mid-execution: cancellation is acknowledged
within two seconds and no lease or motion survives the request.
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
    ZERO_EPSILON,
)

NAVIGATION_PROMPT = "请走到椅子旁边"
CANCEL_PROMPT = "请走到椅子旁边（取消用例）"
NAVIGATION_TOOL_NAME = "send_goal_to__ubrobot_navigation_navigate_to_object"
EXPECTED_TOOLS = {
    "inspect_component",
    "update_parameter",
    NAVIGATION_TOOL_NAME,
}
WHEEL_SIGN_MIN = 0.1


def wait_for(condition, timeout_sec, description):
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.02)
    raise TimeoutError(description)


def forward_signature(joint_samples):
    """Return the peak-motion wheel velocity triple, or None if no motion."""
    moving = [sample for sample in joint_samples if sample.nonzero]
    if not moving:
        return None
    peak = max(
        moving,
        key=lambda sample: max(abs(value) for value in sample.velocities),
    )
    return {"names": list(peak.names), "velocities": list(peak.velocities)}


def assert_forward_signature(signature):
    if signature is None:
        raise AssertionError("mock wheels never showed forward motion")
    velocities = signature["velocities"]
    if len(velocities) != 3:
        raise AssertionError(f"expected 3 wheels, got {velocities}")
    peak = max(abs(value) for value in velocities)
    if peak < WHEEL_SIGN_MIN:
        raise AssertionError(f"peak wheel velocity too small: {velocities}")
    signs = []
    for value in velocities:
        if abs(value) < 0.1 * peak:
            signs.append(0)
        elif value > 0:
            signs.append(1)
        else:
            signs.append(-1)
    if sorted(signs) != [-1, 0, 1]:
        raise AssertionError(
            f"forward signature mismatch: {signature['names']} {velocities}"
        )
    return {"names": signature["names"], "velocities": velocities, "signs": signs}


def assert_stopped(harness, since, deadline=STOP_DEADLINE_SEC):
    """All /cmd_vel samples after `since + deadline` must be zero."""
    cmd_samples, _raw, _joints, _lease, _feedback = harness.snapshot()
    late = [s for s in cmd_samples if s.timestamp > since + deadline]
    nonzero = [s for s in late if s.nonzero]
    if nonzero:
        raise AssertionError(
            f"{len(nonzero)} non-zero /cmd_vel samples after stop deadline"
        )
    return len(late)


def read_planner_requests(log_path):
    if not log_path or not Path(log_path).exists():
        return []
    requests = []
    for line in Path(log_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            requests.append(json.loads(line))
    return requests


def assert_planner_evidence(requests, prompt):
    if not requests:
        raise AssertionError("planner fixture recorded no requests")
    tool_names = set()
    saw_prompt = False
    for record in requests:
        payload = record.get("payload", {})
        for tool in payload.get("tools") or []:
            function = tool.get("function", {})
            if function.get("name"):
                tool_names.add(function["name"])
        for message in payload.get("messages") or []:
            if message.get("role") == "user" and message.get("content") == prompt:
                saw_prompt = True
    if not saw_prompt:
        raise AssertionError("prompt did not reach the planner unchanged")
    missing = EXPECTED_TOOLS - tool_names
    if missing:
        raise AssertionError(f"planner was not offered expected tools: {missing}")
    unexpected = {
        name
        for name in tool_names
        if name.startswith("send_goal_to_") and name != NAVIGATION_TOOL_NAME
    }
    if unexpected:
        raise AssertionError(f"unexpected action tools offered: {unexpected}")
    return {"tools_seen": sorted(tool_names), "requests": len(requests)}


class EndToEndMockTest:
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
        nonzero = [
            s for s in cmd_samples if s.timestamp >= start and s.nonzero
        ]
        active_leases = [
            value for ts, value in lease_samples if ts >= start and value
        ]
        if nonzero:
            raise AssertionError("baseline observed non-zero /cmd_vel")
        if active_leases:
            raise AssertionError("baseline observed an active lease")
        return {"cmd_samples": len(cmd_samples), "seconds": time.monotonic() - start}

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

        # Allow late samples to arrive, then verify the stop deadline.
        time.sleep(STOP_DEADLINE_SEC + POST_STOP_OBSERVATION_SEC)
        zero_after = assert_stopped(self.harness, completed_at)

        cmd_samples, raw_samples, joint_samples, lease_samples, _ = self.harness.snapshot()
        active_leases = [value for _ts, value in lease_samples if value]
        if not active_leases:
            raise AssertionError("no command lease appeared during navigation")
        if not any(s.nonzero for s in raw_samples):
            raise AssertionError("no non-zero raw command during navigation")
        if not any(s.nonzero for s in cmd_samples):
            raise AssertionError("no non-zero /cmd_vel during navigation")

        signature = assert_forward_signature(forward_signature(joint_samples))
        planner = assert_planner_evidence(
            read_planner_requests(self.args.planner_log), NAVIGATION_PROMPT
        )
        return {
            "prompt": NAVIGATION_PROMPT,
            "reply": reply,
            "feedback_samples": len(feedback),
            "duration_sec": completed_at - started_at,
            "active_lease_samples": len(active_leases),
            "wheel_signature": signature,
            "zero_cmd_samples_after_deadline": zero_after,
            "planner": planner,
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

        worker = threading.Thread(target=execute, name="e2e-cancel-probe")
        worker_started = time.monotonic()
        worker.start()
        try:
            # Only samples from THIS request count; earlier goals leave
            # non-empty lease entries in the harness history.
            wait_for(
                lambda: any(
                    value
                    for ts, value in self.harness.snapshot()[3]
                    if ts > worker_started and value
                ),
                30.0,
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
            worker.join(30.0)
            if worker.is_alive():
                raise TimeoutError("cancelled request did not finish")

            # Motion must stop: either cancellation propagated downstream, or
            # the revoked lease gates /cmd_vel. Measure from lease emptiness.
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
                "prompt": CANCEL_PROMPT,
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
                "cancel": self.run_cancel(client),
            }
        finally:
            client.cancel_active()
            client.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--planner-log",
        type=str,
        default="/tmp/mock_planner_requests.jsonl",
        help="JSONL request log shared from the planner fixture container",
    )
    parser.add_argument(
        "--fixture-goal-sec",
        type=float,
        default=3.0,
        help="seconds until the downstream fixture completes a goal",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    test = EndToEndMockTest(args)
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
