#!/usr/bin/env python3
"""No-motion smoke test for the production Chat UI Cortex Action transport."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import threading
import time

from geometry_msgs.msg import Twist
import rclpy
from rclpy.context import Context
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from cortex_client import create_ros_cortex_client


QUERY_PROMPT = (
    "Report whether orchestration is ready. Do not navigate or call tools."
)
CANCEL_PROMPT = (
    "CANCEL_PROBE: report readiness without navigating or calling tools."
)
ZERO_EPSILON = 1.0e-6


def velocity_qos():
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=20,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def lease_qos():
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=20,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


class MotionObserver:
    def __init__(self):
        self.context = Context()
        rclpy.init(context=self.context)
        self.node = Node("chat_cortex_motion_observer", context=self.context)
        self.executor = MultiThreadedExecutor(num_threads=2, context=self.context)
        self.executor.add_node(self.node)
        self.lock = threading.Lock()
        self.cmd_samples = []
        self.lease_samples = []
        self.node.create_subscription(Twist, "/cmd_vel", self._on_cmd, velocity_qos())
        self.node.create_subscription(
            String,
            "/navigation/command_lease",
            self._on_lease,
            lease_qos(),
        )
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()

    def _on_cmd(self, message):
        with self.lock:
            self.cmd_samples.append(
                (
                    time.monotonic(),
                    float(message.linear.x),
                    float(message.linear.y),
                    float(message.angular.z),
                )
            )

    def _on_lease(self, message):
        with self.lock:
            self.lease_samples.append((time.monotonic(), str(message.data)))

    def snapshot(self):
        with self.lock:
            return list(self.cmd_samples), list(self.lease_samples)

    def wait_for_guard(self, timeout_sec=5.0):
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if self.snapshot()[0]:
                return
            time.sleep(0.02)
        raise TimeoutError("no /cmd_vel guard samples discovered")

    def close(self):
        self.executor.shutdown(timeout_sec=2.0)
        self.thread.join(2.0)
        self.node.destroy_node()
        rclpy.shutdown(context=self.context)


def assert_no_motion(observer):
    cmd_samples, lease_samples = observer.snapshot()
    if not cmd_samples:
        raise AssertionError("no /cmd_vel samples were observed")
    nonzero = [
        sample
        for sample in cmd_samples
        if max(abs(value) for value in sample[1:]) > ZERO_EPSILON
    ]
    active_leases = [value for _timestamp, value in lease_samples if value]
    if nonzero:
        raise AssertionError(f"observed {len(nonzero)} non-zero /cmd_vel samples")
    if active_leases:
        raise AssertionError("non-empty navigation lease observed")
    return {
        "cmd_samples": len(cmd_samples),
        "cmd_nonzero_samples": len(nonzero),
        "lease_samples": len(lease_samples),
        "active_lease_samples": len(active_leases),
    }


def run_query(client, observer):
    feedback = []
    started_at = time.monotonic()
    cancel_result = None
    try:
        reply = client.execute(QUERY_PROMPT, on_feedback=feedback.append)
        if not feedback:
            raise AssertionError("Cortex returned no Action feedback")
        if not reply:
            raise AssertionError("Cortex returned no final text")
        time.sleep(0.3)
        return {
            "prompt": QUERY_PROMPT,
            "reply": reply,
            "feedback": feedback,
            "duration_sec": time.monotonic() - started_at,
            **assert_no_motion(observer),
        }
    finally:
        cancel_result = client.cancel_active()
        # A completed query normally has no active goal; the call is still
        # mandatory so exceptional paths cannot leave a goal behind.
        if cancel_result not in (True, False):
            raise AssertionError("unexpected cancellation result")


def run_cancel_probe(client, observer):
    feedback = []
    outcome = {"reply": None, "error": None}

    def execute():
        try:
            outcome["reply"] = client.execute(
                CANCEL_PROMPT,
                on_feedback=feedback.append,
            )
        except Exception as exc:
            outcome["error"] = f"{type(exc).__name__}: {exc}"

    worker = threading.Thread(target=execute, name="cortex-cancel-probe")
    worker.start()
    try:
        deadline = time.monotonic() + 5.0
        while not feedback and time.monotonic() < deadline:
            time.sleep(0.02)
        if not feedback:
            raise TimeoutError("cancel probe received no initial feedback")

        cancel_started = time.monotonic()
        acknowledged = client.cancel_active()
        cancel_latency = time.monotonic() - cancel_started
        if not acknowledged:
            raise AssertionError("Cortex cancellation was not acknowledged")
        if not math.isfinite(cancel_latency) or cancel_latency > 2.0:
            raise AssertionError(
                f"cancellation acknowledgement took {cancel_latency:.3f}s"
            )

        worker.join(10.0)
        if worker.is_alive():
            raise TimeoutError("cancelled Cortex request did not finish")
        time.sleep(0.3)
        return {
            "prompt": CANCEL_PROMPT,
            "feedback": feedback,
            "cancel_acknowledged": acknowledged,
            "cancel_latency_sec": cancel_latency,
            "request_reply": outcome["reply"],
            "request_error": outcome["error"],
            **assert_no_motion(observer),
        }
    finally:
        client.cancel_active()
        worker.join(10.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    observer = MotionObserver()
    client = create_ros_cortex_client()
    try:
        observer.wait_for_guard()
        result = {
            "passed": True,
            "query": run_query(client, observer),
            "cancel_probe": run_cancel_probe(client, observer),
        }
        rendered = json.dumps(result, indent=2, sort_keys=True)
        print(rendered, flush=True)
        if args.output:
            args.output.write_text(rendered + "\n", encoding="utf-8")
        return 0
    except Exception as exc:
        result = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        rendered = json.dumps(result, indent=2, sort_keys=True)
        print(rendered, flush=True)
        if args.output:
            args.output.write_text(rendered + "\n", encoding="utf-8")
        return 1
    finally:
        client.cancel_active()
        client.close()
        observer.close()


if __name__ == "__main__":
    raise SystemExit(main())
