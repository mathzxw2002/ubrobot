#!/usr/bin/env python3
"""Cross-container Cortex navigation mock and failure-injection test client.

The deterministic TrackVisionTarget fixture replaces camera/VLM inference while
retaining the production outer Action, command lease, velocity guard, and
LeKiwi mock-driver data path.  Every goal owned by the harness is cancelled in
``finally``; the orphan-client case uses SIGKILL deliberately to prove that a
process which cannot run its cleanup still loses command authority on timeout.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import rclpy
from geometry_msgs.msg import Twist
from kompass_interfaces.action import TrackVisionTarget
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from ubrobot_interfaces.action import NavigateToObject


OUTER_ACTION = "/ubrobot/navigation/navigate_to_object"
DOWNSTREAM_ACTION = "/track_vision_target"
RAW_TOPIC = "/navigation/raw_cmd_vel"
LEASE_TOPIC = "/navigation/command_lease"
CMD_TOPIC = "/cmd_vel"
JOINT_TOPIC = "/joint_states"
ZERO_EPSILON = 1.0e-4
STOP_DEADLINE_SEC = 0.3
POST_STOP_OBSERVATION_SEC = 0.6


def velocity_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=20,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def lease_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=20,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


@dataclass(frozen=True)
class VelocitySample:
    timestamp: float
    x: float
    y: float
    z: float

    @property
    def nonzero(self) -> bool:
        return max(abs(self.x), abs(self.y), abs(self.z)) > ZERO_EPSILON


@dataclass(frozen=True)
class JointSample:
    timestamp: float
    names: tuple[str, ...]
    velocities: tuple[float, ...]

    @property
    def nonzero(self) -> bool:
        return any(abs(value) > ZERO_EPSILON for value in self.velocities)


class DeterministicTrackingFixture(Node):
    """Fake Kompass Action server driven by a recorded forward-command fixture."""

    def __init__(self):
        super().__init__("deterministic_tracking_fixture")
        self._callback_group = ReentrantCallbackGroup()
        self._publisher = self.create_publisher(Twist, RAW_TOPIC, velocity_qos())
        self._raw_enabled = threading.Event()
        self._raw_enabled.set()
        self._shutdown_requested = threading.Event()
        self._complete_after_sec: float | None = None
        self._server = ActionServer(
            self,
            TrackVisionTarget,
            DOWNSTREAM_ACTION,
            execute_callback=self._execute,
            goal_callback=lambda _request: GoalResponse.ACCEPT,
            cancel_callback=lambda _goal: CancelResponse.ACCEPT,
            callback_group=self._callback_group,
        )

    def configure(self, *, complete_after_sec: float | None) -> None:
        self._complete_after_sec = complete_after_sec
        self._raw_enabled.set()
        self._shutdown_requested.clear()

    def stop_raw_publication(self) -> None:
        self._raw_enabled.clear()

    def stop_active_goals(self) -> None:
        self._raw_enabled.clear()
        self._shutdown_requested.set()

    def _execute(self, goal_handle):
        started_at = time.monotonic()
        result = TrackVisionTarget.Result()
        while rclpy.ok():
            if self._shutdown_requested.is_set():
                goal_handle.abort()
                result.success = False
                result.tracked_duration = time.monotonic() - started_at
                return result
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.success = False
                result.tracked_duration = time.monotonic() - started_at
                return result

            elapsed = time.monotonic() - started_at
            if self._complete_after_sec is not None and elapsed >= self._complete_after_sec:
                goal_handle.succeed()
                result.success = True
                result.tracked_duration = elapsed
                return result

            if self._raw_enabled.is_set():
                command = Twist()
                command.linear.x = 0.04
                self._publisher.publish(command)

            feedback = TrackVisionTarget.Feedback()
            feedback.distance_error = max(0.0, 1.0 - elapsed * 0.1)
            feedback.orientation_error = 0.0
            goal_handle.publish_feedback(feedback)
            time.sleep(0.05)

        result.success = False
        result.tracked_duration = time.monotonic() - started_at
        return result


class HarnessNode(Node):
    def __init__(self):
        super().__init__("cortex_navigation_mock_harness")
        self._lock = threading.Lock()
        self.cmd_samples: list[VelocitySample] = []
        self.raw_samples: list[VelocitySample] = []
        self.joint_samples: list[JointSample] = []
        self.lease_samples: list[tuple[float, str]] = []
        self.feedback: list[dict[str, float | str]] = []
        self.outer_client = ActionClient(self, NavigateToObject, OUTER_ACTION)
        self.downstream_client = ActionClient(self, TrackVisionTarget, DOWNSTREAM_ACTION)
        self.create_subscription(Twist, CMD_TOPIC, self._on_cmd, velocity_qos())
        self.create_subscription(Twist, RAW_TOPIC, self._on_raw, velocity_qos())
        self.create_subscription(String, LEASE_TOPIC, self._on_lease, lease_qos())
        self.create_subscription(JointState, JOINT_TOPIC, self._on_joints, velocity_qos())

    def _on_cmd(self, message: Twist) -> None:
        sample = VelocitySample(
            time.monotonic(), message.linear.x, message.linear.y, message.angular.z
        )
        with self._lock:
            self.cmd_samples.append(sample)

    def _on_raw(self, message: Twist) -> None:
        sample = VelocitySample(
            time.monotonic(), message.linear.x, message.linear.y, message.angular.z
        )
        with self._lock:
            self.raw_samples.append(sample)

    def _on_lease(self, message: String) -> None:
        with self._lock:
            self.lease_samples.append((time.monotonic(), message.data))

    def _on_joints(self, message: JointState) -> None:
        with self._lock:
            self.joint_samples.append(
                JointSample(
                    time.monotonic(),
                    tuple(message.name),
                    tuple(float(value) for value in message.velocity),
                )
            )

    def on_feedback(self, message) -> None:
        feedback = message.feedback
        with self._lock:
            self.feedback.append(
                {
                    "timestamp": time.monotonic(),
                    "phase": feedback.phase,
                    "distance_error": feedback.distance_error,
                    "orientation_error": feedback.orientation_error,
                }
            )

    def snapshot(self):
        with self._lock:
            return (
                list(self.cmd_samples),
                list(self.raw_samples),
                list(self.joint_samples),
                list(self.lease_samples),
                list(self.feedback),
            )


class MockNavigationTest:
    def __init__(self, args):
        self.args = args
        self.fixture = DeterministicTrackingFixture()
        self.harness = HarnessNode()
        self.executor = MultiThreadedExecutor(num_threads=8)
        self.executor.add_node(self.fixture)
        self.executor.add_node(self.harness)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()
        self.metrics: dict[str, object] = {"scenario": args.scenario}

    def close(self) -> None:
        self.fixture.stop_active_goals()
        time.sleep(0.1)
        self.executor.shutdown(timeout_sec=2.0)
        self.fixture.destroy_node()
        self.harness.destroy_node()
        self.spin_thread.join(timeout=2.0)

    def run(self) -> dict[str, object]:
        scenario = self.args.scenario
        if scenario == "baseline":
            return self._baseline()
        if scenario == "goal":
            return self._goal_success()
        if scenario == "cancel":
            return self._cancel()
        if scenario == "timeout":
            return self._timeout()
        if scenario == "orphan_client":
            return self._orphan_client()
        if scenario == "capability_loss":
            return self._capability_loss()
        if scenario == "raw_loss":
            return self._raw_loss()
        if scenario == "stale_downstream":
            return self._stale_downstream()
        if scenario == "driver_restart":
            return self._driver_restart()
        raise AssertionError(f"unsupported scenario: {scenario}")

    def _baseline(self) -> dict[str, object]:
        self._wait_for_topics()
        started = time.monotonic()
        time.sleep(self.args.duration)
        self._assert_all_zero(since=started)
        return self._finish(duration_sec=time.monotonic() - started)

    def _goal_success(self) -> dict[str, object]:
        self.fixture.configure(complete_after_sec=2.0)
        goal_handle = None
        try:
            goal_handle, result_future, accepted_at = self._send_outer(timeout_sec=5.0)
            self._wait_for_motion()
            wrapped = wait_future(result_future, 8.0, "normal navigation result")
            completed_at = time.monotonic()
            assert wrapped.status == 4, f"outer goal did not succeed: status={wrapped.status}"
            assert wrapped.result.status == NavigateToObject.Result.SUCCEEDED
            time.sleep(POST_STOP_OBSERVATION_SEC)
            self._assert_feedback_and_wheel_signature()
            self._assert_motion_only_with_lease()
            stop_latency = self._assert_stopped_by(completed_at, STOP_DEADLINE_SEC)
            return self._finish(
                accepted_at=accepted_at,
                result_status=int(wrapped.result.status),
                stop_latency_sec=stop_latency,
            )
        finally:
            safe_cancel(goal_handle)

    def _cancel(self) -> dict[str, object]:
        self.fixture.configure(complete_after_sec=None)
        goal_handle = None
        try:
            goal_handle, result_future, _ = self._send_outer(timeout_sec=10.0)
            self._wait_for_motion()
            fault_at = time.monotonic()
            cancel_response = wait_future(
                goal_handle.cancel_goal_async(), 3.0, "outer cancellation"
            )
            assert cancel_response.goals_canceling, "outer cancellation was rejected"
            wrapped = wait_future(result_future, 5.0, "cancelled navigation result")
            assert wrapped.result.status == NavigateToObject.Result.CANCELLED
            time.sleep(POST_STOP_OBSERVATION_SEC)
            return self._finish(
                result_status=int(wrapped.result.status),
                stop_latency_sec=self._assert_stopped_by(fault_at, STOP_DEADLINE_SEC),
            )
        finally:
            safe_cancel(goal_handle)

    def _timeout(self) -> dict[str, object]:
        timeout_sec = 1.2
        self.fixture.configure(complete_after_sec=None)
        goal_handle = None
        try:
            goal_handle, result_future, _ = self._send_outer(timeout_sec=timeout_sec)
            self._wait_for_motion()
            lease_started = self._first_active_lease_time()
            wrapped = wait_future(result_future, 5.0, "timed-out navigation result")
            assert wrapped.result.status == NavigateToObject.Result.TIMED_OUT
            deadline = lease_started + timeout_sec
            time.sleep(POST_STOP_OBSERVATION_SEC)
            return self._finish(
                result_status=int(wrapped.result.status),
                stop_latency_sec=self._assert_stopped_by(deadline, STOP_DEADLINE_SEC),
            )
        finally:
            safe_cancel(goal_handle)

    def _orphan_client(self) -> dict[str, object]:
        timeout_sec = 3.0
        self.fixture.configure(complete_after_sec=None)
        self._wait_for_topics()
        discovered = wait_until(
            lambda: self.fixture._publisher.get_subscription_count() >= 2,
            5.0,
        )
        assert discovered, "raw-command fixture did not discover guard and monitor"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--hold-goal",
            "--goal-timeout",
            str(timeout_sec),
        ]
        child = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            accepted_line = read_line_with_timeout(child, 8.0)
            assert accepted_line == "GOAL_ACCEPTED", accepted_line
            self._wait_for_motion()
            lease_started = self._first_active_lease_time()
            child.kill()
            child.wait(timeout=3.0)
            fault_at = time.monotonic()
            expected_timeout = lease_started + timeout_sec
            time.sleep(max(0.0, expected_timeout - time.monotonic()) + POST_STOP_OBSERVATION_SEC)
            return self._finish(
                client_exit_code=child.returncode,
                client_killed_at=fault_at,
                stop_latency_sec=self._assert_stopped_by(
                    expected_timeout, STOP_DEADLINE_SEC
                ),
            )
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=3.0)

    def _capability_loss(self) -> dict[str, object]:
        self.fixture.configure(complete_after_sec=None)
        goal_handle = None
        try:
            goal_handle, _result_future, _ = self._send_outer(timeout_sec=10.0)
            self._wait_for_motion()
            fault_at = time.monotonic()
            killed = kill_process_matching("navigate_to_object_server")
            assert killed, "navigate_to_object_server process was not found"
            time.sleep(STOP_DEADLINE_SEC + POST_STOP_OBSERVATION_SEC)
            return self._finish(
                killed_pids=killed,
                stop_latency_sec=self._assert_stopped_by(fault_at, STOP_DEADLINE_SEC),
            )
        finally:
            safe_cancel(goal_handle)

    def _raw_loss(self) -> dict[str, object]:
        self.fixture.configure(complete_after_sec=None)
        goal_handle = None
        try:
            goal_handle, _result_future, _ = self._send_outer(timeout_sec=10.0)
            self._wait_for_motion()
            fault_at = time.monotonic()
            self.fixture.stop_raw_publication()
            time.sleep(STOP_DEADLINE_SEC + POST_STOP_OBSERVATION_SEC)
            return self._finish(
                stop_latency_sec=self._assert_stopped_by(fault_at, STOP_DEADLINE_SEC),
            )
        finally:
            safe_cancel(goal_handle)

    def _stale_downstream(self) -> dict[str, object]:
        self.fixture.configure(complete_after_sec=None)
        goal_handle = None
        try:
            self._wait_for_topics()
            assert self.harness.downstream_client.wait_for_server(5.0)
            goal = TrackVisionTarget.Goal()
            goal.label = "recorded-chair"
            goal.search_radius = 1.0
            goal.search_timeout = 10.0
            future = self.harness.downstream_client.send_goal_async(goal)
            goal_handle = wait_future(future, 3.0, "direct downstream goal")
            assert goal_handle.accepted
            raw_started = wait_until(
                lambda: any(sample.nonzero for sample in self.harness.snapshot()[1]),
                3.0,
            )
            assert raw_started, "stale downstream goal emitted no raw command"
            started = time.monotonic()
            time.sleep(1.0)
            self._assert_all_zero(since=started)
            assert not any(value for _, value in self.harness.snapshot()[3]), (
                "a downstream-only goal unexpectedly acquired an outer lease"
            )
            return self._finish(raw_without_lease=True)
        finally:
            safe_cancel(goal_handle)

    def _driver_restart(self) -> dict[str, object]:
        self._wait_for_topics()
        started = time.monotonic()
        time.sleep(self.args.duration)
        self._assert_all_zero(since=started)
        joints = [s for s in self.harness.snapshot()[2] if s.timestamp >= started]
        assert joints, "no joint states observed around mock-driver restart"
        gaps = [b.timestamp - a.timestamp for a, b in zip(joints, joints[1:])]
        max_gap = max(gaps, default=0.0)
        assert max_gap >= 0.25, f"mock-driver restart gap not observed: {max_gap:.3f}s"
        assert joints[-1].timestamp - started >= self.args.duration - 1.0, (
            "joint states did not recover after mock-driver restart"
        )
        return self._finish(max_joint_state_gap_sec=max_gap)

    def _send_outer(self, *, timeout_sec: float):
        self._wait_for_topics()
        discovered = wait_until(
            lambda: self.fixture._publisher.get_subscription_count() >= 2,
            5.0,
        )
        assert discovered, "raw-command fixture did not discover guard and monitor"
        assert self.harness.outer_client.wait_for_server(5.0), "outer Action unavailable"
        goal = NavigateToObject.Goal()
        goal.target = "recorded-chair"
        goal.timeout_sec = timeout_sec
        future = self.harness.outer_client.send_goal_async(
            goal, feedback_callback=self.harness.on_feedback
        )
        goal_handle = wait_future(future, 3.0, "outer goal response")
        assert goal_handle.accepted, "outer goal rejected"
        return goal_handle, goal_handle.get_result_async(), time.monotonic()

    def _wait_for_topics(self) -> None:
        ok = wait_until(
            lambda: bool(self.harness.snapshot()[0]) and bool(self.harness.snapshot()[2]),
            8.0,
        )
        assert ok, "guard or mock-driver telemetry did not become available"

    def _wait_for_motion(self) -> None:
        ok = wait_until(
            lambda: any(sample.nonzero for sample in self.harness.snapshot()[0]),
            5.0,
        )
        assert ok, "guarded /cmd_vel never became non-zero"

    def _first_active_lease_time(self) -> float:
        active = [stamp for stamp, value in self.harness.snapshot()[3] if value]
        assert active, "no active command lease observed"
        return active[0]

    def _assert_all_zero(self, *, since: float) -> None:
        cmd, _raw, joints, leases, _feedback = self.harness.snapshot()
        cmd = [sample for sample in cmd if sample.timestamp >= since]
        joints = [sample for sample in joints if sample.timestamp >= since]
        assert cmd, "no /cmd_vel samples observed"
        assert joints, "no /joint_states samples observed"
        assert not any(sample.nonzero for sample in cmd), "non-zero /cmd_vel observed"
        assert not any(sample.nonzero for sample in joints), "non-zero mock wheel velocity observed"
        assert not any(value for stamp, value in leases if stamp >= since), (
            "active command lease observed during no-goal interval"
        )

    def _assert_feedback_and_wheel_signature(self) -> None:
        _cmd, _raw, joints, _leases, feedback = self.harness.snapshot()
        assert feedback, "outer Action feedback was not observed"
        assert any(self._forward_wheel_signature(sample) for sample in joints), (
            "expected omnidirectional forward wheel signature was not observed"
        )

    @staticmethod
    def _forward_wheel_signature(sample: JointSample) -> bool:
        if len(sample.velocities) < 3:
            return False
        values = sample.velocities[:3]
        near_zero = sum(abs(value) <= 0.02 for value in values)
        positive = any(value > 0.02 for value in values)
        negative = any(value < -0.02 for value in values)
        return near_zero >= 1 and positive and negative

    def _assert_motion_only_with_lease(self) -> None:
        cmd, _raw, joints, leases, _feedback = self.harness.snapshot()
        active_ranges = lease_ranges(leases)
        assert active_ranges, "no lease interval recorded"
        for sample in cmd:
            if sample.nonzero:
                assert in_ranges(sample.timestamp, active_ranges), (
                    "guarded command escaped the lease interval"
                )
        for sample in joints:
            if sample.nonzero:
                assert in_ranges(sample.timestamp, active_ranges, grace_sec=0.3), (
                    "mock wheel motion escaped the lease interval"
                )

    def _assert_stopped_by(self, fault_at: float, deadline_sec: float) -> float:
        cmd = [sample for sample in self.harness.snapshot()[0] if sample.timestamp >= fault_at]
        assert cmd, "no guarded samples observed after injected fault"
        stable_zero = None
        for index, sample in enumerate(cmd):
            if not sample.nonzero and not any(later.nonzero for later in cmd[index:]):
                stable_zero = sample.timestamp
                break
        assert stable_zero is not None, "guarded command did not remain zero"
        latency = stable_zero - fault_at
        assert latency <= deadline_sec, (
            f"guarded stop latency {latency:.3f}s exceeds {deadline_sec:.3f}s"
        )
        return latency

    def _finish(self, **extra) -> dict[str, object]:
        cmd, raw, joints, leases, feedback = self.harness.snapshot()
        peak_joint_sample = max(
            joints,
            key=lambda sample: max((abs(value) for value in sample.velocities), default=0.0),
            default=None,
        )
        result = {
            "scenario": self.args.scenario,
            "passed": True,
            "cmd_samples": len(cmd),
            "cmd_nonzero_samples": sum(sample.nonzero for sample in cmd),
            "raw_samples": len(raw),
            "raw_nonzero_samples": sum(sample.nonzero for sample in raw),
            "joint_samples": len(joints),
            "joint_nonzero_samples": sum(sample.nonzero for sample in joints),
            "lease_samples": len(leases),
            "active_lease_samples": sum(bool(value) for _, value in leases),
            "feedback_samples": len(feedback),
            "peak_joint_names": list(peak_joint_sample.names) if peak_joint_sample else [],
            "peak_joint_velocities": (
                list(peak_joint_sample.velocities) if peak_joint_sample else []
            ),
        }
        result.update(extra)
        return result


def hold_goal(timeout_sec: float) -> int:
    """Cortex-client surrogate whose SIGKILL simulates an unclean client death."""
    rclpy.init()
    node = Node("cortex_client_failure_fixture")
    client = ActionClient(node, NavigateToObject, OUTER_ACTION)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    goal_handle = None
    try:
        assert client.wait_for_server(5.0)
        goal = NavigateToObject.Goal()
        goal.target = "recorded-chair"
        goal.timeout_sec = timeout_sec
        goal_handle = wait_future(client.send_goal_async(goal), 3.0, "orphan goal")
        assert goal_handle.accepted
        print("GOAL_ACCEPTED", flush=True)
        while rclpy.ok():
            time.sleep(0.1)
    finally:
        safe_cancel(goal_handle)
        executor.shutdown(timeout_sec=1.0)
        node.destroy_node()
        rclpy.shutdown()
    return 0


def wait_future(future, timeout_sec: float, description: str):
    deadline = time.monotonic() + timeout_sec
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not future.done():
        raise TimeoutError(f"timed out waiting for {description}")
    exception = future.exception()
    if exception is not None:
        raise exception
    return future.result()


def safe_cancel(goal_handle) -> None:
    if goal_handle is None:
        return
    try:
        future = goal_handle.cancel_goal_async()
        response = wait_future(future, 2.0, "finally cancellation")
        if response.goals_canceling:
            wait_future(
                goal_handle.get_result_async(),
                2.0,
                "finally cancellation acknowledgement",
            )
    except Exception as exc:
        print(f"cleanup cancellation incomplete: {exc}", file=sys.stderr, flush=True)


def wait_until(predicate, timeout_sec: float) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return bool(predicate())


def read_line_with_timeout(process: subprocess.Popen, timeout_sec: float) -> str:
    holder: list[str] = []

    def reader():
        if process.stdout is not None:
            holder.append(process.stdout.readline().strip())

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    thread.join(timeout_sec)
    if thread.is_alive():
        raise TimeoutError("orphan-client sender did not accept its goal")
    return holder[0] if holder else ""


def kill_process_matching(pattern: str) -> list[int]:
    own_pid = os.getpid()
    killed: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (OSError, UnicodeDecodeError):
            continue
        if pattern in command:
            os.kill(int(entry.name), signal.SIGTERM)
            killed.append(int(entry.name))
    return killed


def lease_ranges(samples: list[tuple[float, str]]) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    started: float | None = None
    for timestamp, value in samples:
        if value and started is None:
            started = timestamp
        elif not value and started is not None:
            ranges.append((started, timestamp))
            started = None
    if started is not None:
        ranges.append((started, math.inf))
    return ranges


def in_ranges(
    timestamp: float,
    ranges: list[tuple[float, float]],
    *,
    grace_sec: float = 0.0,
) -> bool:
    return any(start <= timestamp <= end + grace_sec for start, end in ranges)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=(
            "baseline",
            "goal",
            "cancel",
            "timeout",
            "orphan_client",
            "capability_loss",
            "raw_loss",
            "stale_downstream",
            "driver_restart",
        ),
        default="baseline",
    )
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hold-goal", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--goal-timeout", type=float, default=1.2, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.hold_goal:
        return hold_goal(args.goal_timeout)

    rclpy.init()
    test = MockNavigationTest(args)
    try:
        result = test.run()
        rendered = json.dumps(result, indent=2, sort_keys=True)
        print(rendered, flush=True)
        if args.output:
            args.output.write_text(rendered + "\n", encoding="utf-8")
        return 0
    except Exception as exc:
        result = test._finish()
        result.update({"passed": False, "error": str(exc)})
        rendered = json.dumps(result, indent=2, sort_keys=True)
        print(rendered, file=sys.stderr, flush=True)
        if args.output:
            args.output.write_text(rendered + "\n", encoding="utf-8")
        return 1
    finally:
        test.close()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
