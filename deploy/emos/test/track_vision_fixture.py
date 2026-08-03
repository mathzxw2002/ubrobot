#!/usr/bin/env python3
"""Standalone deterministic TrackVisionTarget fixture (fake Kompass).

Accepts TrackVisionTarget goals from the NavigateToObject capability server
and publishes a deterministic forward raw_cmd_vel (linear.x = 0.04) so the
full navigation chain (Cortex -> NavigateToObject -> TrackVisionTarget ->
cmd_vel_guard -> /cmd_vel) can be exercised without real Kompass or motion.
Wheels stay off because motor torque remains disabled on the driver.

Usage (inside the emos navigation container):
    python3 /tmp/track_vision_fixture.py [--complete-after 6]
"""

import argparse
import threading
import time

import rclpy
from geometry_msgs.msg import Twist
from kompass_interfaces.action import TrackVisionTarget
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

DOWNSTREAM_ACTION = "/track_vision_target"
RAW_TOPIC = "/navigation/raw_cmd_vel"


def velocity_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=20,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


class DeterministicTrackingFixture(Node):
    """Fake Kompass Action server publishing a forward raw command."""

    def __init__(self, complete_after_sec: float | None = None):
        super().__init__("deterministic_tracking_fixture")
        self._callback_group = ReentrantCallbackGroup()
        self._publisher = self.create_publisher(Twist, RAW_TOPIC, velocity_qos())
        self._raw_enabled = threading.Event()
        self._raw_enabled.set()
        self._shutdown_requested = threading.Event()
        self._complete_after_sec = complete_after_sec
        self._server = ActionServer(
            self,
            TrackVisionTarget,
            DOWNSTREAM_ACTION,
            execute_callback=self._execute,
            goal_callback=lambda _request: GoalResponse.ACCEPT,
            cancel_callback=lambda _goal: CancelResponse.ACCEPT,
            callback_group=self._callback_group,
        )
        self.get_logger().info(
            f"track_vision_fixture serving {DOWNSTREAM_ACTION} "
            f"(complete_after={complete_after_sec}s)"
        )

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
            if (
                self._complete_after_sec is not None
                and elapsed >= self._complete_after_sec
            ):
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complete-after", type=float, default=6.0)
    args = parser.parse_args()

    rclpy.init()
    node = DeterministicTrackingFixture(complete_after_sec=args.complete_after)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
