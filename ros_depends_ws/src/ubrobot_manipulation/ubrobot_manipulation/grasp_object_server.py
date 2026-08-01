"""ROS 2 Action adapter for the controlled GraspObject capability.

Skeleton: the Action server, motion-authority wiring, and lifecycle are
complete; the platform executor binding is intentionally absent until the
arm machines return (see docs/plans/2026-07-31-cortex-grasp-capability.md).
Goals received without an executor binding fail fast with a clear message
instead of hanging.

Platform selection is via the ``UBROBOT_GRASP_PLATFORM`` environment
variable (``piper_station`` | ``go2_piper``); unknown or missing profiles
abort startup — a misconfigured grasp server must never run.
"""

import os
import threading
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from ubrobot_interfaces.action import GraspObject

from .authority import AuthorityTracker
from .lifecycle import (
    GoalBusyError,
    GraspFeedback,
    GraspLifecycleCoordinator,
    GraspStatus,
)
from .policy import get_platform_profile


ACTION_NAME = "/ubrobot/manipulation/grasp_object"
LEASE_TOPIC = "/navigation/command_lease"
CMD_VEL_TOPIC = "/cmd_vel"
PLATFORM_ENV = "UBROBOT_GRASP_PLATFORM"


class RosOuterGoalAdapter:
    def __init__(self, goal_handle):
        self._goal_handle = goal_handle

    def is_cancel_requested(self) -> bool:
        return bool(self._goal_handle.is_cancel_requested)

    def publish_feedback(self, feedback: GraspFeedback) -> None:
        message = GraspObject.Feedback()
        message.phase = feedback.phase
        message.target_distance_m = feedback.target_distance_m
        message.progress = feedback.progress
        self._goal_handle.publish_feedback(message)


class RosMotionAuthorityAdapter:
    """Feeds ROS samples into the fail-closed AuthorityTracker."""

    def __init__(self, node: Node, tracker: AuthorityTracker):
        self._tracker = tracker
        self._lock = threading.Lock()
        lease_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        velocity_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        node.create_subscription(
            String, LEASE_TOPIC, self._on_lease, lease_qos
        )
        node.create_subscription(
            Twist, CMD_VEL_TOPIC, self._on_cmd_vel, velocity_qos
        )

    def _on_lease(self, message: String) -> None:
        with self._lock:
            self._tracker.note_lease(str(message.data), time.monotonic())

    def _on_cmd_vel(self, message: Twist) -> None:
        with self._lock:
            self._tracker.note_cmd_vel(
                message.linear.x,
                message.linear.y,
                message.angular.z,
                time.monotonic(),
            )

    def navigation_lease_active(self) -> bool:
        with self._lock:
            return self._tracker.navigation_lease_active(time.monotonic())

    def base_is_stationary(self) -> bool:
        with self._lock:
            return self._tracker.base_is_stationary(time.monotonic())


def build_executor(node: Node, profile):
    """Bind the platform grasp executor. Not implemented offline.

    The first binding (``piper_graspnet`` for ``piper_station``) arrives
    with the executor milestone; raising here makes an unbound deployment
    fail fast and loud instead of accepting goals it cannot serve.
    """
    raise NotImplementedError(
        f"no grasp executor binding implemented for profile "
        f"'{profile.name}' (executor kind '{profile.executor_kind}'); "
        "see docs/plans/2026-07-31-cortex-grasp-capability.md deferred steps"
    )


class GraspObjectServer(Node):
    def __init__(self):
        super().__init__("grasp_object_server")
        self.declare_parameter("lifecycle_poll_period_sec", 0.05)
        self.declare_parameter("executor_cancel_timeout_sec", 2.0)
        self.declare_parameter("lease_max_age_sec", 0.5)
        self.declare_parameter("cmd_vel_window_sec", 0.5)

        platform_name = os.environ.get(PLATFORM_ENV, "").strip()
        if not platform_name:
            raise RuntimeError(
                f"{PLATFORM_ENV} must name a grasp platform profile "
                "(piper_station | go2_piper)"
            )
        # Raises ValueError for unknown profiles: fail fast at startup.
        self._profile = get_platform_profile(platform_name)
        self.get_logger().info(
            f"grasp capability bound to platform '{self._profile.name}'"
        )

        self._callback_group = ReentrantCallbackGroup()
        tracker = AuthorityTracker(
            lease_max_age_sec=self.get_parameter("lease_max_age_sec").value,
            cmd_vel_window_sec=self.get_parameter("cmd_vel_window_sec").value,
        )
        self._authority = RosMotionAuthorityAdapter(self, tracker)
        self._coordinator = GraspLifecycleCoordinator(
            profile=self._profile,
            poll_period_sec=self.get_parameter("lifecycle_poll_period_sec").value,
            cancellation_timeout_sec=self.get_parameter(
                "executor_cancel_timeout_sec"
            ).value,
        )

        self._reservation_lock = threading.Lock()
        self._pending_reservation = None
        self._action_server = ActionServer(
            self,
            GraspObject,
            ACTION_NAME,
            execute_callback=self._execute,
            goal_callback=self._on_goal,
            cancel_callback=self._on_cancel,
            callback_group=self._callback_group,
        )

    def _on_goal(self, request):
        try:
            reservation = self._coordinator.reserve(
                target=request.target,
                timeout_sec=request.timeout_sec,
            )
        except (ValueError, GoalBusyError) as exc:
            self.get_logger().warning(f"grasp goal rejected: {exc}")
            return GoalResponse.REJECT

        with self._reservation_lock:
            self._pending_reservation = reservation
        return GoalResponse.ACCEPT

    def _on_cancel(self, _goal_handle):
        return CancelResponse.ACCEPT

    def _execute(self, goal_handle):
        with self._reservation_lock:
            reservation = self._pending_reservation
            self._pending_reservation = None

        result = GraspObject.Result()
        if reservation is None:
            result.status = int(GraspStatus.FAILED)
            result.message = "accepted goal has no lifecycle reservation"
            goal_handle.abort()
            return result

        try:
            executor = build_executor(self, self._profile)
        except NotImplementedError as exc:
            result.status = int(GraspStatus.FAILED)
            result.message = str(exc)
            self._coordinator.abandon(reservation)
            goal_handle.abort()
            return result

        outcome = self._coordinator.execute(
            reservation=reservation,
            outer=RosOuterGoalAdapter(goal_handle),
            executor=executor,
            authority=self._authority,
        )
        result.status = int(outcome.status)
        result.message = outcome.message
        if outcome.status == GraspStatus.SUCCEEDED:
            goal_handle.succeed()
        elif outcome.status == GraspStatus.CANCELLED:
            goal_handle.canceled()
        else:
            goal_handle.abort()
        return result


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GraspObjectServer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
