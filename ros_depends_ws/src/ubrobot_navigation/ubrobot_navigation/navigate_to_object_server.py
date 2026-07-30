"""ROS 2 Action adapter for the controlled NavigateToObject capability."""

import threading
import uuid

import rclpy
from action_msgs.msg import GoalStatus
from kompass_interfaces.action import TrackVisionTarget
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from ubrobot_interfaces.action import NavigateToObject

from .downstream_goal import (
    DownstreamFeedback,
    DownstreamResult,
    GoalBusyError,
    NavigationFeedback,
    NavigationLifecycleCoordinator,
    NavigationStatus,
)


ACTION_NAME = "/ubrobot/navigation/navigate_to_object"
DOWNSTREAM_ACTION_NAME = "/track_vision_target"
LEASE_TOPIC = "/navigation/command_lease"
LEASE_HEARTBEAT_PERIOD_SEC = 0.1


class LeasePublisher:
    """Publish one opaque authority token only while an outer goal owns it."""

    def __init__(self, publisher):
        self._publisher = publisher
        self._lock = threading.Lock()
        self._lease_id = None

    def acquire(self) -> str:
        with self._lock:
            self._lease_id = uuid.uuid4().hex
            self._publish(self._lease_id)
            return self._lease_id

    def heartbeat(self) -> None:
        with self._lock:
            if self._lease_id is not None:
                self._publish(self._lease_id)

    def revoke(self) -> None:
        with self._lock:
            self._lease_id = None
            self._publish("")

    def _publish(self, lease_id: str) -> None:
        message = String()
        message.data = lease_id
        self._publisher.publish(message)


class RosOuterGoalAdapter:
    def __init__(self, goal_handle):
        self._goal_handle = goal_handle

    def is_cancel_requested(self) -> bool:
        return bool(self._goal_handle.is_cancel_requested)

    def publish_feedback(self, feedback: NavigationFeedback) -> None:
        message = NavigateToObject.Feedback()
        message.phase = feedback.phase
        message.distance_error = feedback.distance_error
        message.orientation_error = feedback.orientation_error
        self._goal_handle.publish_feedback(message)


class TrackVisionTargetAdapter:
    """Thread-safe bridge from the pure coordinator to rclpy Action futures."""

    def __init__(
        self,
        *,
        client,
        search_radius: float,
        server_timeout_sec: float,
    ):
        self._client = client
        self._search_radius = search_radius
        self._server_timeout_sec = server_timeout_sec
        self._goal_handle = None
        self._goal_response_event = threading.Event()
        self._done_event = threading.Event()
        self._result = None
        self._error = None
        self._cancel_on_late_acceptance = False

    def start(self, target, timeout_sec, feedback_callback) -> bool:
        wait_timeout = min(self._server_timeout_sec, timeout_sec)
        if not self._client.wait_for_server(timeout_sec=wait_timeout):
            raise TimeoutError("downstream TrackVisionTarget server is unavailable")

        goal = TrackVisionTarget.Goal()
        goal.label = target
        goal.search_radius = self._search_radius
        goal.search_timeout = timeout_sec
        # Kompass explicitly defines zero pixel coordinates as "unknown".
        goal.pose_x = 0
        goal.pose_y = 0

        future = self._client.send_goal_async(
            goal,
            feedback_callback=lambda message: feedback_callback(
                DownstreamFeedback(
                    distance_error=message.feedback.distance_error,
                    orientation_error=message.feedback.orientation_error,
                )
            ),
        )
        future.add_done_callback(self._on_goal_response)
        if not self._goal_response_event.wait(wait_timeout):
            self._cancel_on_late_acceptance = True
            if self._goal_handle is not None and self._goal_handle.accepted:
                self._goal_handle.cancel_goal_async()
            raise TimeoutError("downstream goal response timed out")
        if self._error is not None:
            raise self._error
        return bool(self._goal_handle and self._goal_handle.accepted)

    def is_done(self) -> bool:
        return self._done_event.is_set()

    def result(self) -> DownstreamResult:
        if self._error is not None:
            raise self._error
        if self._result is None:
            raise RuntimeError("downstream result requested before completion")
        succeeded = (
            self._result.status == GoalStatus.STATUS_SUCCEEDED
            and self._result.result.success
        )
        message = (
            "target tracking completed"
            if succeeded
            else "downstream target tracking failed"
        )
        return DownstreamResult(success=succeeded, message=message)

    def cancel(self, timeout_sec: float) -> bool:
        if self._goal_handle is None or not self._goal_handle.accepted:
            return True
        event = threading.Event()
        response = {"accepted": False, "error": None}

        def on_cancel_done(future):
            try:
                response["accepted"] = bool(future.result().goals_canceling)
            except Exception as exc:
                response["error"] = exc
            finally:
                event.set()

        future = self._goal_handle.cancel_goal_async()
        future.add_done_callback(on_cancel_done)
        if not event.wait(timeout_sec):
            return False
        if response["error"] is not None:
            raise response["error"]
        return response["accepted"]

    def _on_goal_response(self, future) -> None:
        try:
            self._goal_handle = future.result()
            if self._goal_handle.accepted:
                if self._cancel_on_late_acceptance:
                    self._goal_handle.cancel_goal_async()
                result_future = self._goal_handle.get_result_async()
                result_future.add_done_callback(self._on_result)
        except Exception as exc:
            self._error = exc
        finally:
            self._goal_response_event.set()

    def _on_result(self, future) -> None:
        try:
            self._result = future.result()
        except Exception as exc:
            self._error = exc
        finally:
            self._done_event.set()


class NavigateToObjectServer(Node):
    def __init__(self):
        super().__init__("navigate_to_object_server")
        self.declare_parameter("search_radius", 1.0)
        self.declare_parameter("downstream_server_timeout_sec", 5.0)
        self.declare_parameter("downstream_cancel_timeout_sec", 2.0)
        self.declare_parameter("lifecycle_poll_period_sec", 0.05)

        self._callback_group = ReentrantCallbackGroup()
        lease_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        publisher = self.create_publisher(String, LEASE_TOPIC, lease_qos)
        self._lease = LeasePublisher(publisher)
        self.create_timer(
            LEASE_HEARTBEAT_PERIOD_SEC,
            self._lease.heartbeat,
            callback_group=self._callback_group,
        )

        self._coordinator = NavigationLifecycleCoordinator(
            poll_period_sec=self.get_parameter(
                "lifecycle_poll_period_sec"
            ).value,
            cancellation_timeout_sec=self.get_parameter(
                "downstream_cancel_timeout_sec"
            ).value,
        )
        self._downstream_client = ActionClient(
            self,
            TrackVisionTarget,
            DOWNSTREAM_ACTION_NAME,
            callback_group=self._callback_group,
        )
        self._reservation_lock = threading.Lock()
        self._pending_reservation = None
        self._action_server = ActionServer(
            self,
            NavigateToObject,
            ACTION_NAME,
            execute_callback=self._execute,
            goal_callback=self._on_goal,
            cancel_callback=self._on_cancel,
            callback_group=self._callback_group,
        )

    def revoke_lease(self) -> None:
        self._lease.revoke()

    def _on_goal(self, request):
        try:
            reservation = self._coordinator.reserve(
                target=request.target,
                timeout_sec=request.timeout_sec,
            )
        except (ValueError, GoalBusyError) as exc:
            self.get_logger().warning(f"navigation goal rejected: {exc}")
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

        result = NavigateToObject.Result()
        if reservation is None:
            result.status = int(NavigationStatus.FAILED)
            result.message = "accepted goal has no lifecycle reservation"
            goal_handle.abort()
            self._lease.revoke()
            return result

        downstream = TrackVisionTargetAdapter(
            client=self._downstream_client,
            search_radius=self.get_parameter("search_radius").value,
            server_timeout_sec=self.get_parameter(
                "downstream_server_timeout_sec"
            ).value,
        )
        outcome = self._coordinator.execute(
            reservation=reservation,
            outer=RosOuterGoalAdapter(goal_handle),
            downstream=downstream,
            lease=self._lease,
        )
        result.status = int(outcome.status)
        result.message = outcome.message
        if outcome.status == NavigationStatus.SUCCEEDED:
            goal_handle.succeed()
        elif outcome.status == NavigationStatus.CANCELLED:
            goal_handle.canceled()
        else:
            goal_handle.abort()
        return result


def main(args=None) -> None:
    rclpy.init(args=args)
    node = NavigateToObjectServer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.revoke_lease()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
