"""Short-lived command lease guard for navigation velocity output."""

from dataclasses import dataclass
import math
from typing import Callable, Optional

from .policy import (
    COMMAND_FRESHNESS_SEC,
    command_is_fresh,
    lease_is_fresh,
    sanitize_twist,
)


ZERO_TWIST = (0.0, 0.0, 0.0)
GUARD_PERIOD_SEC = 0.05
REVOCATION_ZERO_SAMPLES = 3


@dataclass(frozen=True)
class GuardOutput:
    twist: tuple[float, float, float]
    error: Optional[str] = None


class CmdVelGuardState:
    """DDS-independent state machine; all time comes from the injected clock."""

    def __init__(
        self,
        *,
        clock: Callable[[], float],
        lease_timeout_sec: float = COMMAND_FRESHNESS_SEC,
        raw_command_timeout_sec: float = COMMAND_FRESHNESS_SEC,
    ):
        self._clock = clock
        self._lease_timeout_sec = _positive_finite(
            "lease_timeout_sec",
            lease_timeout_sec,
        )
        self._raw_command_timeout_sec = _positive_finite(
            "raw_command_timeout_sec",
            raw_command_timeout_sec,
        )
        self._lease_id: Optional[str] = None
        self._lease_time: Optional[float] = None
        self._raw_twist: Optional[tuple[float, float, float]] = None
        self._raw_time: Optional[float] = None
        self._raw_lease_id: Optional[str] = None
        self._forced_zero_ticks = 0

    @property
    def lease_id(self) -> Optional[str]:
        return self._lease_id

    def on_lease(self, lease_id: str) -> None:
        """Renew a matching lease, acquire a new lease, or revoke with empty text."""
        normalized = lease_id.strip()
        now = self._clock()

        if not normalized:
            self._lease_id = None
            self._lease_time = None
            self._invalidate_raw_command()
            self._forced_zero_ticks = max(
                self._forced_zero_ticks,
                REVOCATION_ZERO_SAMPLES,
            )
            return

        if normalized != self._lease_id:
            self._invalidate_raw_command()
        self._lease_id = normalized
        self._lease_time = now

    def on_raw_command(
        self,
        linear_x: float,
        linear_y: float,
        angular_z: float,
    ) -> None:
        """Associate a raw command with the lease active when it arrived."""
        self._raw_twist = (linear_x, linear_y, angular_z)
        self._raw_time = self._clock()
        self._raw_lease_id = self._lease_id

    def tick(self) -> GuardOutput:
        """Produce the command for the next fixed-rate `/cmd_vel` publication."""
        if self._forced_zero_ticks:
            self._forced_zero_ticks -= 1
            return GuardOutput(ZERO_TWIST)

        if self._raw_twist is None:
            return GuardOutput(ZERO_TWIST)

        if not self._velocity_is_finite():
            return GuardOutput(ZERO_TWIST, "non-finite velocity command")

        now = self._clock()
        active_lease_is_fresh = (
            self._lease_id is not None
            and self._lease_time is not None
            and self._raw_lease_id == self._lease_id
            and lease_is_fresh(
                active=True,
                heartbeat_age_sec=now - self._lease_time,
                max_age_sec=self._lease_timeout_sec,
            )
        )
        raw_command_is_fresh = (
            self._raw_time is not None
            and command_is_fresh(
                now - self._raw_time,
                self._raw_command_timeout_sec,
            )
        )
        return GuardOutput(
            sanitize_twist(
                linear_x=self._raw_twist[0],
                linear_y=self._raw_twist[1],
                angular_z=self._raw_twist[2],
                lease_fresh=active_lease_is_fresh,
                command_fresh=raw_command_is_fresh,
            )
        )

    def _velocity_is_finite(self) -> bool:
        try:
            return all(math.isfinite(value) for value in self._raw_twist or ())
        except TypeError:
            return False

    def _invalidate_raw_command(self) -> None:
        self._raw_twist = None
        self._raw_time = None
        self._raw_lease_id = None


def main(args=None) -> None:
    """Run the ROS adapter while keeping the guard state unit-testable without ROS."""
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String

    velocity_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )
    lease_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )

    class CmdVelGuardNode(Node):
        def __init__(self):
            super().__init__("cmd_vel_guard")
            self.declare_parameter(
                "lease_timeout_sec",
                COMMAND_FRESHNESS_SEC,
            )
            self.declare_parameter(
                "raw_command_timeout_sec",
                COMMAND_FRESHNESS_SEC,
            )
            self.declare_parameter("guard_period_sec", GUARD_PERIOD_SEC)
            self._state = CmdVelGuardState(
                clock=self._now_sec,
                lease_timeout_sec=self.get_parameter(
                    "lease_timeout_sec"
                ).value,
                raw_command_timeout_sec=self.get_parameter(
                    "raw_command_timeout_sec"
                ).value,
            )
            self._last_reported_error = None
            self._publisher = self.create_publisher(Twist, "/cmd_vel", velocity_qos)
            self.create_subscription(
                Twist,
                "/navigation/raw_cmd_vel",
                self._on_raw_command,
                velocity_qos,
            )
            self.create_subscription(
                String,
                "/navigation/command_lease",
                self._on_lease,
                lease_qos,
            )
            guard_period_sec = _positive_finite(
                "guard_period_sec",
                self.get_parameter("guard_period_sec").value,
            )
            self.create_timer(guard_period_sec, self._on_guard_tick)

        def _now_sec(self) -> float:
            return self.get_clock().now().nanoseconds / 1_000_000_000.0

        def _on_raw_command(self, message: Twist) -> None:
            self._state.on_raw_command(
                message.linear.x,
                message.linear.y,
                message.angular.z,
            )

        def _on_lease(self, message: String) -> None:
            previous = self._state.lease_id
            self._state.on_lease(message.data)
            current = self._state.lease_id
            if current == previous:
                return
            if current is None:
                self.get_logger().info("navigation command lease revoked")
            else:
                self.get_logger().info(f"navigation command lease active: {current}")

        def _on_guard_tick(self) -> None:
            output = self._state.tick()
            message = Twist()
            message.linear.x, message.linear.y, message.angular.z = output.twist
            self._publisher.publish(message)
            if output.error and output.error != self._last_reported_error:
                self.get_logger().error(output.error)
            self._last_reported_error = output.error

    rclpy.init(args=args)
    node = CmdVelGuardNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _positive_finite(name: str, value: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return result
