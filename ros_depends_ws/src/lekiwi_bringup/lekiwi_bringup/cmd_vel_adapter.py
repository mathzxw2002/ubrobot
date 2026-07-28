"""Validate EMOS Twist commands and publish safe TwistStamped commands."""

from geometry_msgs.msg import Twist, TwistStamped
import rclpy
from rclpy.node import Node

from .velocity_safety import VelocityLimits, sanitize_velocity


class CmdVelAdapter(Node):
    def __init__(self) -> None:
        super().__init__("lekiwi_cmd_adapter")

        self.declare_parameter("input_topic", "/cmd_vel")
        self.declare_parameter("output_topic", "/lekiwi_base_controller/cmd_vel")
        self.declare_parameter("max_linear_x", 0.05)
        self.declare_parameter("max_linear_y", 0.05)
        self.declare_parameter("max_angular_z", 0.20)
        self.declare_parameter("command_timeout", 0.25)
        self.declare_parameter("watchdog_period", 0.05)

        self._limits = VelocityLimits(
            linear_x=float(self.get_parameter("max_linear_x").value),
            linear_y=float(self.get_parameter("max_linear_y").value),
            angular_z=float(self.get_parameter("max_angular_z").value),
        )
        self._command_timeout = float(self.get_parameter("command_timeout").value)
        watchdog_period = float(self.get_parameter("watchdog_period").value)
        if self._command_timeout <= 0.0 or watchdog_period <= 0.0:
            raise ValueError("watchdog timing parameters must be greater than zero")

        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        self._publisher = self.create_publisher(TwistStamped, output_topic, 10)
        self._subscription = self.create_subscription(
            Twist,
            input_topic,
            self._on_command,
            10,
        )
        self._last_valid_command_ns: int | None = None
        self._watchdog = self.create_timer(watchdog_period, self._on_watchdog)

        self._publish(0.0, 0.0, 0.0)
        self.get_logger().info(
            f"Validating {input_topic} -> {output_topic} with "
            f"{self._command_timeout:.3f}s timeout"
        )

    def _publish(self, linear_x: float, linear_y: float, angular_z: float) -> None:
        message = TwistStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "base_link"
        message.twist.linear.x = linear_x
        message.twist.linear.y = linear_y
        message.twist.angular.z = angular_z
        self._publisher.publish(message)

    def _on_command(self, message: Twist) -> None:
        linear_x, linear_y, angular_z, valid = sanitize_velocity(
            message.linear.x,
            message.linear.y,
            message.angular.z,
            self._limits,
        )
        if not valid:
            self.get_logger().error("Rejected non-finite /cmd_vel; publishing zero")
            self._publish(0.0, 0.0, 0.0)
            return

        self._last_valid_command_ns = self.get_clock().now().nanoseconds
        self._publish(linear_x, linear_y, angular_z)

    def _on_watchdog(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        if self._last_valid_command_ns is None:
            self._publish(0.0, 0.0, 0.0)
            return

        age_seconds = (now_ns - self._last_valid_command_ns) / 1_000_000_000.0
        if age_seconds > self._command_timeout:
            self._publish(0.0, 0.0, 0.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CmdVelAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node._publish(0.0, 0.0, 0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
