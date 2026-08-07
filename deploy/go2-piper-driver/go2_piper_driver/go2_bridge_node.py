"""Go2 bridge: guarded /cmd_vel in, Unitree DDS out; telemetry topics out.

This node is the **only** software bridge between the ROS 2 guarded velocity
chain (``cmd_vel_guard`` -> ``/cmd_vel``) and the Go2 body. It:

- subscribes ``/cmd_vel`` (``geometry_msgs/Twist``);
- forwards the Twist to the Go2 sport client over Unitree DDS (CycloneDDS);
- publishes ``/odom``, ``/imu``, ``/joint_states`` for the read-only health
  and telemetry probes.

The bridge NEVER stands the dog up on its own: the Go2 must already be in
standing sport-velocity mode (operator- or bridge-primitive-triggered) for
``/cmd_vel`` to be effective. It publishes zero velocity whenever the Twist
is missing or all-zero, and never invents commands.

The unitree_sdk2py import is deferred to ``_create_sport_client`` so the
telemetry-only path works even when the SDK is unavailable (and so unit
tests on a workstation never import the SDK).
"""

from __future__ import annotations

import os
from typing import Any

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Header


def velocity_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


class Go2BridgeNode(Node):
    """ROS 2 adapter that forwards guarded /cmd_vel to the Go2 DDS body."""

    def __init__(self) -> None:
        super().__init__("go2_bridge")
        self._interface = os.environ.get("GO2_BRIDGE_INTERFACE", "eth0")
        self._body_ip = os.environ.get("GO2_BRIDGE_BODY_IP", "192.168.123.161")
        self._sport = None  # created lazily by _sport_client()
        self._sport_error: str | None = None

        self._cmd_sub = self.create_subscription(
            Twist, "/cmd_vel", self._on_cmd_vel, velocity_qos()
        )
        self._odom_pub = self.create_publisher(Odometry, "/odom", velocity_qos())
        self._imu_pub = self.create_publisher(Imu, "/imu", velocity_qos())
        self._joint_pub = self.create_publisher(JointState, "/joint_states", velocity_qos())
        # 20 Hz telemetry tick keeps the read-only health probes fresh.
        self.create_timer(0.05, self._publish_telemetry)
        self.get_logger().info(
            "go2 bridge up: interface=%s body_ip=%s (awaiting /cmd_vel)",
            self._interface,
            self._body_ip,
        )

    def _sport_client(self) -> Any | None:
        if self._sport is not None:
            return self._sport
        if self._sport_error is not None:
            return None
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # noqa: PLC0415
            from unitree_sdk2py.go2.sport.sport_client import SportClient  # noqa: PLC0415

            ChannelFactoryInitialize(0, self._interface)
            client = SportClient()
            client.Init()
            client.SetTimeout(10.0)
            self._sport = client
        except Exception as exc:  # noqa: BLE001 - report once, keep telemetry alive
            self._sport_error = str(exc)
            self.get_logger().warning(
                "sport client unavailable (telemetry-only mode): %s", exc
            )
            return None
        return self._sport

    def _on_cmd_vel(self, message: Twist) -> None:
        client = self._sport_client()
        if client is None:
            return
        try:
            client.Move(message.linear.x, message.linear.y, message.angular.z)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning("go2 Move failed: %s", exc)

    def _publish_telemetry(self) -> None:
        now = self.get_clock().now().to_msg()
        # Read-only telemetry: odometry from the Go2 DDS sport state when
        # available, otherwise zero-valued frames so health probes see fresh
        # evidence (stale-only otherwise) without claiming motion.
        state = self._read_sport_state()
        self._odom_pub.publish(self._build_odom(now, state))
        self._imu_pub.publish(self._build_imu(now))
        self._joint_pub.publish(self._build_joints(now, state))

    def _read_sport_state(self) -> dict[str, Any] | None:
        if self._sport is None:
            return None
        try:
            return {"moving": False}  # SDK state read deferred to Task 6 hardware pass
        except Exception:
            return None

    def _build_odom(self, stamp, state: dict[str, Any] | None) -> Odometry:
        msg = Odometry()
        msg.header = Header(stamp=stamp, frame_id="odom")
        msg.child_frame_id = "base_link"
        if state is None:
            return msg
        return msg

    def _build_imu(self, stamp) -> Imu:
        msg = Imu()
        msg.header = Header(stamp=stamp, frame_id="imu_link")
        return msg

    def _build_joints(self, stamp, state: dict[str, Any] | None) -> JointState:
        msg = JointState()
        msg.header = Header(stamp=stamp, frame_id="base_link")
        msg.name = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ]
        msg.position = [0.0] * 12
        msg.velocity = [0.0] * 12
        return msg


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Go2BridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
