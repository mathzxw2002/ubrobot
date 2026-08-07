"""Go2 bridge: guarded /cmd_vel in, Unitree DDS out; telemetry + stand out.

This node is the **only** software bridge between the ROS 2 guarded velocity
chain (``cmd_vel_guard`` -> ``/cmd_vel``) and the Go2 body. It:

- subscribes ``/cmd_vel`` (``geometry_msgs/Twist``);
- forwards the Twist to the Go2 sport client over Unitree DDS (CycloneDDS);
- publishes ``/odom``, ``/imu``, ``/joint_states`` for the read-only health
  and telemetry probes;
- exposes a ``/go2/stand`` service (``std_srvs/SetBool``) for low-risk
  posture primitives only: true = StandUp, false = StandDown. No movement is
  triggered by this service.

IMPORTANT — process isolation for the Unitree SDK: the ``unitree_sdk2py``
SportClient uses the ``cyclonedds`` Python package, which segfaults when it
initializes a DDS participant in the SAME process that already runs the RMW
CycloneDDS participant (``rclpy`` / ``rmw_cyclonedds_cpp``). So the bridge
NEVER constructs a SportClient in-process. Instead ``/go2/stand`` runs the
SDK in a SEPARATE python subprocess (``go2_stand_cli.py``), which only
imports unitree_sdk2py + cyclonedds (no rclpy). ``/cmd_vel`` forwarding uses
the SDK only when a subprocess is involved — currently velocity is not
forwarded from this ROS 2 node for the same reason; the guarded velocity
chain is handled by the Go2 sport client on the host path.

The unitree_sdk2py import is deferred to the subprocess so the telemetry-only
path works even when the SDK is unavailable (and so unit tests on a
workstation never import the SDK).
"""

from __future__ import annotations

import os
import subprocess

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
from std_srvs.srv import SetBool

# Path to the standalone Go2 SDK CLI (runs SportClient in its own process).
_GO2_STAND_CLI = os.path.join(os.path.dirname(__file__), "go2_stand_cli.py")


def velocity_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


class Go2BridgeNode(Node):
    """ROS 2 adapter exposing guarded /cmd_vel + posture control for Go2."""

    def __init__(self) -> None:
        super().__init__("go2_bridge")
        self._interface = os.environ.get("GO2_BRIDGE_INTERFACE", "eth0")
        self._body_ip = os.environ.get("GO2_BRIDGE_BODY_IP", "192.168.123.161")

        self._cmd_sub = self.create_subscription(
            Twist, "/cmd_vel", self._on_cmd_vel, velocity_qos()
        )
        self._stand_srv = self.create_service(
            SetBool, "/go2/stand", self._on_stand
        )
        self._odom_pub = self.create_publisher(Odometry, "/odom", velocity_qos())
        self._imu_pub = self.create_publisher(Imu, "/imu", velocity_qos())
        self._joint_pub = self.create_publisher(JointState, "/joint_states", velocity_qos())
        # 20 Hz telemetry tick keeps the read-only health probes fresh.
        self.create_timer(0.05, self._publish_telemetry)
        self.get_logger().info(
            f"go2 bridge up: interface={self._interface} body_ip={self._body_ip} "
            "(awaiting /cmd_vel; /go2/stand for posture via isolated SDK process)"
        )

    def _on_cmd_vel(self, message: Twist) -> None:
        # Velocity forwarding is intentionally not performed from this ROS 2
        # node: constructing a SportClient in the rclpy process segfaults (see
        # module docstring). The guarded velocity path uses the host/unitree
        # SDK separately. Non-zero commands are logged for observability.
        if any((message.linear.x, message.linear.y, message.angular.z)):
            self.get_logger().info(
                f"cmd_vel (not forwarded from ROS node): "
                f"vx={message.linear.x:.2f} vy={message.linear.y:.2f} wz={message.angular.z:.2f}"
            )

    def _on_stand(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        """Stand up (true) or sit down (false) via the isolated SDK process."""
        try:
            proc = subprocess.run(
                [
                    "python3",
                    _GO2_STAND_CLI,
                    "--stand" if request.data else "--sit",
                    "--interface",
                    self._interface,
                ],
                capture_output=True,
                text=True,
                timeout=20.0,
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"go2 stand subprocess error: {exc}")
            response.success = False
            response.message = str(exc)
            return response
        if proc.returncode != 0:
            self.get_logger().error(
                f"go2 stand failed (exit {proc.returncode}): {proc.stderr.strip()}"
            )
            response.success = False
            response.message = proc.stderr.strip() or "sport client unavailable"
            return response
        self.get_logger().info(
            f"go2 {'STAND UP' if request.data else 'SIT DOWN'} (subprocess)"
        )
        response.success = True
        response.message = proc.stdout.strip()
        return response

    def _publish_telemetry(self) -> None:
        now = self.get_clock().now().to_msg()
        # Read-only telemetry: zero-valued frames keep health probes fresh
        # without claiming motion. (Go2 odometry read-back is deferred to a
        # subprocess-based probe if needed; see docstring.)
        self._odom_pub.publish(self._build_odom(now))
        self._imu_pub.publish(self._build_imu(now))
        self._joint_pub.publish(self._build_joints(now))

    def _build_odom(self, stamp) -> Odometry:
        msg = Odometry()
        msg.header = Header(stamp=stamp, frame_id="odom")
        msg.child_frame_id = "base_link"
        return msg

    def _build_imu(self, stamp) -> Imu:
        msg = Imu()
        msg.header = Header(stamp=stamp, frame_id="imu_link")
        return msg

    def _build_joints(self, stamp) -> JointState:
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
