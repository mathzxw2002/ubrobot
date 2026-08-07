"""Piper arm driver node (hardware layer, go2-piper-driver container).

Executes guarded joint/gripper commands on the Piper arm over CAN. It is the
ONLY node allowed to touch ``piper_sdk`` inside the container:

- subscribes ``/piper/joint_cmd`` (``sensor_msgs/JointState``): 6 joint
  positions in RADIANS + an optional 7th gripper position (mm);
- converts to the SDK ``JointCtrl`` / ``GripperCtrl`` units and executes;
- publishes ``/piper/joint_states`` (radians) and ``/piper/arm_status``
  (plain diagnostic string) for telemetry/health.

Safety:
- the node does NOT enable torque itself — ``/piper/enable`` (std_srvs/SetBool)
  must be called first; commands before enable are dropped (fail-closed);
- commanded joints are clamped to the Piper SDK joint limits (deg) so a bad
  plan can never ask the hardware for an out-of-range angle;
- the piper_sdk import is deferred so the node starts (and can be unit-tested)
  even where the SDK is absent, and always reports status truthfully.
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import SetBool

# SDK joint limits (degrees), from piper_sdk documentation / S1 probe.
_JOINT_LIMITS_DEG: tuple[tuple[float, float], ...] = (
    (-150.0, 150.0),
    (0.0, 180.0),
    (-170.0, 0.0),
    (-100.0, 100.0),
    (-70.0, 70.0),
    (-120.0, 120.0),
)


def joint_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


class PiperDriverNode(Node):
    """Executes Piper joint/gripper commands over CAN (hardware layer)."""

    def __init__(
        self,
        *,
        can_port: str | None = None,
        sdk_factory: Callable[[str], Any] | None = None,
    ) -> None:
        super().__init__("piper_driver")
        self._can_port = can_port or os.environ.get("PIPER_CAN_INTERFACE", "can0")
        self._sdk_factory = sdk_factory or self._default_sdk_factory
        self._piper: Any = None
        self._sdk_error: str | None = None
        self._enabled = False
        self._lock_state: dict[str, Any] = {}

        self._cmd_sub = self.create_subscription(
            JointState, "/piper/joint_cmd", self._on_joint_cmd, joint_qos()
        )
        self._state_pub = self.create_publisher(
            JointState, "/piper/joint_states", joint_qos()
        )
        self._status_pub = self.create_publisher(
            String, "/piper/arm_status", joint_qos()
        )
        self._enable_srv = self.create_service(
            SetBool, "/piper/enable", self._on_enable
        )
        self.create_timer(0.1, self._publish_state)
        self.get_logger().info(
            "piper driver up: can=%s (torque NOT enabled)", self._can_port
        )

    # ---------------------------------------------------------------- public

    def _piper_iface(self) -> Any | None:
        """Lazily connect the SDK; returns None (fail-closed) on failure."""
        if self._piper is not None:
            return self._piper
        if self._sdk_error is not None:
            return None
        try:
            self._piper = self._sdk_factory(self._can_port)
        except Exception as exc:  # noqa: BLE001 - report once, stay telemetry-only
            self._sdk_error = str(exc)
            self.get_logger().warning("piper SDK unavailable: %s", exc)
            return None
        return self._piper

    def _on_enable(self, request: SetBool.Request, _response: SetBool.Response) -> SetBool.Response:
        piper = self._piper_iface()
        if piper is None:
            self.get_logger().error("enable rejected: piper SDK unavailable")
            return SetBool.Response(success=False, message="piper SDK unavailable")
        try:
            if request.data:
                piper.EnablePiper()
                piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
                self._enabled = True
                self.get_logger().info("piper torque ENABLED")
                return SetBool.Response(success=True, message="torque enabled")
            piper.DisablePiper()
            self._enabled = False
            self.get_logger().info("piper torque DISABLED")
            return SetBool.Response(success=True, message="torque disabled")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error("enable failed: %s", exc)
            self._enabled = False
            return SetBool.Response(success=False, message=str(exc))

    def _on_joint_cmd(self, msg: JointState) -> None:
        if not self._enabled:
            self.get_logger().warning("joint command dropped: torque not enabled")
            return
        piper = self._piper_iface()
        if piper is None:
            self.get_logger().error("joint command dropped: piper SDK unavailable")
            return
        if len(msg.position) < 6:
            self.get_logger().warning(
                "joint command dropped: need >=6 positions, got %d", len(msg.position)
            )
            return
        try:
            joints_deg = [math.degrees(float(v)) for v in msg.position[:6]]
            clamped = [
                min(max(deg, lo), hi)
                for deg, (lo, hi) in zip(joints_deg, _JOINT_LIMITS_DEG, strict=True)
            ]
            # SDK JointCtrl expects thousandths of a degree.
            piper.JointCtrl(*[int(round(d * 1000.0)) for d in clamped])
            if len(msg.position) >= 7 and msg.position[6] is not None:
                gripper_mm = min(max(float(msg.position[6]), 0.0), 10.0)
                piper.GripperCtrl(int(round(gripper_mm * 10000.0)), 1000, 0x01, 0)
            self._lock_state = {
                "joints_deg": clamped,
                "gripper_mm": float(msg.position[6]) if len(msg.position) >= 7 else None,
            }
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error("joint command failed: %s", exc)

    def _publish_state(self) -> None:
        piper = self._piper_iface()
        state = JointState()
        state.header.stamp = self.get_clock().now().to_msg()
        state.name = [f"joint{i}" for i in range(1, 7)] + ["gripper"]
        if piper is None:
            state.position = [0.0] * 7
            state.velocity = [0.0] * 7
            state.effort = [0.0] * 7
        else:
            try:
                js = piper.GetArmJointMsgs().joint_state
                positions = [getattr(js, f"joint_{i}") / 1000.0 for i in range(1, 7)]
                positions = [math.radians(v) for v in positions]  # deg -> rad
                g = piper.GetArmGripperMsgs()
                gripper_mm = (
                    getattr(g.gripper_state, "grippers_angle", 0) / 10000.0
                )
                state.position = positions + [gripper_mm]
                state.velocity = [0.0] * 7
                state.effort = [0.0] * 7
            except Exception:
                state.position = [0.0] * 7
                state.velocity = [0.0] * 7
                state.effort = [0.0] * 7
        self._state_pub.publish(state)
        self._status_pub.publish(
            String(
                data=(
                    f"enabled={self._enabled} sdk={'ok' if piper is not None else 'unavailable'}"
                    f"{(' error=' + self._sdk_error) if self._sdk_error else ''}"
                )
            )
        )

    def _default_sdk_factory(self, can_port: str) -> Any:
        from piper_sdk import C_PiperInterface_V2  # noqa: PLC0415 - deferred

        piper = C_PiperInterface_V2(can_port)
        piper.ConnectPort()
        return piper


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PiperDriverNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
