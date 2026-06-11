"""Republish EMOS detections with image-aligned header stamps."""

import rclpy
from automatika_embodied_agents.msg import Detections2D
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


def _has_stamp(stamp):
    return stamp.sec != 0 or stamp.nanosec != 0


def _stamp_to_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def _set_stamp_from_ns(stamp, stamp_ns):
    stamp.sec = stamp_ns // 1_000_000_000
    stamp.nanosec = stamp_ns % 1_000_000_000


class DetectionHeaderFixer(Node):
    """Make the top-level detection header match the RGB-D payload header."""

    def __init__(self):
        super().__init__("fix_detection_header")
        self.declare_parameter("input_topic", "/vision_detections_raw")
        self.declare_parameter("output_topic", "/vision_detections")

        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
        )

        self._publisher = self.create_publisher(Detections2D, output_topic, qos)
        self._last_stamp_ns = None
        self.create_subscription(Detections2D, input_topic, self._callback, qos)
        self.get_logger().info(
            f"Republishing detections from {input_topic} to {output_topic}"
        )

    def _callback(self, msg):
        if _has_stamp(msg.image.header.stamp):
            msg.header.stamp = msg.image.header.stamp
        elif _has_stamp(msg.depth.header.stamp):
            msg.header.stamp = msg.depth.header.stamp

        if msg.image.header.frame_id:
            msg.header.frame_id = msg.image.header.frame_id
        elif msg.depth.header.frame_id:
            msg.header.frame_id = msg.depth.header.frame_id

        if _has_stamp(msg.header.stamp):
            stamp_ns = _stamp_to_ns(msg.header.stamp)
            if self._last_stamp_ns is not None and stamp_ns <= self._last_stamp_ns:
                stamp_ns = self._last_stamp_ns + 1
                _set_stamp_from_ns(msg.header.stamp, stamp_ns)
            self._last_stamp_ns = stamp_ns

        self._publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DetectionHeaderFixer()
    try:
        rclpy.spin(node)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
