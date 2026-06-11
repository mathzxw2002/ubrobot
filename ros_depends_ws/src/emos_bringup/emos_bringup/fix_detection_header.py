"""Republish EMOS detections with image-aligned header stamps."""

import rclpy
from automatika_embodied_agents.msg import Detections2D
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


def _has_stamp(stamp):
    return stamp.sec != 0 or stamp.nanosec != 0


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

        self._publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DetectionHeaderFixer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
