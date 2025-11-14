#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import math

class OdomListener(Node):
    def __init__(self):
        super().__init__('odom_listener')
        self.subscription = self.create_subscription(
            Odometry,
            '/RGBD/Odometry',
            self.odom_callback,
            10
        )
        self.get_logger().info("Subscribed to /RGBD/Odometry")

    def odom_callback(self, msg: Odometry):
        # 提取位置
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # 提取四元数
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        # 四元数 -> yaw
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        self.get_logger().info(f"x: {x:.3f}, y: {y:.3f}, yaw: {yaw:.3f} rad")

def main(args=None):
    rclpy.init(args=args)
    node = OdomListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
