#!/usr/bin/env python3
# coding: utf-8

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu
import os
import time

class PoseImuRecorder(Node):
    def __init__(self, max_frames=1000):
        super().__init__('pose_imu_recorder')

        self.max_frames = max_frames
        self.frame_count = 0

        # 保存目录
        self.save_dir = os.path.expanduser('/mnt/d/hp60c_frames')
        os.makedirs(self.save_dir, exist_ok=True)
        self.pose_file_path = os.path.join(self.save_dir, 'pose_imu.txt')
        self.pose_file = open(self.pose_file_path, 'w')
        self.pose_file.write("# timestamp tx ty tz qx qy qz qw ax ay az\n")

        self.latest_imu = None

        # 订阅 ORB-SLAM2 位姿
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/RGBD/CameraPose',
            self.pose_callback,
            10
        )

        # 订阅 IMU
        self.imu_sub = self.create_subscription(
            Imu,
            '/ascamera_hp60c/imu',  # 根据实际 topic 改
            self.imu_callback,
            10
        )

    def imu_callback(self, msg: Imu):
        # 保存最新的 imu 数据
        self.latest_imu = msg

    def pose_callback(self, msg: PoseStamped):
        if self.frame_count >= self.max_frames:
            self.get_logger().info(f"Captured {self.max_frames} frames, shutting down...")
            self.pose_file.close()
            rclpy.shutdown()
            return

        timestamp = f"{msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}"

        # 位姿
        tx = msg.pose.position.x
        ty = msg.pose.position.y
        tz = msg.pose.position.z
        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w

        # IMU
        if self.latest_imu is not None:
            ax = self.latest_imu.linear_acceleration.x
            ay = self.latest_imu.linear_acceleration.y
            az = self.latest_imu.linear_acceleration.z
        else:
            ax = ay = az = 0.0

        # 写入文件
        self.pose_file.write(f"{timestamp} {tx} {ty} {tz} {qx} {qy} {qz} {qw} {ax} {ay} {az}\n")
        self.pose_file.flush()

        self.frame_count += 1
        self.get_logger().info(f"Saved frame {self.frame_count}: pose + imu")

def main(args=None):
    rclpy.init(args=args)
    node = PoseImuRecorder(max_frames=200)  # 可以修改抓取帧数
    rclpy.spin(node)

if __name__ == '__main__':
    main()
