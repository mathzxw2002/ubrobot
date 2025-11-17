from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 启动RealSense相机节点
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            parameters=[{
                'enable_gyro': True,
                'enable_accel': True,
                'depth_module.profile': '640x480x30',
                'color_module.profile': '640x480x30'
            }]
        ),
        # 启动RTAB-Map节点，发布odom_bridge话题
        Node(
            package='rtabmap_ros',
            executable='rtabmap',
            parameters=[{
                'rgb_topic': '/camera/color/image_raw',
                'depth_topic': '/camera/depth/image_rect_raw',
                'camera_info_topic': '/camera/color/camera_info',
                'imu_topic': '/camera/imu',
                'odom_topic': '/odom_bridge'  # 自定义odom_bridge话题
            }]
        )
    ])