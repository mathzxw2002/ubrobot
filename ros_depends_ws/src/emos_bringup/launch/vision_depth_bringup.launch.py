"""Bring up RealSense, RGB-D odom, scan, and TFs for EMOS vision tracking."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    color_profile = LaunchConfiguration("color_profile")
    depth_profile = LaunchConfiguration("depth_profile")
    approx_sync_max_interval = LaunchConfiguration("approx_sync_max_interval")

    base_to_camera_x = LaunchConfiguration("base_to_camera_x")
    base_to_camera_y = LaunchConfiguration("base_to_camera_y")
    base_to_camera_z = LaunchConfiguration("base_to_camera_z")

    realsense_launch = PathJoinSubstitution(
        [FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "color_profile",
                default_value="640x480x15",
                description="RealSense color stream profile.",
            ),
            DeclareLaunchArgument(
                "depth_profile",
                default_value="640x480x15",
                description="RealSense depth stream profile.",
            ),
            DeclareLaunchArgument(
                "approx_sync_max_interval",
                default_value="0.05",
                description="RTAB-Map RGB/depth approximate sync window in seconds.",
            ),
            DeclareLaunchArgument(
                "base_to_camera_x",
                default_value="0.10",
                description="Camera x offset from base_link, in meters.",
            ),
            DeclareLaunchArgument(
                "base_to_camera_y",
                default_value="0.030",
                description="Camera y offset from base_link, in meters.",
            ),
            DeclareLaunchArgument(
                "base_to_camera_z",
                default_value="0.20",
                description="Camera z offset from base_link, in meters.",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(realsense_launch),
                launch_arguments={
                    "enable_color": "true",
                    "enable_depth": "true",
                    "align_depth.enable": "true",
                    "enable_sync": "true",
                    "enable_rgbd": "true",
                    "rgb_camera.color_profile": color_profile,
                    "depth_module.depth_profile": depth_profile,
                }.items(),
            ),
            # Kompass defaults to frames.depth == camera_depth_link.
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="base_to_camera_depth_link_tf",
                arguments=[
                    base_to_camera_x,
                    base_to_camera_y,
                    base_to_camera_z,
                    "0",
                    "0",
                    "0",
                    "base_link",
                    "camera_depth_link",
                ],
                output="screen",
            ),
            # RealSense publishes camera_link -> camera_* frames; this connects
            # the RealSense tree to the robot body tree.
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="base_to_camera_link_tf",
                arguments=[
                    base_to_camera_x,
                    base_to_camera_y,
                    base_to_camera_z,
                    "0",
                    "0",
                    "0",
                    "base_link",
                    "camera_link",
                ],
                output="screen",
            ),
            Node(
                package="rtabmap_odom",
                executable="rgbd_odometry",
                name="rgbd_odometry",
                output="screen",
                parameters=[
                    {
                        "frame_id": "base_link",
                        "odom_frame_id": "odom",
                        "publish_tf": True,
                        "approx_sync": True,
                        "approx_sync_max_interval": approx_sync_max_interval,
                        "topic_queue_size": 50,
                        "sync_queue_size": 50,
                        "qos": 1,
                        "qos_camera_info": 1,
                    }
                ],
                remappings=[
                    ("rgb/image", "/camera/camera/color/image_raw"),
                    ("depth/image", "/camera/camera/aligned_depth_to_color/image_raw"),
                    ("rgb/camera_info", "/camera/camera/color/camera_info"),
                    ("odom", "/odom"),
                ],
            ),
            Node(
                package="depthimage_to_laserscan",
                executable="depthimage_to_laserscan_node",
                name="depthimage_to_laserscan",
                output="screen",
                remappings=[
                    ("depth", "/camera/camera/aligned_depth_to_color/image_raw"),
                    (
                        "depth_camera_info",
                        "/camera/camera/aligned_depth_to_color/camera_info",
                    ),
                    ("scan", "/scan"),
                ],
            ),
        ]
    )
