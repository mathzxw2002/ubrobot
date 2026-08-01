"""Compose sensors with the guarded semantic navigation capability."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    start_sensors = LaunchConfiguration("start_sensors")
    lease_timeout_sec = LaunchConfiguration("lease_timeout_sec")
    raw_command_timeout_sec = LaunchConfiguration("raw_command_timeout_sec")
    guard_period_sec = LaunchConfiguration("guard_period_sec")
    start_grasp_server = LaunchConfiguration("start_grasp_server")
    grasp_platform = LaunchConfiguration("grasp_platform")
    grasp_executor = LaunchConfiguration("grasp_executor")
    fixture_phase_delay_sec = LaunchConfiguration("fixture_phase_delay_sec")

    sensor_launch = PathJoinSubstitution(
        [
            FindPackageShare("emos_bringup"),
            "launch",
            "vision_depth_bringup.launch.py",
        ]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "start_sensors",
                default_value="true",
                description="Include the validated RealSense/odom/scan/TF chain.",
            ),
            DeclareLaunchArgument(
                "lease_timeout_sec",
                default_value="0.25",
                description="Maximum command-lease heartbeat age.",
            ),
            DeclareLaunchArgument(
                "raw_command_timeout_sec",
                default_value="0.25",
                description="Maximum raw velocity command age.",
            ),
            DeclareLaunchArgument(
                "guard_period_sec",
                default_value="0.05",
                description="Fixed guarded /cmd_vel publication period.",
            ),
            DeclareLaunchArgument(
                "start_grasp_server",
                default_value="false",
                description="Start grasp only for an explicit offline harness.",
            ),
            DeclareLaunchArgument(
                "grasp_platform",
                default_value="piper_station",
                description="Offline grasp platform profile.",
            ),
            DeclareLaunchArgument(
                "grasp_executor",
                default_value="none",
                description="Use fixture only; real bindings are deferred.",
            ),
            DeclareLaunchArgument(
                "fixture_phase_delay_sec",
                default_value="0.05",
                description="Delay used by the deterministic grasp fixture.",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(sensor_launch),
                condition=IfCondition(start_sensors),
            ),
            Node(
                package="ubrobot_navigation",
                executable="navigate_to_object_server",
                name="navigate_to_object_server",
                output="screen",
            ),
            Node(
                package="ubrobot_manipulation",
                executable="grasp_object_server",
                name="grasp_object_server",
                output="screen",
                condition=IfCondition(start_grasp_server),
                env={
                    "UBROBOT_GRASP_PLATFORM": grasp_platform,
                    "UBROBOT_GRASP_EXECUTOR": grasp_executor,
                },
                parameters=[
                    {
                        "fixture_phase_delay_sec": ParameterValue(
                            fixture_phase_delay_sec,
                            value_type=float,
                        )
                    }
                ],
            ),
            Node(
                package="ubrobot_navigation",
                executable="cmd_vel_guard",
                name="cmd_vel_guard",
                output="screen",
                parameters=[
                    {
                        "lease_timeout_sec": ParameterValue(
                            lease_timeout_sec,
                            value_type=float,
                        ),
                        "raw_command_timeout_sec": ParameterValue(
                            raw_command_timeout_sec,
                            value_type=float,
                        ),
                        "guard_period_sec": ParameterValue(
                            guard_period_sec,
                            value_type=float,
                        ),
                    }
                ],
            ),
        ]
    )
