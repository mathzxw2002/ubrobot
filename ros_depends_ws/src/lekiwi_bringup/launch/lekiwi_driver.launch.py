"""Start the LeKiwi ros2_control stack in safe mock mode by default."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _launch_nodes(context):
    hardware_mode = LaunchConfiguration("hardware_mode").perform(context)
    if hardware_mode != "mock":
        raise RuntimeError(
            "Only hardware_mode:=mock is implemented. Real hardware is intentionally disabled."
        )

    description_file = PathJoinSubstitution(
        [FindPackageShare("lekiwi_description"), "urdf", "lekiwi_base.urdf.xacro"]
    )
    controllers_file = PathJoinSubstitution(
        [FindPackageShare("lekiwi_bringup"), "config", "controllers.yaml"]
    )
    robot_description = {
        "robot_description": Command(
            [
                FindExecutable(name="xacro"),
                " ",
                description_file,
                " hardware_mode:=",
                hardware_mode,
            ]
        )
    }

    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[robot_description],
        ),
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            output="screen",
            parameters=[robot_description, controllers_file],
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
            output="screen",
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["lekiwi_base_controller", "--controller-manager", "/controller_manager"],
            output="screen",
        ),
        Node(
            package="lekiwi_bringup",
            executable="cmd_vel_adapter",
            output="screen",
            parameters=[
                {
                    "input_topic": "/cmd_vel",
                    "output_topic": "/lekiwi_base_controller/cmd_vel",
                    "max_linear_x": 0.05,
                    "max_linear_y": 0.05,
                    "max_angular_z": 0.20,
                    "command_timeout": 0.25,
                    "watchdog_period": 0.05,
                }
            ],
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("hardware_mode", default_value="mock"),
            OpaqueFunction(function=_launch_nodes),
        ]
    )
