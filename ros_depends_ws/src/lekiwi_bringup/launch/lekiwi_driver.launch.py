"""Start the LeKiwi ros2_control stack in safe mock mode by default."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _launch_nodes(context):
    hardware_mode = LaunchConfiguration("hardware_mode").perform(context)
    enable_real_hardware = LaunchConfiguration("enable_real_hardware").perform(context).lower()
    enable_motor_torque = LaunchConfiguration("enable_motor_torque").perform(context).lower()
    if hardware_mode not in {"mock", "real"}:
        raise RuntimeError(
            f"Unsupported hardware_mode: {hardware_mode!r}. Expected 'mock' or 'real'."
        )
    if enable_real_hardware not in {"true", "false"}:
        raise RuntimeError("enable_real_hardware must be 'true' or 'false'.")
    if enable_motor_torque not in {"true", "false"}:
        raise RuntimeError("enable_motor_torque must be 'true' or 'false'.")
    if hardware_mode == "real" and enable_real_hardware != "true":
        raise RuntimeError(
            "Real LeKiwi hardware is locked. Re-run with enable_real_hardware:=true "
            "only after lifting the wheels and verifying the stable serial device."
        )
    if enable_motor_torque == "true" and (
        hardware_mode != "real" or enable_real_hardware != "true"
    ):
        raise RuntimeError(
            "LeKiwi motor torque requires hardware_mode:=real and "
            "enable_real_hardware:=true."
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
                " enable_motor_torque:=",
                enable_motor_torque,
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
            DeclareLaunchArgument("enable_real_hardware", default_value="false"),
            DeclareLaunchArgument("enable_motor_torque", default_value="false"),
            OpaqueFunction(function=_launch_nodes),
        ]
    )
