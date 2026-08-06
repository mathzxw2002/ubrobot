from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Bring up the Go2 ROS 2 bridge node.

    The bridge subscribes the guarded ``/cmd_vel`` (from ``cmd_vel_guard``)
    and publishes ``/odom``, ``/imu``, ``/joint_states`` for the read-only
    health/telemetry probes. It does NOT stand the dog up or enable sport
    mode: that is the operator's pre-step (or a future bridge primitive).
    """
    return LaunchDescription(
        [
            Node(
                package="go2_bridge",
                executable="go2_bridge_node",
                name="go2_bridge",
                output="screen",
            )
        ]
    )
