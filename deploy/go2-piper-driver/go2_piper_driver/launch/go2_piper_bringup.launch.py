from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Bring up the Go2+Piper hardware driver container.

    - ``go2_bridge``: guarded ``/cmd_vel`` -> Go2 Unitree DDS body, publishes
      ``/odom``, ``/imu``, ``/joint_states``.
    - ``piper_driver``: ``/piper/joint_cmd`` -> Piper CAN (JointCtrl/Gripper
      Ctrl), publishes ``/piper/joint_states`` + ``/piper/arm_status``,
      torque gate via ``/piper/enable``.

    Neither node stands the dog up nor enables arm torque on its own; those
    are operator pre-steps (or the /piper/enable service).
    """
    return LaunchDescription(
        [
            Node(
                package="go2_piper_driver",
                executable="go2_bridge_node",
                name="go2_bridge",
                output="screen",
            ),
            Node(
                package="go2_piper_driver",
                executable="piper_driver_node",
                name="piper_driver",
                output="screen",
            ),
        ]
    )
