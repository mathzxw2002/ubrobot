from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description() -> LaunchDescription:
    """Bring up the Go2+Piper hardware driver container.

    - ``go2_bridge``: guarded ``/cmd_vel`` -> Go2 Unitree DDS body, publishes
      ``/odom``, ``/imu``, ``/joint_states``.
    - ``piper_driver``: ``/piper/joint_cmd`` -> Piper CAN (JointCtrl/Gripper
      Ctrl), publishes ``/piper/joint_states`` + ``/piper/arm_status``,
      torque gate via ``/piper/enable``.

    Neither node stands the dog up nor enables arm torque on its own; those
    are operator pre-steps (or the /piper/enable service).

    The console_scripts for an ament_python package install to ``bin/``, not
    the ``libexec`` dir that ``Node(executable=)`` resolves, so we execute
    them by absolute path (robust across install layouts).
    """
    return LaunchDescription(
        [
            ExecuteProcess(
                cmd=["/opt/go2_ws/bin/go2_bridge_node"],
                name="go2_bridge",
                output="screen",
            ),
            ExecuteProcess(
                cmd=["/opt/go2_ws/bin/piper_driver_node"],
                name="piper_driver",
                output="screen",
            ),
        ]
    )
