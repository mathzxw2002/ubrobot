"""Robot Edge ROS-side adapters (M6, read-only).

Importing this package must never import ``rclpy``: workstation tests and
fixture mode run without a ROS installation. The real ROS node is created
only when :func:`robot_edge.ros.context.create_ros_context` is explicitly
called in hardware mode.
"""

from robot_edge.ros.backend import RosReadonlyBackend, create_readonly_ros_backend

__all__ = ["RosReadonlyBackend", "create_readonly_ros_backend"]
