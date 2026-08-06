ARG ROS_BASE_IMAGE=ros:jazzy-ros-base-noble

FROM ${ROS_BASE_IMAGE} AS builder

SHELL ["/bin/bash", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-colcon-common-extensions \
        ros-jazzy-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws
# The Go2 bridge is a small rclpy node that subscribes /cmd_vel (Twist) and
# publishes /odom + /imu + /joint_states for the dock, and forwards Twist to
# the Go2 body over Unitree DDS (CycloneDDS). No unitree SDK binary is
# compiled here: the node uses the python unitree_sdk2py at runtime if the
# host SDK is mounted (see compose volumes); the read-only telemetry path
# works without it.
COPY deploy/go2-driver/go2_bridge src/go2_bridge
COPY deploy/go2-driver/launch src/go2_launch

RUN source /opt/ros/jazzy/setup.bash \
    && colcon build --merge-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF

FROM ${ROS_BASE_IMAGE} AS runtime

LABEL org.opencontainers.image.title="UBRobot Go2 Bridge" \
      org.opencontainers.image.description="ROS 2 Jazzy bridge from guarded /cmd_vel to the Go2 Unitree DDS body" \
      org.opencontainers.image.version="0.1.0"

SHELL ["/bin/bash", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ros-jazzy-geometry-msgs \
        ros-jazzy-nav-msgs \
        ros-jazzy-sensor-msgs \
        ros-jazzy-rmw-cyclonedds-cpp \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir unitree-sdk2py

COPY --from=builder /ws/install /opt/go2_ws
COPY deploy/go2-driver/cyclonedds.xml /etc/cyclonedds/cyclonedds.xml

ENV ROS_DISTRO=jazzy \
    ROS_HOME=/tmp/ros-home \
    ROS_LOG_DIR=/tmp/ros-logs \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    CYCLONEDDS_URI=/etc/cyclonedds/cyclonedds.xml \
    PYTHONPATH=/opt/go2_ws/lib/python3.12/site-packages

USER root
CMD ["ros2", "launch", "go2_launch", "go2_bringup.launch.py"]
