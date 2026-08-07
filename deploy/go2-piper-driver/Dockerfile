ARG ROS_BASE_IMAGE=ros:jazzy-ros-base-noble

FROM ${ROS_BASE_IMAGE} AS builder

SHELL ["/bin/bash", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-colcon-common-extensions \
        ros-jazzy-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws
# Go2+Piper hardware driver package: go2_bridge_node (guarded /cmd_vel ->
# Go2 Unitree DDS body) and piper_driver_node (/piper/joint_cmd -> Piper CAN).
# No unitree SDK binary is compiled here: the go2 node uses the python
# unitree_sdk2py at runtime; the piper node uses piper_sdk (pip-installed in
# the runtime stage).
COPY deploy/go2-piper-driver/go2_piper_driver src/go2_piper_driver
COPY deploy/go2-piper-driver/launch src/go2_launch

RUN source /opt/ros/jazzy/setup.bash \
    && colcon build --merge-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF

FROM ${ROS_BASE_IMAGE} AS runtime

LABEL org.opencontainers.image.title="UBRobot Go2+Piper Driver" \
      org.opencontainers.image.description="ROS 2 Jazzy hardware driver: /cmd_vel to Go2 DDS body, Piper arm CAN commands" \
      org.opencontainers.image.version="0.1.0"

SHELL ["/bin/bash", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ros-jazzy-geometry-msgs \
        ros-jazzy-nav-msgs \
        ros-jazzy-sensor-msgs \
        ros-jazzy-std-srvs \
        ros-jazzy-rmw-cyclonedds-cpp \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir unitree-sdk2py piper-sdk

COPY --from=builder /ws/install /opt/go2_ws
COPY deploy/go2-piper-driver/cyclonedds.xml /etc/cyclonedds/cyclonedds.xml

ENV ROS_DISTRO=jazzy \
    ROS_HOME=/tmp/ros-home \
    ROS_LOG_DIR=/tmp/ros-logs \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    CYCLONEDDS_URI=/etc/cyclonedds/cyclonedds.xml \
    PYTHONPATH=/opt/go2_ws/lib/python3.12/site-packages

USER root
CMD ["ros2", "launch", "go2_launch", "go2_piper_bringup.launch.py"]
