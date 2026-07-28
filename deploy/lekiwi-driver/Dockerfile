# syntax=docker/dockerfile:1.7

ARG ROS_BASE_IMAGE=ros:jazzy-ros-base-noble

FROM ${ROS_BASE_IMAGE} AS builder

SHELL ["/bin/bash", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-colcon-common-extensions \
        ros-jazzy-hardware-interface \
        ros-jazzy-xacro \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws
COPY ros_depends_ws/src/lekiwi_description src/lekiwi_description
COPY ros_depends_ws/src/lekiwi_bringup src/lekiwi_bringup

RUN source /opt/ros/jazzy/setup.bash \
    && colcon build --merge-install --cmake-args -DCMAKE_BUILD_TYPE=Release

FROM ${ROS_BASE_IMAGE} AS runtime

LABEL org.opencontainers.image.title="UBRobot LeKiwi Base Driver" \
      org.opencontainers.image.description="Safe ROS 2 Jazzy ros2_control runtime for LeKiwi" \
      org.opencontainers.image.version="0.1.0-mock"

SHELL ["/bin/bash", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ros-jazzy-geometry-msgs \
        ros-jazzy-joint-state-broadcaster \
        ros-jazzy-omni-wheel-drive-controller \
        ros-jazzy-rmw-fastrtps-cpp \
        ros-jazzy-robot-state-publisher \
        ros-jazzy-ros2-control \
        ros-jazzy-ros2-controllers \
        ros-jazzy-xacro \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 --shell /bin/bash lekiwi \
    && usermod --append --groups dialout lekiwi

COPY --from=builder /ws/install /opt/lekiwi_ws
COPY deploy/lekiwi-driver/entrypoint.sh /usr/local/bin/lekiwi-entrypoint.sh
COPY deploy/lekiwi-driver/healthcheck.sh /usr/local/bin/lekiwi-healthcheck.sh

RUN chmod 0755 /usr/local/bin/lekiwi-entrypoint.sh /usr/local/bin/lekiwi-healthcheck.sh \
    && mkdir -p /tmp/ros-logs \
    && chown lekiwi:lekiwi /tmp/ros-logs

ENV ROS_DISTRO=jazzy \
    ROS_LOG_DIR=/tmp/ros-logs

USER lekiwi
ENTRYPOINT ["/usr/local/bin/lekiwi-entrypoint.sh"]
CMD ["ros2", "launch", "lekiwi_bringup", "lekiwi_driver.launch.py", "hardware_mode:=mock"]
