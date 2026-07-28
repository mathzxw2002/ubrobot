ARG EMOS_BASE_IMAGE=ghcr.io/automatika-robotics/emos:jazzy-latest
FROM ${EMOS_BASE_IMAGE}

ARG KOMPASS_CORE_VERSION=0.8.1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libboost-system1.83.0 \
        libfcl0.7 \
        liboctomap1.9t64 \
        libompl16t64 \
        ros-jazzy-depthimage-to-laserscan \
        ros-jazzy-realsense2-camera \
        ros-jazzy-realsense2-camera-msgs \
        ros-jazzy-rtabmap-odom \
        ros-jazzy-rtabmap-ros \
    && python3 -m pip install --no-cache-dir --break-system-packages \
        "kompass-core==${KOMPASS_CORE_VERSION}" \
    && rm -rf /var/lib/apt/lists/*
