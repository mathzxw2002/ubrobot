#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig, Cv2Rotation
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from ..config import RobotConfig


def so101_cameras_config() -> dict[str, CameraConfig]:
    return {
        #"top": OpenCVCameraConfig(
        #    index_or_path=1, fps=30, width=640, height=480, rotation=Cv2Rotation.ROTATE_180#front
        #),
        "front": OpenCVCameraConfig(
            index_or_path=1, fps=30, width=640, height=480, rotation=Cv2Rotation.NO_ROTATION#wrist
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=0, fps=30, width=640, height=480, rotation=Cv2Rotation.ROTATE_180#front
        ),
    }


@RobotConfig.register_subclass("so101_follower")
@dataclass
class SO101FollowerConfig(RobotConfig):
    # Port to connect to the arm
    port: str | None = None

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=so101_cameras_config)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False


@dataclass
class SO101HostConfig:
    """Configuration for the LeKiwi host, which runs on the robot itself."""

    # zmq port for receiving commands
    port_zmq_cmd: int = 10001
    # zmq port for sending observations
    port_zmq_observations: int = 10002
    # After this time, the host will shut down.
    connection_time_s: float = 3600
    # max frequency of the control loop
    max_loop_freq_hz: float = 30


@RobotConfig.register_subclass("so101_client")
@dataclass
class SO101ClientConfig(RobotConfig):
    """Configuration for the SO101 client, which runs on the teleop machine."""

    remote_ip: str
    port_zmq_cmd: int = 10001
    port_zmq_observations: int = 10002
    polling_timeout_ms: int = 100
    connect_timeout_s: int = 5
    cameras: dict[str, CameraConfig] = field(default_factory=so101_cameras_config)
