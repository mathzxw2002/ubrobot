# !/usr/bin/env python

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
import os
os.environ["RUST_LOG"] = "error"
os.environ["WGPU_BACKEND"] = "gl" 
import time

from lerobot.robots.so101_follower.so101_client import SO101Client, SO101ClientConfig
from lerobot.teleoperators.so100_leader import SO100Leader, SO100LeaderConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30

# Create the robot and teleoperator configurations
# IMPORTANT: Make sure to replace the remote_ip with the IP of the machine running the so101_host
robot_config = SO101ClientConfig(remote_ip="10.88.228.133", id="my_so101")
teleop_arm_config = SO100LeaderConfig(port="COM3", id="my_awesome_leader_arm1")

# Initialize the robot and teleoperator
robot = SO101Client(robot_config)
leader_arm = SO100Leader(teleop_arm_config)

# Connect to the robot and teleoperator
# To connect you already should have this script running on the SO101 robot: `python -m lerobot.robots.so101_follower.so101_host --robot.id=my_so101`
robot.connect()
leader_arm.connect()

# Init rerun viewer
init_rerun(session_name="so101_teleop_networked")

if not robot.is_connected or not leader_arm.is_connected:
    raise ValueError("Robot or teleop is not connected!")

print("Starting teleop loop...")
while True:
    t0 = time.perf_counter()

    # Get robot observation
    observation = robot.get_observation()

    # Get teleop action
    action = leader_arm.get_action()

    # Send action to robot
    _ = robot.send_action(action)

    # Visualize
    log_rerun_data(observation=observation, action=action)

    busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
