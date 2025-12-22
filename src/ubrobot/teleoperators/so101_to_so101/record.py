# !/usr/bin/env python

# Set environment variable and logging to suppress wgpu warnings.
import os
os.environ["RUST_LOG"] = "error"
os.environ["WGPU_BACKEND"] = "gl" 

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

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.robots.so101_follower.so101_client import SO101Client, SO101ClientConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.so100_leader import SO100Leader, SO100LeaderConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

NUM_EPISODES = 10
FPS = 30
EPISODE_TIME_SEC = 15
RESET_TIME_SEC = 4
TASK_DESCRIPTION = "Grab pens and place into pen holder."
# IMPORTANT: Set this to your desired local path for saving the dataset.
LOCAL_DATASET_PATH = "./data/7"
# --- Robot and Teleop Configurations ---
# IMPORTANT: Make sure to replace the remote_ip with the IP of the machine running the so101_host11
robot_config = SO101ClientConfig(remote_ip="10.88.228.133", id="my_so100")
# IMPORTANT: Make sure to replace the port with the correct one for your leader arm.
leader_arm_config = SO100LeaderConfig(port="COM3", id="my_awesome_leader_arm1")

# --- Initialize Robot and Teleoperator ---
robot = SO101Client(robot_config)
leader_arm = SO100Leader(leader_arm_config)

# TODO(Steven): Update this example to use pipelines
teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

# --- Configure the Dataset Features ---
action_features = hw_to_dataset_features(robot.action_features, ACTION)
obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
dataset_features = {**action_features, **obs_features}

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id="local-so101-recordings", # A dummy repo_id is still needed, but data will be saved locally
    root=LOCAL_DATASET_PATH,
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# --- Connect to Robot and Teleop ---
# To connect, you should already have the host script running on the SO101 robot:
# `python -m lerobot.robots.so101_follower.so101_host --robot.id=my_so101 --robot.port=/path/to/robot/port`
robot.connect()
leader_arm.connect()

# --- Initialize Keyboard Listener and Rerun Visualization ---
listener, events = init_keyboard_listener()
init_rerun(session_name="so101_record")

if not robot.is_connected or not leader_arm.is_connected:
    raise ValueError("Robot or teleop is not connected!")

print("DEBUG: Starting record loop...")
recorded_episodes = 0
while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
    print(f"DEBUG: Main loop start. Episode {recorded_episodes}. Stop event: {events['stop_recording']}")
    log_say(f"Recording episode {recorded_episodes}")

    # --- Main Record Loop ---
    print("DEBUG: Entering main record_loop.")
    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        dataset=dataset,
        teleop=[leader_arm],  # Only the leader arm is used for teleoperation
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
    )
    print("DEBUG: Exited main record_loop.")

    # --- Reset the Environment ---
    print("DEBUG: Checking if reset is needed.")
    if not events["stop_recording"] and (
        (recorded_episodes < NUM_EPISODES - 1) or events["rerecord_episode"]
    ):
        log_say("Reset the environment")
        print("DEBUG: Entering reset record_loop.")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop=[leader_arm],
            control_time_s=RESET_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )
        print("DEBUG: Exited reset record_loop.")

    if events["rerecord_episode"]:
        log_say("Re-record episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        print("DEBUG: Re-recording, continuing to next loop iteration.")
        continue

    # --- Save Episode ---
    print("DEBUG: Saving episode...")
    dataset.save_episode()
    print("DEBUG: Episode saved.")
    recorded_episodes += 1
    print(f"DEBUG: recorded_episodes incremented to {recorded_episodes}.")
    print(f"DEBUG: Main loop end. Episode {recorded_episodes - 1}. Stop event: {events['stop_recording']}")

# --- Clean Up ---
log_say("Stop recording")
robot.disconnect()
leader_arm.disconnect()
listener.stop()

dataset.finalize()

