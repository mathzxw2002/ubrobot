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
import os
os.environ["RUST_LOG"] = "error"
os.environ["WGPU_BACKEND"] = "gl" 

from datetime import datetime

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor import make_default_processors
from src.ubrobot.robots.so101_follower.so101_client import SO101Client, SO101ClientConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

NUM_EPISODES = 10
FPS = 30
EPISODE_TIME_SEC = 600
TASK_DESCRIPTION = "Grab pens and place into pen holder."
# IMPORTANT: Set this to the path of the policy you want to evaluate.
# Can be a local path or a Hugging Face Hub repo ID.
HF_MODEL_ID = "/home/sany/060000/pretrained_model"
# IMPORTANT: Set this to your desired local path for saving the evaluation dataset.
LOCAL_EVAL_PATH = f"/home/sany/so101_evaluations_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# --- Robot Configuration ---
# IMPORTANT: Make sure to replace the remote_ip with the IP of the machine running the so101_host
robot_config = SO101ClientConfig(remote_ip="192.168.18.159", id="my_so101")

# --- Initialize Robot ---
robot = SO101Client(robot_config)

# --- Create Policy and Processors from Factory ---
# This will dynamically load the correct policy and processors based on the model's config file.
policy_cfg = PreTrainedConfig.from_pretrained(HF_MODEL_ID, local_files_only=False)
policy_cfg.pretrained_path = HF_MODEL_ID

# --- Configure the Dataset Features ---
action_features = hw_to_dataset_features(robot.action_features, ACTION)
obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
dataset_features = {**action_features, **obs_features}

# --- Create the Dataset for Evaluation ---
# A dataset object is still needed to get metadata like stats for the processors.
dataset = LeRobotDataset.create(
    repo_id=LOCAL_EVAL_PATH, # Use local path as repo_id for local saving
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# Create policy-specific processors
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy_cfg,
    pretrained_path=policy_cfg.pretrained_path,
    dataset_stats=dataset.meta.stats,
    preprocessor_overrides={"device_processor": {"device": policy_cfg.device}},
)

# Create the policy instance
policy = make_policy(policy_cfg, ds_meta=dataset.meta)

# --- Connect to Robot ---
robot.connect()

# Create default processors for teleoperation and robot control
teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

# --- Initialize Keyboard Listener and Rerun Visualization ---
listener, events = init_keyboard_listener()
init_rerun(session_name="so101_evaluate")

if not robot.is_connected:
    raise ValueError("Robot is not connected!")

print("Starting evaluate loop...")
recorded_episodes = 0
while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"Running inference, recording eval episode {recorded_episodes} of {NUM_EPISODES}")

    # --- Main Record Loop (in inference mode) ---
    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
    )

    # --- Reset the Environment (manual teleoperation) ---
    if not events["stop_recording"] and (
        (recorded_episodes < NUM_EPISODES - 1) or events["rerecord_episode"]
    ):
        log_say("Reset the environment by teleoperating the robot to its initial state.")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            control_time_s=EPISODE_TIME_SEC,
            single_task="Resetting environment",
            display_data=True,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )

    if events["rerecord_episode"]:
        log_say("Re-record episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    # --- Save Episode ---
    dataset.save_episode()
    recorded_episodes += 1

# --- Clean Up ---
log_say("Stop recording")
robot.disconnect()
listener.stop()

# Finalize dataset without pushing to hub
dataset.finalize()

print(f"Evaluation finished. Data saved locally at: {os.path.abspath(LOCAL_EVAL_PATH)}")
