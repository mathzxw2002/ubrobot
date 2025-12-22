import os
os.environ["RUST_LOG"] = "error"
os.environ["WGPU_BACKEND"] = "gl" 

from datetime import datetime

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.robots.piper.robot import PiperRobot
from lerobot.robots.piper.config import PiperRobotConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig, Cv2Rotation
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
# from lerobot.utils.visualization_utils import init_rerun # Rerun import commented out

NUM_EPISODES = 10
FPS = 30
EPISODE_TIME_SEC = 600
TASK_DESCRIPTION = "Task description for Piper robot."
# IMPORTANT: Set this to the path of the policy you want to evaluate.
HF_MODEL_ID = "/home/sany/robot/lerobot/lerobot/outputs/test/checkpoints/080000/pretrained_model"
LOCAL_EVAL_PATH = f"./data/piper_evaluations_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# --- Robot Configuration ---
robot_config = PiperRobotConfig(
    follower_arm={
        "main": {
            "can_name": "can0",
            "motors": {
                'joint_1': [1, 'agilex_piper'],
                'joint_2': [2, 'agilex_piper'],
                'joint_3': [3, 'agilex_piper'],
                'joint_4': [4, 'agilex_piper'],
                'joint_5': [5, 'agilex_piper'],
                'joint_6': [6, 'agilex_piper'],
                'gripper': [7, 'agilex_piper']
            }
        }
    },
    cameras={ # Re-enabled cameras
        "wrist": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=FPS, rotation=Cv2Rotation.NO_ROTATION),
    }
)

# --- Initialize Robot ---
robot = PiperRobot(robot_config)

# --- Create Policy and Processors from Factory ---
policy_cfg = PreTrainedConfig.from_pretrained(HF_MODEL_ID, local_files_only=True)
policy_cfg.pretrained_path = HF_MODEL_ID

# --- Configure the Dataset Features ---
action_features = hw_to_dataset_features(robot.action_features, ACTION)
obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
dataset_features = {**action_features, **obs_features}

# --- Create the Dataset for Evaluation ---
dataset = LeRobotDataset.create(
    repo_id=LOCAL_EVAL_PATH,
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

# --- Initialize Keyboard Listener ---
listener, events = init_keyboard_listener()
# init_rerun(session_name="piper_evaluate") # Rerun call commented out

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
        display_data=False, # Set to False to avoid display issues
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
            display_data=False, # Set to False to avoid display issues
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

dataset.finalize()

print(f"Evaluation finished. Data saved locally at: {os.path.abspath(LOCAL_EVAL_PATH)}")