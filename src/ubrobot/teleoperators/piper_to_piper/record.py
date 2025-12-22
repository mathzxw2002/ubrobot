import os
os.environ["RUST_LOG"] = "error"
os.environ["WGPU_BACKEND"] = "gl"

import argparse

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
# Removed VideoEncodingManager import
from lerobot.processor import make_default_processors
from lerobot.robots.piper.robot import PiperRobot
from lerobot.robots.piper.config import PiperRobotConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.piper.teleoperator import PiperTeleoperator
from lerobot.teleoperators.piper.config import PiperTeleoperatorConfig, PiperKeyboardTeleopConfig
from lerobot.teleoperators.piper.teleop_keyboard import PiperKeyboardTeleop
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig, Cv2Rotation
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
# from lerobot.utils.visualization_utils import init_rerun # Rerun import commented out

# --- Script Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--teleop_type", type=str, default="gamepad", choices=["gamepad", "keyboard"], help="Type of teleoperation to use.")
args = parser.parse_args()

# --- Main Configuration ---
NUM_EPISODES = 1 # Changed to 2 for quicker testing based on user's output
FPS = 30 # Based on user's output
EPISODE_TIME_SEC = 5 # Based on user's output
RESET_TIME_SEC = 1
TASK_DESCRIPTION = "Task description for Piper robot."
LOCAL_DATASET_PATH = "./data/piper_dataset"
TELEOP_TYPE = args.teleop_type

# --- Robot and Teleop Configurations ---
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
        "camera_1": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=FPS, rotation=Cv2Rotation.NO_ROTATION),
        # Example with a 90 degree rotation:
        # "camera_2": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=FPS, rotation=Cv2Rotation.ROTATE_90),
    }
)

if TELEOP_TYPE == "gamepad":
    teleop_config = PiperTeleoperatorConfig()
    teleoperator = PiperTeleoperator(teleop_config)
else: # keyboard
    teleop_config = PiperKeyboardTeleopConfig()
    teleoperator = PiperKeyboardTeleop(teleop_config)


# --- Initialize Robot ---
robot = PiperRobot(robot_config)

teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

# --- Configure the Dataset Features ---
action_features = hw_to_dataset_features(robot.action_features, ACTION)
obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
dataset_features = {**action_features, **obs_features}

print(f"\n--- Debug: Dataset Features being passed to LeRobotDataset.create ---")
import json
print(json.dumps(dataset_features, indent=2))
print("---------------------------------------------------------------------\n")

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id="local-piper-recordings",
    root=LOCAL_DATASET_PATH,
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# --- Connect to Robot and Teleop ---
robot.connect()
teleoperator.connect()

# --- Debug: Test Camera Capture ---
print("\n--- Testing Camera Capture ---")
if robot.cameras:
    for cam_name, cam_obj in robot.cameras.items():
        try:
            test_frame = cam_obj.read()
            print(f"Camera '{cam_name}' captured frame with shape: {test_frame.shape}")
        except Exception as e:
            print(f"Error capturing frame from camera '{cam_name}': {e}")
else:
    print("No cameras configured for the robot.")
print("--- Camera Capture Test Finished ---\n")

# --- Initialize Keyboard Listener ---
listener, events = init_keyboard_listener()
# init_rerun(session_name="piper_record") # Removed rerun call # <--- Removed this call

if not robot.is_connected or not teleoperator.is_connected:
    raise ValueError("Robot or teleop is not connected!")

# --- Main Recording Session Loop ---
recorded_episodes = 0
while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
    # --- Improved Prompt for Recording Episode ---
    input(f"\n--- Press Enter to START recording Episode {recorded_episodes + 1}/{NUM_EPISODES} ({TASK_DESCRIPTION}) ---")
    log_say(f"Recording episode {recorded_episodes}")

    # --- Main Record Loop ---
    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        dataset=dataset,
        teleop=[teleoperator],
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True, # Set to False to avoid display issues
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
    )

    # --- Reset the Environment ---
    if not events["stop_recording"] and (
        (recorded_episodes < NUM_EPISODES - 1) or events["rerecord_episode"]
    ):
        input(f"\n--- Press Enter to RESET environment for next episode ---")
        log_say("Reset the environment")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop=[teleoperator],
            control_time_s=RESET_TIME_SEC,
            single_task=TASK_DESCRIPTION,
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
        print("\n--- Episode RE-RECORDED. Starting again ---") # Clearer message
        continue

    # --- Save Episode ---
    dataset.save_episode()
    recorded_episodes += 1
    print(f"\n--- Episode {recorded_episodes}/{NUM_EPISODES} RECORDED and SAVED. ---") # Clearer message

# --- Clean Up ---
log_say("Stop recording")
robot.disconnect()
teleoperator.disconnect()
if listener is not None:
    listener.stop()


dataset.finalize() # Ensure this is called
print("\n--- Data recording session FINISHED. ---")