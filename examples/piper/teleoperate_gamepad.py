import os
os.environ["RUST_LOG"] = "error"
os.environ["WGPU_BACKEND"] = "gl"

import time
import argparse

from ubrobot.robots.piper.robot import PiperRobot
from ubrobot.robots.piper.config import PiperRobotConfig
from ubrobot.teleoperators.piper.teleoperator import PiperTeleoperator
from ubrobot.teleoperators.piper.config import PiperTeleoperatorConfig, PiperKeyboardTeleopConfig
from ubrobot.teleoperators.piper.teleop_keyboard import PiperKeyboardTeleop
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig, Cv2Rotation
from lerobot.processor import make_default_processors
from lerobot.scripts.lerobot_record import record_loop # record_loop can be used for teleoperation without recording
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
# from lerobot.utils.visualization_utils import init_rerun # Removed rerun import

# --- Script Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--teleop_type", type=str, default="gamepad", choices=["gamepad", "keyboard"], help="Type of teleoperation to use.")
args = parser.parse_args()


FPS = 30 # Frames per second for teleoperation loop
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
    cameras={
        #"camera_1": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=FPS, rotation=Cv2Rotation.NO_ROTATION),
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


# --- Initialize Robot and Teleoperator ---
robot = PiperRobot(robot_config)

# --- Processors (for potential data visualization, though not recording here) ---
teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

# --- Connect to Robot and Teleop ---
robot.connect()
teleoperator.connect()

# --- Initialize Keyboard Listener and Rerun Visualization ---
listener, events = init_keyboard_listener()
# init_rerun(session_name="piper_teleoperate") # Removed rerun call

#print("----------------", robot.is_connected, teleoperator.is_connected)
if not robot.is_connected or not teleoperator.is_connected:
    raise ValueError("Robot or teleop is not connected!")

print("Starting teleoperation loop...")
log_say("Teleoperating Piper robot.")

try:
    while not events["stop_recording"]:
        t0 = time.perf_counter()

        # Get robot observation
        observation = robot.get_observation()

        # Get teleop action
        action = teleoperator.get_action()

        # Send action to robot
        robot.send_action(action)

        # Visualize (if Rerun is configured)
        # log_rerun_data(observation=observation, action=action) # Removed rerun call

        # Wait to maintain desired frame rate
        time_to_wait = 1.0 / FPS - (time.perf_counter() - t0)
        if time_to_wait > 0:
            time.sleep(time_to_wait)

except KeyboardInterrupt:
    print("\nTeleoperation interrupted by user.")
finally:
    # --- Clean Up ---
    log_say("Stopping teleoperation.")
    robot.disconnect()
    teleoperator.disconnect()
    if listener is not None:
        listener.stop()
    print("Teleoperation finished.")
