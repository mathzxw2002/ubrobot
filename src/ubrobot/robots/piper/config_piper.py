from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig, Cv2Rotation
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from lerobot.cameras import ColorMode

from lerobot.robots import RobotConfig


def piper_cameras_config() -> dict[str, CameraConfig]:
    return {
        #"wrist": OpenCVCameraConfig(
        #    index_or_path="/dev/video2", fps=30, width=480, height=640, rotation=Cv2Rotation.ROTATE_90
        #),

        "wrist": RealSenseCameraConfig(
            serial_number_or_name="336222070923", # Replace with actual SN
            fps=30,
            width=640,
            height=480,
            color_mode=ColorMode.BGR, # Request BGR output
            rotation=Cv2Rotation.NO_ROTATION,
            use_depth=True
        ),
    }

@RobotConfig.register_subclass("piper")
@dataclass
class PiperConfig(RobotConfig):
    can_interface: str = "can0"
    bitrate: int = 1_000_000
    # Piper SDK returns 6 joints; keep order stable
    joint_names: list[str] = field(default_factory=lambda: [f"joint_{i+1}" for i in range(6)])
    # Optional sign flips applied symmetrically to obs/actions (length must match joints)
    joint_signs: list[int] = field(default_factory=lambda: [-1, 1, 1, -1, 1, -1])
    # Allow teleop joints (e.g., SO101) to reference Piper joints directly by name
    joint_aliases: dict[str, str] = field(
        default_factory=lambda: {
            "shoulder_pan": "joint_1",
            "shoulder_lift": "joint_2",
            "elbow_flex": "joint_3",
            "wrist_flex": "joint_5",
            "wrist_roll": "joint_6",
        }
    )
    # Expose gripper as "gripper.pos" in mm if True
    include_gripper: bool = True
    # Optional cameras; leave empty when not used
    '''cameras: dict[str, CameraConfig] = field(
        default_factory=
            lambda: {
            "wrist": OpenCVCameraConfig(
                index_or_path=4, 
                width=640, 
                height=480, 
                fps=30, 
                fourcc="MJPG"
            )
        }
    )'''
    cameras: dict[str, CameraConfig] = field(
        default_factory=
            lambda: {
            "wrist": RealSenseCameraConfig(
                serial_number_or_name="336222070923", # Replace with actual SN
                fps=30,
                width=640,
                height=480,
                color_mode=ColorMode.BGR, # Request BGR output
                rotation=Cv2Rotation.NO_ROTATION,
                use_depth=True
            )
        }
    )
    
    #cameras: dict[str, CameraConfig] = field(
    #    default_factory=dict
    #)
    # When False, expose normalized [-100,100] joint percents; when True, degrees/mm
    use_degrees: bool = False
    # Timeout in seconds to wait for SDK EnablePiper during connect
    enable_timeout: float = 5.0

@dataclass
class PiperHostConfig:
    # Network Configuration
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    # Duration of the application (set to 10h)
    connection_time_s: int = 36000

    # Watchdog: stop the robot if no command is received for over 0.5 seconds (set to 10h).
    watchdog_timeout_ms: int = 36000000

    # If robot jitters decrease the frequency and monitor cpu load with `top` in cmd
    max_loop_freq_hz: int = 30

@RobotConfig.register_subclass("piper_client")
@dataclass
class PiperClientConfig(RobotConfig):
    # Network Configuration
    remote_ip: str
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            # Movement
            "forward": "w",
            "backward": "s",
            "left": "a",
            "right": "d",
            "rotate_left": "z",
            "rotate_right": "x",
            # Speed control
            "speed_up": "r",
            "speed_down": "f",
            # quit teleop
            "quit": "q",
        }
    )

    cameras: dict[str, CameraConfig] = field(default_factory=piper_cameras_config)
    
    polling_timeout_ms: int = 15
    connect_timeout_s: int = 5
