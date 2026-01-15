
from dataclasses import dataclass, field
from lerobot.robots.config import RobotConfig
from ubrobot.motors.piper.motors_bus import PiperMotorsBusConfig
from lerobot.cameras.configs import CameraConfig
from typing import Dict

@RobotConfig.register_subclass("piper")
@dataclass
class PiperRobotConfig(RobotConfig):
    type: str = "piper"
    follower_arm: Dict[str, PiperMotorsBusConfig] = field(default_factory=dict)
    cameras: Dict[str, CameraConfig] = field(default_factory=dict)
