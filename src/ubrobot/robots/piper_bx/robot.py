import time
import torch
from dataclasses import replace

from ubrobot.motors.piper.motors_bus import PiperMotorsBus, PiperMotorsBusConfig
from lerobot.robots.robot import Robot
from lerobot.robots.config import RobotConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from ubrobot.robots.piper.config import PiperRobotConfig

class PiperRobot(Robot):
    config_class = PiperRobotConfig
    name = "piper"
    def __init__(self, config: RobotConfig | None = None, **kwargs):
        if config is None:
            # TODO: This should be PiperRobotConfig, which needs to be defined
            config = RobotConfig(type="piper")
        # Overwrite config arguments using kwargs
        self.config = replace(config, **kwargs)
        self._is_connected = False
        self.robot_type = self.config.type

        # build cameras
        self.cameras = make_cameras_from_configs(self.config.cameras)

        # build piper motors
        motors_bus_config = PiperMotorsBusConfig(**self.config.follower_arm["main"])
        self.arm = PiperMotorsBus(motors_bus_config)

        self.logs = {}

    @property
    def observation_features(self) -> dict:
        cam_ft = {
            cam_key: (cam.height, cam.width, 3)
            for cam_key, cam in self.cameras.items()
        }
        
        motor_ft = {f"{name}.pos": float for name in self.arm.motor_names}
        
        return {**cam_ft, **motor_ft}

    @property
    def action_features(self) -> dict:
        return {f"{name}.pos": float for name in self.arm.motor_names}

    @property
    def is_connected(self) -> bool:
        return self._is_connected
    
    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self) -> None:
        """Connect piper and cameras"""
        if self._is_connected:
            raise DeviceAlreadyConnectedError(
                "Piper is already connected. Do not run `robot.connect()` twice."
            )

        # connect piper
        self.arm.connect(enable=True)
        print("piper conneted")

        # connect cameras
        for name in self.cameras:
            self.cameras[name].connect()
            # TODO: This logic is flawed, if one camera fails to connect, is_connected will be False
            # self.is_connected = self.is_connected and self.cameras[name].is_connected
            print(f"camera {name} conneted")

        print("All connected")
        self._is_connected = True

        self.run_calibration()

    def disconnect(self) -> None:
        """move to home position, disenable piper and cameras"""
        # disconnect piper
        self.arm.safe_disconnect()
        print("piper disable after 5 seconds")
        time.sleep(5)
        self.arm.connect(enable=False)

        # disconnect cameras
        if len(self.cameras) > 0:
            for cam in self.cameras.values():
                cam.disconnect()

        self._is_connected = False
    
    def run_calibration(self):
        """move piper to the home position"""
        if not self._is_connected:
            raise ConnectionError()

        self.arm.apply_calibration()

    def send_action(self, action: dict) -> dict: # Updated type hint to dict
        """Write the predicted actions from policy to the motors"""
        if not self._is_connected:
            raise DeviceNotConnectedError(
                "Piper is not connected. You need to run `robot.connect()`."
            )

        # Reconstruct the list of joint values in the correct order
        # The order is determined by self.arm.motor_names
        target_joints = [action[name] for name in self.action_features]
        
        self.arm.write(target_joints)

        return action # Return the original action dict

    def get_observation(self) -> dict:
        """capture current images and joint positions"""
        if not self._is_connected:
            raise DeviceNotConnectedError(
                "Piper is not connected. You need to run `robot.connect()`."
            )

        # read current joint positions
        before_read_t = time.perf_counter()
        state = self.arm.read()  # 6 joints + 1 gripper
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        state = {f"{name}.pos": val for name, val in state.items()}

        # read images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            # images[name] = torch.from_numpy(images[name]) # Removed torch conversion
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionnaries and format to pytorch
        obs_dict = {**state, **images}
        return obs_dict

    def __del__(self):
        if self._is_connected:
            self.disconnect()