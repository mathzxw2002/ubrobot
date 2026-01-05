import pygame
import threading
import time
from typing import Dict, Any

from lerobot.teleoperators.teleoperator import Teleoperator

class PiperTeleoperator(Teleoperator):
    name = "piper"
    def __init__(self, config):
        super().__init__(config)
        self.speed_level = 0
        self.speed_multipliers = [1, 2, 3]
        # Initialize pygame and joystick
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise Exception("No gamepad detected.")
        
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        
        # Initialize joint and gripper states
        self.joints = [0.0] * 6  # 6 arm joints
        self.gripper = 0.0
        self.speeds = [0.0] * 6
        self.gripper_speed = 0.0
        
        # Define joint limits (in radians)
        _joint_factor = 57324.840764
        # self.joint_limits = [
        #     (-92000 / _joint_factor, 92000 / _joint_factor),
        #     (-1300 / _joint_factor, 90000 / _joint_factor),
        #     (-80000 / _joint_factor, 2400 / _joint_factor),
        #     (-90000 / _joint_factor, 90000 / _joint_factor),
        #     (-77000 / _joint_factor, 19000 / _joint_factor),
        #     (-90000 / _joint_factor, 90000 / _joint_factor),
        # ]
        self.joint_limits = [
            (-2.63, 2.63),
            (0, 3),#大概，不好测
            (-2.95, 0.043),
            (-1.774, 1.75),
            (-1.24, 0.52),
            (-3.27, 0.39),
        ]
        self.running = False
        self.thread = None

    def connect(self):
        self.running = True
        self.thread = threading.Thread(target=self.update_joints)
        self.thread.start()

    def disconnect(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()
        pygame.quit()

    def update_joints(self):
        while self.running:
            pygame.event.pump()
            #设置重置
            reset_btn = self.joystick.get_button(9)
            if reset_btn:
                self.reset()
                time.sleep(0.2)
            #切换速度    
            speed_toggle_btn = self.joystick.get_button(10)
            if speed_toggle_btn:
                self.speed_level = (self.speed_level + 1) % 3
                time.sleep(0.25)
            m = self.speed_multipliers[self.speed_level]
            
            # Get joystick and button inputs
            left_x = -self.joystick.get_axis(0)
            left_y = -self.joystick.get_axis(1)
            right_x = -self.joystick.get_axis(2)
            right_y = self.joystick.get_axis(3)
            
            # Deadzone
            if abs(left_x) < 0.40: left_x = 0.0
            if abs(left_y) < 0.40: left_y = 0.0
            if abs(right_x) < 0.40: right_x = 0.0
            if abs(right_y) < 0.40: right_y = 0.0
            
            if self.joystick.get_axis(4) > 0:
                circle = True
            else:
                circle = False
                
            if self.joystick.get_axis(5) > 0:
                cross = True
            else:
                cross = False    
            
            up = self.joystick.get_button(12)
            down = self.joystick.get_button(11)
            left = self.joystick.get_button(13)
            right = self.joystick.get_button(14)
            

            # Map inputs to speeds
            self.speeds[0] = left_x * 0.01 * m
            self.speeds[1] = left_y * 0.01 * m
            self.speeds[2] = right_x * 0.01 * m
            self.speeds[4] = right_y * 0.01 * m
            self.speeds[3] = -(0.01 if up else (-0.01 if down else 0.0)) * m
            self.speeds[5] = (0.01 if right else (-0.01 if left else 0.0)) * m
            self.gripper_speed = (0.01 if circle else (-0.01 if cross else 0.0)) * m
            
            # Integrate speeds to positions
            for i in range(6):
                self.joints[i] += self.speeds[i]
            self.gripper += self.gripper_speed
            
            # Clamp joint values
            for i in range(6):
                min_val, max_val = self.joint_limits[i]
                self.joints[i] = max(min_val, min(max_val, self.joints[i]))

            self.gripper = max(0.0, min(0.08, self.gripper))
            time.sleep(0.02)

    
    def get_action(self) -> Dict[str, Any]:
        if not self.running:
            raise Exception("Gamepad teleoperator is not connected.")
            
        motor_names = [f"joint_{i+1}" for i in range(6)] + ["gripper"]
        positions = self.joints + [self.gripper]
        return {f"{name}.pos": pos for name, pos in zip(motor_names, positions)}

    def is_connected(self):
        return self.running
        
    def reset(self):
        self.joints = [0.0, 0.0, 0.0, 0.0, 0.0, -1.85]
        self.gripper = 0.0
        self.speeds = [0.0] * 6
        self.gripper_speed = 0.0

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass
    def feedback_features(self) -> dict:
        pass
    def configure(self) -> None:
        pass
        
    @property
    def action_features(self) -> dict:
         return {f"{name}.pos": float for name in (f"joint_{i+1}" for i in range(6)) + ("gripper",)}

    def get_teleop_events(self) -> dict[str, Any]:
        # Placeholder for events, can be expanded
        return {}
