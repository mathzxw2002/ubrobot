import torch
import time
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardTeleopConfig
from typing import Any

class PiperKeyboardTeleop(KeyboardTeleop):
    def __init__(self, config: KeyboardTeleopConfig):
        super().__init__(config)
        self.joint_increment = 0.05 # Radians per key press
        self.gripper_increment = 0.01 # Gripper value per key press
        self.last_action_time = time.time()
        # action_delay: Time in seconds to wait between processing consecutive keyboard actions.
        # Reduce this value for faster response, but be aware it might lead to less stable movements.
        self.action_delay = 0.05 # seconds

        # Initialize current commanded positions to zeros
        self.current_commanded_positions = torch.zeros(7, dtype=torch.float32)

        # DEFINE JOINT LIMITS HERE (in radians) - Derived from piper_sdk comments and joint_factor
        # joint_factor = 57324.840764 (1000*180/pi)
        # rad = SDK_unit / joint_factor
        _joint_factor = 57324.840764
        self._joint_limits = [
            (-92000 / _joint_factor, 92000 / _joint_factor),      # Joint 1: (-1.6048, 1.6048) rad
            (-1300 / _joint_factor, 90000 / _joint_factor),      # Joint 2: (-0.0227, 1.5700) rad
            (-80000 / _joint_factor, 2400 / _joint_factor),      # Joint 3: (-1.3955, 0.0418) rad
            (-90000 / _joint_factor, 90000 / _joint_factor),     # Joint 4: (-1.5700, 1.5700) rad
            (-77000 / _joint_factor, 19000 / _joint_factor),     # Joint 5: (-1.3431, 0.3314) rad
            (-90000 / _joint_factor, 90000 / _joint_factor),     # Joint 6: (-1.5700, 1.5700) rad
            (0.0, 0.08)                                        # Gripper: (0.0, 0.08) (assuming these are already in 'user' units for the gripper)
        ]


    def get_action(self) -> dict[str, Any]:
        # Process keyboard inputs
        self._drain_pressed_keys()

        delta_joint_values = torch.zeros(7, dtype=torch.float32) # Calculate changes based on key presses

        # Only process action if enough time has passed since last action
        if (time.time() - self.last_action_time) < self.action_delay:
            # Return a flat dictionary of current positions
            return {f"joint_{i+1}.pos" if i < 6 else "gripper.pos": self.current_commanded_positions[i].item() for i in range(7)}

        # Joint 1 (Base Rotation): Q/A
        if self.current_pressed.get('q', False):
            delta_joint_values[0] += self.joint_increment
        if self.current_pressed.get('a', False):
            delta_joint_values[0] -= self.joint_increment

        # Joint 2 (Shoulder Lift): W/S
        if self.current_pressed.get('w', False):
            delta_joint_values[1] += self.joint_increment
        if self.current_pressed.get('s', False):
            delta_joint_values[1] -= self.joint_increment

        # Joint 3 (Elbow Flex): E/D
        if self.current_pressed.get('e', False):
            delta_joint_values[2] += self.joint_increment
        if self.current_pressed.get('d', False):
            delta_joint_values[2] -= self.joint_increment

        # Joint 4 (Wrist Flex): R/F
        if self.current_pressed.get('r', False):
            delta_joint_values[3] += self.joint_increment
        if self.current_pressed.get('f', False):
            delta_joint_values[3] -= self.joint_increment

        # Joint 5 (Wrist Rotate): T/G
        if self.current_pressed.get('t', False):
            delta_joint_values[4] += self.joint_increment
        if self.current_pressed.get('g', False):
            delta_joint_values[4] -= self.joint_increment

        # Joint 6 (End Effector Pitch/Yaw): Y/H
        if self.current_pressed.get('y', False):
            delta_joint_values[5] += self.joint_increment
        if self.current_pressed.get('h', False):
            delta_joint_values[5] -= self.joint_increment
        
        # Gripper: Z/X
        if self.current_pressed.get('z', False):
            delta_joint_values[6] += self.gripper_increment
        if self.current_pressed.get('x', False):
            delta_joint_values[6] -= self.gripper_increment

        # Update the commanded positions by adding the deltas
        self.current_commanded_positions += delta_joint_values

        # Clamp joint values to stay within physical limits
        for i in range(len(self.current_commanded_positions)):
            min_limit, max_limit = self._joint_limits[i]
            self.current_commanded_positions[i] = torch.clamp(
                self.current_commanded_positions[i], min_limit, max_limit
            )

        self.last_action_time = time.time()
        
        # Return a flat dictionary of current positions
        return {f"joint_{i+1}.pos" if i < 6 else "gripper.pos": self.current_commanded_positions[i].item() for i in range(7)}

    def get_teleop_events(self) -> dict[str, Any]:
        # Implement specific event mapping if needed
        # For now, it delegates to the base KeyboardTeleop implementation
        # The base class already handles ESC key for disconnect, and arrow keys for control flow.
        return super().get_teleop_events()