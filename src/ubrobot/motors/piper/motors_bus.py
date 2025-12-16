import time
from typing import Dict, Tuple
from piper_sdk import *
from lerobot.motors.motors_bus import MotorsBus, Motor, MotorNormMode
from dataclasses import dataclass, field

@dataclass
class PiperMotorsBusConfig:
    can_name: str = "can0"
    motors: Dict[str, Tuple[int, str]] = field(default_factory=dict)


class PiperMotorsBus(MotorsBus):


    """


        对Piper SDK的二次封装


    """


    apply_drive_mode: bool = False


    available_baudrates: list[int] = []


    default_baudrate: int = 0


    default_timeout: int = 0


    model_baudrate_table: dict[str, dict] = {}


    model_ctrl_table: dict[str, dict] = {}


    model_encoding_table: dict[str, dict] = {}


    model_number_table: dict[str, int] = {}


    model_resolution_table: dict[str, int] = {}


    normalized_data: list[str] = []





    def __init__(


        self,


        config: PiperMotorsBusConfig


    ):


        # Convert config.motors (Dict[str, Tuple[int, str]]) to Dict[str, Motor] for super().__init__
        converted_motors = {
            name: Motor(id=motor_id, model=model_name, norm_mode=MotorNormMode.IDENTITY)
            for name, (motor_id, model_name) in config.motors.items()
        }
        super().__init__(port=config.can_name, motors=converted_motors)
        self.piper = C_PiperInterface_V2(config.can_name)
        self.piper.ConnectPort()
        # Keep the original config.motors for internal use if needed, or directly use the converted_motors
        # For now, let's assume super().__init__ manages self.motors and use that.
        self.init_joint_position = [0.0, 0.0, 0.0, 0.0, 0.0, -1.85, 0.0] # [6 joints + 1 gripper] * 0.0
        self.safe_disable_position = [0.0, 0.0, 0.0, 0.0, 0.52, -1.85, 0.0]
        self.pose_factor = 1000 # 单位 0.001mm
        self.joint_factor = 57324.840764 # 1000*180/3.14， rad -> 度（单位0.001度）

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    def connect(self, enable:bool) -> bool:
        '''
            使能机械臂并检测使能状态,尝试5s,如果使能超时则退出程序
        '''
        enable_flag = False
        loop_flag = False
        # 设置超时时间（秒）
        timeout = 5
        # 记录进入循环前的时间
        start_time = time.time()
        while not (loop_flag):
            elapsed_time = time.time() - start_time
            print(f"--------------------")
            enable_list = []
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status)
            if(enable):
                enable_flag = all(enable_list)
                self.piper.EnableArm(7)
                self.piper.GripperCtrl(0,1000,0x01, 0)
            else:
                # move to safe disconnect position
                enable_flag = any(enable_list)
                self.piper.DisableArm(7)
                self.piper.GripperCtrl(0,1000,0x02, 0)
            print(f"使能状态: {enable_flag}")
            print(f"--------------------")
            if(enable_flag == enable):
                loop_flag = True
                enable_flag = True
            else:
                loop_flag = False
                enable_flag = False
            # 检查是否超过超时时间
            if elapsed_time > timeout:
                print(f"超时....")
                enable_flag = False
                loop_flag = True
                break
            time.sleep(0.5)
        resp = enable_flag
        print(f"Returning response: {resp}")
        return resp

    def apply_calibration(self):
        """
            移动到初始位置
        """
        self.write(target_joint=self.init_joint_position)

    def write(self, target_joint:list):
        """
            Joint control
            - target joint: in radians
                joint_1 (float): 关节 1角度 (-92000~92000) / 57324.840764
                joint_2 (float): 关节 2角度 -1300 ~ 90000 / 57324.840764
                joint_3 (float): 关节 3角度 2400 ~ -80000 / 57324.840764
                joint_4 (float): 关节 4角度 -90000~90000 / 57324.840764
                joint_5 (float): 关节 5角度 19000~-77000 / 57324.840764
                joint_6 (float): 关节 6角度 -90000~90000 / 57324.840764
                gripper_range: 夹爪角度 0~0.08
        """
        joint_0 = round(target_joint[0]*self.joint_factor)
        joint_1 = round(target_joint[1]*self.joint_factor)
        joint_2 = round(target_joint[2]*self.joint_factor)
        joint_3 = round(target_joint[3]*self.joint_factor)
        joint_4 = round(target_joint[4]*self.joint_factor)
        joint_5 = round(target_joint[5]*self.joint_factor)
        gripper_range = round(target_joint[6]*1000*1000)

        self.piper.MotionCtrl_2(0x01, 0x01, 100, 0x00) # joint control
        self.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
        self.piper.GripperCtrl(abs(gripper_range), 1000, 0x01, 0) # 单位 0.001°


    def read(self) -> Dict:
        """
            - 机械臂关节消息,单位0.001度
            - 机械臂夹爪消息
        """
        joint_msg = self.piper.GetArmJointMsgs()
        joint_state = joint_msg.joint_state

        gripper_msg = self.piper.GetArmGripperMsgs()
        gripper_state = gripper_msg.gripper_state

        return {
            "joint_1": joint_state.joint_1,
            "joint_2": joint_state.joint_2,
            "joint_3": joint_state.joint_3,
            "joint_4": joint_state.joint_4,
            "joint_5": joint_state.joint_5,
            "joint_6": joint_state.joint_6,
            "gripper": gripper_state.grippers_angle
        }

    def safe_disconnect(self):
        """
            Move to safe disconnect position
        """
        self.write(target_joint=self.safe_disable_position)

    def _assert_protocol_is_compatible(self, instruction_name: str) -> None:
        pass

    def _handshake(self) -> None:
        pass

    def _find_single_motor(self, motor: str, initial_baudrate: int | None) -> tuple[int, int]:
        pass

    def configure_motors(self) -> None:
        pass

    def disable_torque(self, motors: int | str | list[str] | None = None, num_retry: int = 0) -> None:
        pass

    def _disable_torque(self, motor: int, model: str, num_retry: int = 0) -> None:
        pass

    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        pass

    @property
    def is_calibrated(self) -> bool:
        return True

    def read_calibration(self) -> dict[str, any]:
        return {}

    def write_calibration(self, calibration_dict: dict[str, any], cache: bool = True) -> None:
        pass

    def _get_half_turn_homings(self, positions: dict[any, any]) -> dict[any, any]:
        return {}

    def _encode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
        return ids_values

    def _decode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
        return ids_values

    def _split_into_byte_chunks(self, value: int, length: int) -> list[int]:
        return []

    def broadcast_ping(self, num_retry: int = 0, raise_on_error: bool = False) -> dict[int, int] | None:
        return {}
    
    def _validate_motors(self) -> None:
        # Piper does not use the generic motor validation from MotorsBus base class
        # as its communication is handled directly by piper_sdk.
        pass