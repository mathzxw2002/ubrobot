import time
import copy
import numpy as np
import datetime

from ubrobot.robots.unitree_go2_robot import UnitreeGo2Robot
from PIL import Image as PIL_Image
from .controllers import Mpc_controller, PID_controller
from thread_utils import ReadWriteLock

import threading
from ubrobot.robots.vlm import RobotVLM
from ubrobot.robots.nav import RobotAction, RobotNav, ControlMode

from ubrobot.robots.lekiwi.config_lekiwi_base import LeKiwiConfig
from ubrobot.robots.lekiwi.lekiwi_base import LeKiwi

import cv2

import os
from ubrobot.cameras.camera_odom import CameraOdom

class Go2Manager():
    
    def __init__(self):

        # 控制模式相关
        self.policy_init = True
        self.mpc = None
        self.pid = PID_controller(Kp_trans=2.0, Kd_trans=0.0, Kp_yaw=1.5, Kd_yaw=0.0, max_v=0.6, max_w=0.5)
        self.http_idx = -1

        # nav 
        self.global_nav_instruction_str = None
        #self.nav_action = None
        self.nav_annotated_img = None

        #self.camera_odom = CameraOdom("419522070679")  #348522070565
        self.camera_odom = CameraOdom("348522070565")

        # 读写锁相关
        self.mpc_rw_lock = ReadWriteLock()

        self.odom = None
        #self.odom_queue = deque(maxlen=50)
        self.vel = None

        # vlm model
        self.vlm = RobotVLM()

        # nav model
        self.nav = RobotNav()

        self.planning_thread_instance = threading.Thread(target=self.vln_planning_thread, daemon=True)
        #self.robot_arm_serving_thread_instance = threading.Thread(target=self.robot_arm_serving_thread, daemon=True)
        
        # unitree go2 dog
        #self.go2client = UnitreeGo2Robot()

        # robot arm config
        #self.cfg = PiperServerConfig()
        #self.robot_arm = PiperHost(self.cfg.host)

        self.lekiwi_cfg = LeKiwiConfig()
        self.lekiwi_base = LeKiwi(self.lekiwi_cfg)
        self.lekiwi_base.connect()
    
    def get_observation(self):
        rgb_image, depth_image, self.odom, self.vel = self.camera_odom.get_odom_observation()
        return rgb_image, depth_image, self.odom

    def vln_planning_thread(self):
        FPS = 30
        while True:
            t0 = time.time()
            rgb_image, depth, odom_infer = self.get_observation()

            nav_action, self.nav_annotated_img = self.get_nav_action_by_usrinstruction(self.policy_init, self.http_idx, rgb_image, depth, self.global_nav_instruction_str, odom_infer)
            
            # TODO if get STOP action signal, stop, waiting for next instruction
            if nav_action is not None:
                if nav_action.stop_cmd:
                    self.http_idx = -1
                    self.policy_init = True
                    self.move(0.0, 0.0, 0.0)
                else:
                    self.http_idx += 1
                    self.policy_init = False
                    print("get action...", nav_action.actions)
                    # send action
                    self.send_action(nav_action)
            '''else:
                #print("nav action is none", self.global_nav_instruction_str)
                if self.global_nav_instruction_str is None:
                    # if nav_action is None, stop first
                    #print("entering stop action....")
                    self.http_idx = -1
                    self.policy_init = True
                    self.move(0.0, 0.0, 0.0)'''
            # sleep
            time.sleep(max(0, 1.0 / FPS - (time.time() - t0)))
    
    def nav_by_user_instruction(self, instruction: str):
        # TODO implement this by LLM
        if instruction == "stop" or instruction == "STOP":
            self.global_nav_instruction_str = None

            print("start stopping lekiwi base...")
            self.move(0, 0, 0)
            self.lekiwi_base.stop_base()
        else:
            self.global_nav_instruction_str = instruction
        self.http_idx = -1
        self.policy_init = True
       
    def get_nav_action_by_usrinstruction(self, policy_init, http_idx, rgb_image, depth, instruction, odom):
        nav_action = None
        vis_annotated_img = None
        if odom is not None and rgb_image is not None and depth is not None and instruction is not None:
            if instruction == "stop"  or instruction == "STOP": # TODO implement this by LLM
                nav_action = RobotAction()
                nav_action.stop_cmd = True
                vis_annotated_img = rgb_image
            else:
                start = time.time()
                nav_action, vis_annotated_img = self.nav._dual_sys_eval(policy_init, http_idx, rgb_image, depth, instruction, odom)
                print(f"idx: {http_idx} step in get_nav_action_by_usrinstruction() cost {time.time() - start}")
        else:
            nav_action = None
            vis_annotated_img = rgb_image
            #print(f"odom: {odom}, rgb_image: {rgb_image}, instruction:{instruction}")
        return nav_action, vis_annotated_img
    
    def send_action(self, act):
        # first check current odom info, [x, y, yaw, v_x, w_z]
        robot_nav_current_state = list(self.odom) + list(self.vel)
        formatted_values = [f"{num:.2f}" for num in robot_nav_current_state]
        print(f"current odom ([x, y, yaw, v_x, w_z]): {' , '.join(formatted_values)}")
        if act.current_control_mode == ControlMode.MPC_Mode:
            self.mpc_rw_lock.acquire_write()
            if self.mpc is None:
                self.mpc = Mpc_controller(np.array(act.trajs_in_world))
            else:
                self.mpc.update_ref_traj(np.array(act.trajs_in_world))
            self.mpc_rw_lock.release_write()

            # MPC模式：基于轨迹的最优控制
            odom = self.odom if self.odom else None
            if self.mpc is not None and odom is not None:
                local_mpc = self.mpc
                opt_u_controls, opt_x_states = local_mpc.solve(np.array(odom))
                v, w = opt_u_controls[0, 0], opt_u_controls[0, 1]
                self.move(v, 0.0, w)
        elif act.current_control_mode == ControlMode.PID_Mode:
            odom = copy.deepcopy(self.odom) if self.odom is not None else None
            # compute homo_odom by odom
            # 计算齐次变换矩阵
            yaw = odom[2]
            R0 = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            homo_odom = np.eye(4)
            homo_odom[:2, :2] = R0
            homo_odom[:2, 3] = odom[:2]
            
            vel = copy.deepcopy(self.vel) if self.vel is not None else None
            if homo_odom is not None and vel is not None and act.homo_goal is not None:
                v, w, e_p, e_r = self.pid.solve(homo_odom, act.homo_goal, vel)
                if v < 0.0:
                    v = 0.0
                self.move(v, 0.0, w)
    
    def robot_arm_serving_thread(self):
        self.robot_arm.start_serving_teleoperation(self.cfg)
    
    def start_threads(self):
        self.planning_thread_instance.start()
        #self.robot_arm_serving_thread_instance.start()
        print("✅ Go2Manager: control thread and planning thread started successfully")

    def move(self, vx, vy, vyaw):
        action = {"x.vel": vx, "y.vel": 0, "theta.vel": vyaw}
        #current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # ms precision
        #print(f"[{current_time}] receive move command [vx, vy, vyaw] {vx:.2f}, {vy:.2f}, {vyaw:.2f}")
        print("moving action...", action)
        self.lekiwi_base.send_action(action)
        #self.go2client.Move(vx, vy, vyaw) #vx, vy, vyaw
    
    def visualize_robot_observation(self):
        rgb_image, _ = self.get_robot_arm_image_observation()
        if rgb_image is None:
            return None, self.nav_annotated_img
        else:
            color_image_pil = PIL_Image.fromarray(rgb_image)
            return color_image_pil, self.nav_annotated_img

    def get_robot_arm_image_observation(self):
        observation = self.robot_arm.get_robot_arm_observation_local()
        color_image = observation["wrist"] # TODO get "wrist" from configuration, avoid hard coding
        depth_image = observation["wrist_depth"]

        #color_image_pil = PIL_Image.fromarray(color_image)
        #color_image_pil.save("./output_image.png")

        #intrin = observation["wrist_intrinsics"]

        '''tem_file_path = "./output_image.png"
        if os.path.isfile(tem_file_path):
            image_orig = cv2.imread()
            if image_orig is None:
                return None, None
            else:
                color_image = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
                depth_image = None
                return color_image, depth_image
        else:
            return None, None'''
        return color_image, depth_image
    
    # main entrance the user interaction
    def agent_response(self, instruction):
        # parse user instruction, TODO solve this by llm intent understanding
        llm_response_txt = ""
        if instruction.startswith("nav:"):
            instruction = instruction.removeprefix("nav:").strip()
            self.nav_by_user_instruction(instruction)
            llm_response_txt = "Yes, Let's Start with Command: " + instruction
        elif instruction.startswith("grasp:"):
            manipulate_img_output, _ = self.get_robot_arm_image_observation()
            user_input_txt = instruction + ". Answer shortly."
            llm_response_txt = self.vlm.reasoning_vlm_infer(manipulate_img_output, None, None, user_input_txt)
        else:
            #TODO use all seen images
            rgb_image, _, _ = self.get_observation() # this is from frontal camera, works for lekiwi base and unitree dog
            user_input_txt = instruction + ". Answer shortly."
            #instruction = "Navigate to the charging dock near the blue door"
            prompt = f"""
            You are an expert robot navigation planner. Analyze the provided image/video sequence. Identify all obstacles and the target goal. Use Chain-of-Thought reasoning to plan a safe, collision-free path that respects physical laws and social norms.

            Output Requirements:
            1. Reasoning: Describe the identified obstacles and the chosen navigation strategy.
            2. Trajectory: Provide a planned trajectory as a list of 2D/3D waypoints in [x, y] or [x, y, z] format, relative to the robot's current position (0, 0, 0).
            3. Format: Return the result strictly in JSON format.

            Goal: {user_input_txt}
            """
            llm_response_txt = self.vlm.reasoning_vlm_infer(rgb_image, None, None, prompt)

        print("============================================llm_response_txt", llm_response_txt)
        return llm_response_txt

if __name__ == "__main__":

    print("======= Starting Go2Manager Core =======")
    try:
        manager = Go2Manager()
        manager.start_threads()
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        print("======= Go2Manager Core Exited =======")
