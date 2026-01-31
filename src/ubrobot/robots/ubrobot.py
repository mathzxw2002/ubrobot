import time
from datetime import datetime
import copy
from collections import deque
import numpy as np

import sys
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

from ubrobot.robots.piper.piper_host import PiperHost, PiperServerConfig

from ubrobot.robots.pointcloud import PointCloudPerception

from PIL import Image as PIL_Image
from .controllers import Mpc_controller, PID_controller
from thread_utils import ReadWriteLock

import threading
import traceback

from ubrobot.robots.vlm import RobotVLM
from ubrobot.robots.nav import RobotNav, ControlMode

from dataclasses import dataclass

sys.path.append("/home/unitree/ubrobot/ros_depends_ws/src/rtabmap_odom_py/odom")

import rs_odom_module

class Go2Manager():
    def __init__(self):

        # 控制模式相关
        self.policy_init = True
        self.mpc = None
        self.pid = PID_controller(Kp_trans=2.0, Kd_trans=0.0, Kp_yaw=1.5, Kd_yaw=0.0, max_v=0.6, max_w=0.5)
        self.http_idx = -1

        # nav 
        self.global_nav_instruction_str = None
        self.nav_action = None
        self.nav_annotated_img = None

        # odom manager
        # Initialize the hardware and RTAB-Map
        print("Initializing D435i in front and Odometry...")
        self.tracker = None
        try:
            self.tracker = rs_odom_module.RealsenseOdom(camera_serial="419522070679")
            print("Waiting for camera data...")
            # Give the camera and RTAB-Map 2-3 seconds to sync and receive the first frame
            time.sleep(3.0)
        except RuntimeError as e:
            print("Failed to Initialize D435i in front", e)
            exit(1)

        # 读写锁相关
        self.mpc_rw_lock = ReadWriteLock()

        self.rgb_image = None
        self.depth_image = None

        self.odom = None
        self.odom_queue = deque(maxlen=50)
        self.vel = None

        # vlm model
        self.vlm = RobotVLM()

        # nav model
        self.nav = RobotNav()

        self.control_thread_instance = threading.Thread(target=self._control_thread, daemon=True)
        self.planning_thread_instance = threading.Thread(target=self._planning_thread, daemon=True)
        self.robot_arm_serving_thread_instance = threading.Thread(target=self._robot_arm_serving_thread, daemon=True)
        
        # unitree go2 dog
        self.go2client = None
        ChannelFactoryInitialize(0, "eth0") # default net card
        self.go2client = SportClient()
        self.go2client.SetTimeout(10.0)
        self.go2client.Init()
        # TODO set slow mode
        self.go2client.SpeedLevel(-1)

        # robot arm config
        self.cfg = PiperServerConfig()
        self.robot_arm = PiperHost(self.cfg.host)

        self.pc = PointCloudPerception()
    
    def get_observation(self):

        # 1. 获取相机内参
        '''intrinsics = self.tracker.get_camera_intrinsics()
        print("相机内参：")
        print(f"  焦距：fx={intrinsics['fx']:.2f}, fy={intrinsics['fy']:.2f}")
        print(f"  主点：cx={intrinsics['cx']:.2f}, cy={intrinsics['cy']:.2f}")
        print(f"  分辨率：{intrinsics['width']}x{intrinsics['height']}")
        print(f"  深度缩放因子：{intrinsics['scale']}")'''

        # get the current pose on-demand
        pose = self.tracker.get_pose_with_twist()
        # get speed info, including linear.x and angular.z
        twist = self.tracker.get_odom_twist()
        
        #if pose:
        #    print(f"\r位姿：x={pose[0]:.4f}, y={pose[1]:.4f}, yaw={pose[5]:.4f} | "
        #        f"速度：线速度={twist.linear_x:.2f}m/s, 角速度={twist.angular_z:.2f}rad/s", 
        #        end="")
        #else:
        #    print("\r位姿跟踪丢失 | 速度：0.00m/s, 0.00rad/s", end="")

        # get rgb image
        rgb_img = self.tracker.get_rgb_image()
        if not rgb_img.size == 0:
            # numpy数组可直接用于OpenCV处理（注意：RGB转BGR）
            self.rgb_image = rgb_img

        # get depth image
        depth_img = self.tracker.get_depth_image()
        if not depth_img.size == 0:
            self.depth_image = depth_img
            self.depth_image -= 0.0
            self.depth_image[np.where(self.depth_image < 0)] = 0
            self.depth_image[np.isnan(self.depth_image)] = 0
            self.depth_image[np.isinf(self.depth_image)] = 0
        
        if pose:
            # update pose info
            self.odom = [pose[0], pose[1], pose[5]]
            self.odom_queue.append((time.time(), copy.deepcopy(self.odom)))
            self.vel = [twist.linear_x, twist.angular_z]
        
        #image = PIL_Image.fromarray(self.rgb_image).convert('RGB')
        depth = (np.clip(self.depth_image * 10000.0, 0, 65535)).astype(np.uint16)

        #rgb_time = self.rgb_time
        #self.rgb_depth_rw_lock.release_read()

        #self.odom_rw_lock.acquire_read()
        #min_diff = 1e10
        odom_infer = None
        #for odom in self.odom_queue:
        #    diff = abs(odom[0] - rgb_time)
        #    if diff < min_diff:
        #        min_diff = diff
        #        odom_infer = copy.deepcopy(odom[1])
        #self.odom_rw_lock.release_read()
        odom_infer = self.odom
        return self.rgb_image, depth, odom_infer
        
    def get_next_planning(self):
        nav_action = self.nav_action
        vis_annotated_img = self.nav_annotated_img
        return nav_action, vis_annotated_img
    
    '''def reasoning_vlm(self, image_pil: PIL_Image.Image, instruction:str):
        response_restult_str = None
        response_restult_str = self.vlm.reasoning_vlm_infer(image_pil, instruction)
        return response_restult_str'''
    
    def set_user_instruction(self, instruction: str):
        # TODO implement this by LLM
        if instruction == "stop" or instruction == "STOP":
            self.global_nav_instruction_str = None
        else:
            self.global_nav_instruction_str = instruction
        self.http_idx = -1
        self.policy_init = True
       
    def get_action(self, policy_init, http_idx, rgb_image, depth, instruction, odom):
        nav_action = None
        vis_annotated_img = None
        if odom is not None and rgb_image is not None and depth is not None and instruction is not None:
            start = time.time()
            nav_action, vis_annotated_img = self.nav._dual_sys_eval(policy_init, http_idx, rgb_image, depth, instruction, odom)
            print(f"idx: {http_idx} step in get_action() cost {time.time() - start}")
        else:
            nav_action = None
            vis_annotated_img = rgb_image
        return nav_action, vis_annotated_img

    def _control_thread(self):
        while True:
            if self.global_nav_instruction_str is None:
                time.sleep(0.01)
                continue
            
            act = self.nav_action
            if act is None:
                time.sleep(0.01)
                continue
            time.sleep(0.1)
    
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
    
    def _robot_arm_serving_thread(self):
        self.robot_arm.start_serving_teleoperation(self.cfg)

    def _planning_thread(self):
        FPS = 30

        while True:
            t0 = time.time()

            rgb_image, depth, odom_infer = self.get_observation()
            nav_action, vis_annotated_img = self.get_action(self.policy_init, self.http_idx, rgb_image, depth, self.global_nav_instruction_str, odom_infer)
            
            # TODO if get STOP action signal, stop, waiting for next instruction
            self.nav_action = nav_action
            self.nav_annotated_img = vis_annotated_img
            # TODO double check
            if nav_action is not None:
                self.http_idx += 1
                self.policy_init = False

                print("get action...", nav_action.actions)
                # send action
                self.send_action(self.nav_action)
            # sleep
            time.sleep(max(0, 1.0 / FPS - (time.time() - t0)))
    
    def start_threads(self):
        self.planning_thread_instance.start()
        self.control_thread_instance.start()
        self.robot_arm_serving_thread_instance.start()
        print("✅ Go2Manager: control thread and planning thread started successfully")

    def move(self, vx, vy, vyaw):
        action = {"x.vel": vx,
                  "y.vel": 0,
                  "theta.vel": vyaw
                  }
        # self.lekiwi_base.send_action(action)
        
        if self.go2client is None:
            print("Go2 Sport Client NOT initialized!")
            return
        else:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # ms precision
            print(f"[{current_time}] receive move command [vx, vy, vyaw] {vx:.2f}, {vy:.2f}, {vyaw:.2f}")
            #self.go2client.Move(vx, vy, vyaw) #vx, vy, vyaw

    def go2_robot_stop(self):
        if self.go2client is None:
            print("Go2 Sport Client NOT initialized!")
            return
        else:
            self.go2client.StopMove()

    def go2_robot_standup(self):
        if self.go2client is None:
            print("Go2 Sport Client NOT initialized!")
            return
        else:
            self.go2client.StandUp()

    def go2_robot_standdown(self):
        if self.go2client is None:
            print("Go2 Sport Client NOT initialized!")
            return
        else:
            self.go2client.StandDown()

    def go2_robot_move(self):
        if self.go2client is None:
            print("Go2 Sport Client NOT initialized!")
            return -1
        else:
            self.go2client.SpeedLevel(-1) # slow 
            ret = self.go2client.Move(0.3,0,0)
            time.sleep(1)

            self.go2client.StopMove()
            return ret
    
    def get_robot_arm_image_observation(self):
        observation = self.robot_arm.get_robot_arm_observation_local()
        color_image = observation["wrist"] # TODO get "wrist" from configuration, avoid hard coding
        depth_image = observation["wrist_depth"]
        #print("get observation in arm action...", observation)
        # TODO rgb for vis, possible need to revise for algorithm
        color_image_pil = PIL_Image.fromarray(color_image)
        depth_image_pil = PIL_Image.fromarray(depth_image)
        return color_image_pil, depth_image_pil
    
    def get_robot_arm_manipulate_action(self):
        instruction = "Locate objects in current image and return theirs coordinates as json format. answer shortly."
        observation = self.robot_arm.get_robot_arm_observation_local()
        color_image = observation["wrist"] # TODO get "wrist" from configuration, avoid hard coding
        depth_image = observation["wrist_depth"]
        
        #res = self.vlm.vlm_infer_grounding(color_image, instruction)
        
        #instruction = "reach for the small wooden square block without collision"
        intrin = observation["wrist_intrinsics"]
        #response_restult_str_traj = self.vlm.vlm_infer_traj(color_image, depth_image, intrin, instruction)

        response_restult_str_traj = self.vlm.reasoning_vlm_infer(color_image, depth_image, intrin, instruction)

        #self.pc.convertRGBD2PointClouds(color_image, depth_image, intrin, "./rgbd_point_cloud.ply")

        print(response_restult_str_traj)
        #print(res)

if __name__ == "__main__":

    print("======= Starting Go2Manager Core =======")
    try:
        # 初始化 Go2Manager 实例
        manager = Go2Manager()
        manager.start_threads()
    except KeyboardInterrupt:
        print("\n======= Stopping Go2Manager Core =======")
        manager.nav_task_reset()
    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()
    finally:
        print("======= Go2Manager Core Exited =======")
