import random
import time
from pathlib import Path
from collections import OrderedDict
import json
import requests
import copy
import io
import math
from collections import deque
from enum import Enum
import numpy as np

import sys
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.go2.sport.sport_client import (
    SportClient,
    PathPoint,
    SPORT_PATH_POINT_SIZE,
)

#import rospy
#from message_filters import ApproximateTimeSynchronizer, Subscriber
#from cv_bridge import CvBridge
#from geometry_msgs.msg import Twist
#from nav_msgs.msg import Odometry
#from sensor_msgs.msg import Image

from PIL import Image as PIL_Image

from .controllers import Mpc_controller, PID_controller
from thread_utils import ReadWriteLock

import cv2
import threading
import os
import traceback

from .vlm import RobotVLM
from .nav import RobotNav, RobotAction, ControlMode

from dataclasses import dataclass

import sys
sys.path.append("/home/unitree/ubrobot/ros_depends_ws/src/rtabmap_odom_py/odom")

import rs_odom_module

@dataclass
class TestOption:
    name: str
    id: int

option_list = [
    TestOption(name="damp", id=0),         
    TestOption(name="stand_up", id=1),     
    TestOption(name="stand_down", id=2),   
    TestOption(name="move forward", id=3),         
    TestOption(name="move lateral", id=4),    
    TestOption(name="move rotate", id=5),  
    TestOption(name="stop_move", id=6),  
    TestOption(name="hand stand", id=7),
    TestOption(name="balanced stand", id=9),     
    TestOption(name="recovery", id=10),       
    TestOption(name="left flip", id=11),      
    TestOption(name="back flip", id=12),
    TestOption(name="free walk", id=13),  
    TestOption(name="free bound", id=14), 
    TestOption(name="free avoid", id=15),  
    TestOption(name="walk upright", id=17),
    TestOption(name="cross step", id=18),
    TestOption(name="free jump", id=19)       
]

class Go2Manager():
    def __init__(self):
        #rospy.init_node('go2_manager', anonymous=True) 

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
        print("Initializing D435i and Odometry...")
        self.tracker = None
        try:
            self.tracker = rs_odom_module.RealsenseOdom(camera_serial="419522070679")
            print("Waiting for camera data...")
            # Give the camera and RTAB-Map 2-3 seconds to sync and receive the first frame
            time.sleep(3.0)
        except RuntimeError as e:
            print("初始化失败：", e)
            exit(1)

        # 读写锁相关
        self.rgb_depth_rw_lock = ReadWriteLock()
        self.odom_rw_lock = ReadWriteLock()
        self.mpc_rw_lock = ReadWriteLock()
        self.act_rw_lock = ReadWriteLock()
        self.nav_rw_lock = ReadWriteLock()

        #rgb_down_sub = Subscriber("/cam_front/camera/color/image_raw", Image)
        #depth_down_sub = Subscriber("/cam_front/camera/aligned_depth_to_color/image_raw", Image)

        #self.syncronizer = ApproximateTimeSynchronizer([rgb_down_sub, depth_down_sub], 1, 0.1)
        #self.syncronizer.registerCallback(self.rgb_depth_down_callback)
        #self.odom_sub = rospy.Subscriber("/rtabmap/odom", Odometry, self.odom_callback)

        #self.cv_bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None
        self.new_image_arrived = False
        #self.rgb_time = 0.0

        self.odom = None
        self.odom_queue = deque(maxlen=50)
        self.vel = None

        # vlm model
        self.vlm = RobotVLM()

        # nav model
        self.nav = RobotNav()

        self.control_thread_instance = threading.Thread(target=self._control_thread, daemon=True)
        self.planning_thread_instance = threading.Thread(target=self._planning_thread, daemon=True)
        self.odom_thread_instance = threading.Thread(target=self._odom_thread, daemon=True)

        # nav action
        self.act = None

        # unitree go2 dog
        self.go2client = None
        ChannelFactoryInitialize(0, "eth0") # default net card
        self.go2client = SportClient()  
        self.go2client.SetTimeout(10.0)
        self.go2client.Init()
    
    def get_observation(self):
        # TODO  加锁
        image = PIL_Image.fromarray(self.rgb_image).convert('RGB')
        return image
    
    def get_next_planning(self):
        self.nav_rw_lock.acquire_read()
        nav_action = copy.deepcopy(self.nav_action)
        vis_annotated_img = copy.deepcopy(self.nav_annotated_img)
        self.nav_rw_lock.release_read()
        return nav_action, vis_annotated_img
    
    def reasoning_vlm(self, image_pil: PIL_Image.Image, instruction:str):
        response_restult_str = None
        response_restult_str = self.vlm.reasoning_vlm_infer(image_pil, instruction)
        return response_restult_str
    
    def set_user_instruction(self, instruction: str):
        self.global_nav_instruction_str = instruction

    def get_rgb_depth_odom(self):
        self.rgb_depth_rw_lock.acquire_read()
        rgb_image_pil = PIL_Image.fromarray(self.rgb_image)
        depth = (np.clip(self.depth_image * 10000.0, 0, 65535)).astype(np.uint16)
        depth_pil = PIL_Image.fromarray(depth)

        rgb_time = self.rgb_time
        self.rgb_depth_rw_lock.release_read()

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
        return odom_infer, rgb_image_pil, depth_pil
    
    def nav_policy_infer(self, policy_init, http_idx, rgb_image_pil, depth_pil, instruction, odom):        
        nav_action, vis_annotated_img = self.nav._dual_sys_eval(policy_init, http_idx, rgb_image_pil, depth_pil, instruction, odom)
        return nav_action, vis_annotated_img

    def _control_thread(self):
        while True:
            if self.global_nav_instruction_str is None:
                time.sleep(0.01)
                continue
            
            self.act_rw_lock.acquire_read()
            act = copy.deepcopy(self.act)
            self.act_rw_lock.release_read()
            if act is None:
                time.sleep(0.01)
                continue
            self.send_action(act)
            time.sleep(0.1)
    
    def send_action(self, act):
        if act.current_control_mode == ControlMode.MPC_Mode:
            self.mpc_rw_lock.acquire_write()
            if self.mpc is None:
                self.mpc = Mpc_controller(np.array(act.trajs_in_world))
            else:
                self.mpc.update_ref_traj(np.array(act.trajs_in_world))
            self.mpc_rw_lock.release_write()

            # MPC模式：基于轨迹的最优控制
            self.odom_rw_lock.acquire_read()
            odom = copy.deepcopy(self.odom) if self.odom else None
            self.odom_rw_lock.release_read()
            if self.mpc is not None and odom is not None:
                local_mpc = self.mpc
                opt_u_controls, opt_x_states = local_mpc.solve(np.array(odom))
                v, w = opt_u_controls[0, 0], opt_u_controls[0, 1]

                #self.move(v, 0.0, w)
        elif act.current_control_mode == ControlMode.PID_Mode:
            odom = copy.deepcopy(self.odom) if self.odom is not None else None
            # compute homo_odom by odom
            # 计算齐次变换矩阵
            yaw = odom[2]
            R0 = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            homo_odom = np.eye(4)
            homo_odom[:2, :2] = R0
            homo_odom[:2, 3] = odom[:2]
            
            #homo_odom = copy.deepcopy(self.homo_odom) if self.homo_odom is not None else None
            vel = copy.deepcopy(self.vel) if self.vel is not None else None
            if homo_odom is not None and vel is not None and act.homo_goal is not None:
                v, w, e_p, e_r = self.pid.solve(homo_odom, act.homo_goal, vel)
                if v < 0.0:
                    v = 0.0

                #print(v, w)
                #self.move(v, 0.0, w)
    

    def _planning_thread(self):

        while True:
            start_time = time.time()
            time.sleep(0.05)

            if not self.new_image_arrived:
                time.sleep(0.01)
                continue
            self.new_image_arrived = False

            odom_infer, rgb_image_pil, depth_pil = self.get_rgb_depth_odom()
            if odom_infer is not None and rgb_image_pil is not None and depth_pil is not None and self.global_nav_instruction_str is not None:

                start = time.time()
                nav_action, vis_annotated_img = self.nav_policy_infer(self.policy_init, self.http_idx, rgb_image_pil, depth_pil, self.global_nav_instruction_str, self.odom)

                self.nav_action = nav_action
                self.nav_annotated_img = vis_annotated_img

                self.policy_init = False
                self.http_idx += 1
                print(f"idx: {self.http_idx} after dual_sys_eval {time.time() - start}")
                
                self.act_rw_lock.acquire_write()
                self.act = nav_action
                self.act_rw_lock.release_write()
            else:
                #print(f"skip planning. odom_infer: {odom_infer is not None} rgb_bytes: {rgb_bytes is not None} depth_bytes: {depth_bytes is not None}")
                time.sleep(0.1)
            DESIRED_TIME = 0.3
            time.sleep(max(0, DESIRED_TIME - (time.time() - start_time)))
    
    def _odom_thread(self):
        # 1. 获取相机内参
        '''intrinsics = self.tracker.get_camera_intrinsics()
        print("相机内参：")
        print(f"  焦距：fx={intrinsics['fx']:.2f}, fy={intrinsics['fy']:.2f}")
        print(f"  主点：cx={intrinsics['cx']:.2f}, cy={intrinsics['cy']:.2f}")
        print(f"  分辨率：{intrinsics['width']}x{intrinsics['height']}")
        print(f"  深度缩放因子：{intrinsics['scale']}")'''

        while True:
            # Get the current pose on-demand
            pose = self.tracker.get_pose_with_twist()
            # 2. 获取速度（对应 odom_twist 的 linear.x 和 angular.z）
            twist = self.tracker.get_odom_twist()
            
            #if pose:
            #    print(f"\r位姿：x={pose[0]:.4f}, y={pose[1]:.4f}, yaw={pose[5]:.4f} | "
            #        f"速度：线速度={twist.linear_x:.2f}m/s, 角速度={twist.angular_z:.2f}rad/s", 
            #        end="")
            #else:
            #    print("\r位姿跟踪丢失 | 速度：0.00m/s, 0.00rad/s", end="")

            # 3. 获取RGB图像并显示
            rgb_img = self.tracker.get_rgb_image()
            if not rgb_img.size == 0:
                # numpy数组可直接用于OpenCV处理（注意：RGB转BGR）
                #print("================================== rgb image...")
                self.rgb_image = rgb_img
                #rgb_cv = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
                #cv2.imwrite('./rgb.png', rgb_cv)

            # 4. 获取深度图像并显示
            depth_img = self.tracker.get_depth_image()
            if not depth_img.size == 0:
                # 归一化深度图像用于显示
                #print("saving................depth image")
                depth_normalized = cv2.normalize(np.array(depth_img), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

            
                #raw_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, '16UC1')
                #raw_depth[np.isnan(raw_depth)] = 0
                #raw_depth[np.isinf(raw_depth)] = 0
                #self.depth_image = raw_depth / 1000.0
                self.depth_image = depth_img
                self.depth_image -= 0.0
                self.depth_image[np.where(self.depth_image < 0)] = 0
                self.depth_image[np.isnan(self.depth_image)] = 0
                self.depth_image[np.isinf(self.depth_image)] = 0

            #self.rgb_time = rgb_msg.header.stamp.secs + rgb_msg.header.stamp.nsecs / 1.0e9
            
            # 标记图像更新
            self.new_image_arrived = True

            if pose:
                # 更新位姿
                self.odom = [pose[0], pose[1], pose[5]]
                self.odom_queue.append((time.time(), copy.deepcopy(self.odom)))
                self.vel = [twist.linear_x, twist.angular_z]
            
            # Note: As discussed, too slow (like 1s) will cause tracking loss if moving.
            time.sleep(0.05) # ~20Hz recommended

    def start_threads(self):
        self.planning_thread_instance.start()
        self.control_thread_instance.start()
        self.odom_thread_instance.start()
        print("✅ Go2Manager: control thread and planning thread started successfully")

    '''def rgb_depth_down_callback(self, rgb_msg, depth_msg):
        """处理下视彩色图像和对齐后的深度图像消息"""
        # 处理彩色图像
        raw_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')[:, :, :]
        self.rgb_image = raw_image

        # 处理深度图像
        raw_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, '16UC1')
        raw_depth[np.isnan(raw_depth)] = 0
        raw_depth[np.isinf(raw_depth)] = 0
        self.depth_image = raw_depth / 1000.0
        self.depth_image -= 0.0
        self.depth_image[np.where(self.depth_image < 0)] = 0

        # 保存数据和时间戳
        self.rgb_depth_rw_lock.acquire_write()
        self.rgb_time = rgb_msg.header.stamp.secs + rgb_msg.header.stamp.nsecs / 1.0e9
        self.rgb_depth_rw_lock.release_write()

        # 标记图像更新
        self.new_image_arrived = True

    def odom_callback(self, msg):
        """处理里程计消息，更新机器人位姿和速度"""
        self.odom_rw_lock.acquire_write()
        # 计算偏航角
        zz = msg.pose.pose.orientation.z
        ww = msg.pose.pose.orientation.w
        yaw = math.atan2(2 * zz * ww, 1 - 2 * zz * zz)
        # 更新位姿
        self.odom = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]
        self.odom_queue.append((time.time(), copy.deepcopy(self.odom)))
        self.vel = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]
        # 更新速度
        self.odom_rw_lock.release_write()'''

    def move(self, vx, vy, vyaw):
        """发布机器人线速度和角速度控制指令"""
        #request = Twist()
        #request.linear.x = vx
        #request.linear.y = 0.0
        #request.angular.z = vyaw

        # 发送指令到机器人基座（可根据需要启用）
        action = {"x.vel": vx,
                  "y.vel": 0,
                  "theta.vel": vyaw
                  }
        # self.lekiwi_base.send_action(action)

    def nav_task_reset(self):
        """重置导航任务，停止机器人运动"""
        self.global_nav_instruction_str = None
        self.move(0.0, 0.0, 0.0)
        print("✅ Go2Manager: nav task reset and robot stopped")

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

if __name__ == "__main__":

    print("======= Starting Go2Manager Core =======")
    try:
        # 初始化 Go2Manager 实例
        manager = Go2Manager()
        # 启动控制线程和规划线程
        manager.start_threads()

        #rospy.spin()
    except KeyboardInterrupt:
        print("\n======= Stopping Go2Manager Core =======")
        manager.nav_task_reset()
    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()
    finally:
        print("======= Go2Manager Core Exited =======")
