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

import rospy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

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
        rospy.init_node('go2_manager', anonymous=True) 

        # 控制模式相关
        self.policy_init = True
        self.mpc = None
        self.pid = PID_controller(Kp_trans=2.0, Kd_trans=0.0, Kp_yaw=1.5, Kd_yaw=0.0, max_v=0.6, max_w=0.5)
        self.http_idx = -1

        # nav 
        self.global_nav_instruction_str = None
        self.nav_action = None
        self.nav_annotated_img = None

        # 读写锁相关
        self.rgb_depth_rw_lock = ReadWriteLock()
        self.odom_rw_lock = ReadWriteLock()
        self.mpc_rw_lock = ReadWriteLock()
        self.act_rw_lock = ReadWriteLock()
        self.nav_rw_lock = ReadWriteLock()

        rgb_down_sub = Subscriber("/cam_front/camera/color/image_raw", Image)
        depth_down_sub = Subscriber("/cam_front/camera/aligned_depth_to_color/image_raw", Image)

        self.syncronizer = ApproximateTimeSynchronizer([rgb_down_sub, depth_down_sub], 1, 0.1)
        self.syncronizer.registerCallback(self.rgb_depth_down_callback)
        self.odom_sub = rospy.Subscriber("/rtabmap/odom", Odometry, self.odom_callback)


        # 控制指令发布器
        self.control_pub = rospy.Publisher('/cmd_vel_bridge', Twist, queue_size=5)

        self.cv_bridge = CvBridge()
        self.rgb_image = None
        self.rgb_bytes = None
        self.depth_image = None
        self.depth_bytes = None
        self.new_image_arrived = False
        self.rgb_time = 0.0

        self.odom = None
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.odom_queue = deque(maxlen=50)
        self.homo_odom = None
        self.vel = None

        # vlm model
        self.vlm = RobotVLM()

        # nav model
        self.nav = RobotNav()

        self.control_thread_instance = threading.Thread(target=self._control_thread, daemon=True)
        self.planning_thread_instance = threading.Thread(target=self._planning_thread, daemon=True)

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
        image_bytes = io.BytesIO()
        image_pil.save(image_bytes, format="JPEG")
        image_bytes.seek(0)
        response_restult_str = self.vlm.reasoning_vlm_infer(image_bytes, instruction)
        return response_restult_str
    
    def set_user_instruction(self, instruction: str):
        self.global_nav_instruction_str = instruction

    def get_rgb_depth_odom(self):
        self.rgb_depth_rw_lock.acquire_read()
        rgb_bytes = copy.deepcopy(self.rgb_bytes)
        depth_bytes = copy.deepcopy(self.depth_bytes)
        rgb_time = self.rgb_time
        self.rgb_depth_rw_lock.release_read()

        self.odom_rw_lock.acquire_read()
        min_diff = 1e10
        odom_infer = None
        for odom in self.odom_queue:
            diff = abs(odom[0] - rgb_time)
            if diff < min_diff:
                min_diff = diff
                odom_infer = copy.deepcopy(odom[1])
        self.odom_rw_lock.release_read()
        return rgb_bytes, depth_bytes, odom_infer
    
    def nav_policy_infer(self, policy_init, http_idx, image_bytes, depth_bytes, instruction, odom, homo_odom):
        nav_action, vis_annotated_img = self.nav._dual_sys_eval(policy_init, http_idx, image_bytes, depth_bytes, instruction, odom, homo_odom)
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
            #self.homo_goal = act.homo_goal
            homo_odom = copy.deepcopy(self.homo_odom) if self.homo_odom is not None else None
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

            rgb_bytes, depth_bytes, odom_infer = self.get_rgb_depth_odom()
            if odom_infer is not None and rgb_bytes is not None and depth_bytes is not None and self.global_nav_instruction_str is not None:

                start = time.time()
                nav_action, vis_annotated_img = self.nav_policy_infer(self.policy_init, self.http_idx, rgb_bytes, depth_bytes, self.global_nav_instruction_str, self.odom, self.homo_odom)

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

    def start_threads(self):
        self.planning_thread_instance.start()
        self.control_thread_instance.start()
        print("✅ Go2Manager: control thread and planning thread started successfully")

    def rgb_depth_down_callback(self, rgb_msg, depth_msg):
        """处理下视彩色图像和对齐后的深度图像消息"""
        # 处理彩色图像
        raw_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')[:, :, :]
        self.rgb_image = raw_image
        image = PIL_Image.fromarray(self.rgb_image)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')
        image_bytes.seek(0)

        # 处理深度图像
        raw_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, '16UC1')
        raw_depth[np.isnan(raw_depth)] = 0
        raw_depth[np.isinf(raw_depth)] = 0
        self.depth_image = raw_depth / 1000.0
        self.depth_image -= 0.0
        self.depth_image[np.where(self.depth_image < 0)] = 0
        depth = (np.clip(self.depth_image * 10000.0, 0, 65535)).astype(np.uint16)
        depth = PIL_Image.fromarray(depth)
        depth_bytes = io.BytesIO()
        depth.save(depth_bytes, format='PNG')
        depth_bytes.seek(0)

        # 保存数据和时间戳
        self.rgb_depth_rw_lock.acquire_write()
        self.rgb_bytes = image_bytes
        self.rgb_time = rgb_msg.header.stamp.secs + rgb_msg.header.stamp.nsecs / 1.0e9
        self.last_rgb_time = self.rgb_time
        self.depth_bytes = depth_bytes
        self.depth_time = depth_msg.header.stamp.secs + depth_msg.header.stamp.nsecs / 1.0e9
        self.last_depth_time = self.depth_time
        self.rgb_depth_rw_lock.release_write()

        # 标记图像更新
        #self.new_vis_image_arrived = True
        self.new_image_arrived = True

    def odom_callback(self, msg):
        """处理里程计消息，更新机器人位姿和速度"""
        #self.odom_cnt += 1
        self.odom_rw_lock.acquire_write()
        # 计算偏航角
        zz = msg.pose.pose.orientation.z
        ww = msg.pose.pose.orientation.w
        yaw = math.atan2(2 * zz * ww, 1 - 2 * zz * zz)
        # 更新位姿
        self.odom = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]
        self.odom_queue.append((time.time(), copy.deepcopy(self.odom)))
        #self.odom_timestamp = time.time()
        # 更新速度
        self.linear_vel = msg.twist.twist.linear.x
        self.angular_vel = msg.twist.twist.angular.z
        self.odom_rw_lock.release_write()

        # 计算齐次变换矩阵
        R0 = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        self.homo_odom = np.eye(4)
        self.homo_odom[:2, :2] = R0
        self.homo_odom[:2, 3] = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        self.vel = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]

    def move(self, vx, vy, vyaw):
        """发布机器人线速度和角速度控制指令"""
        request = Twist()
        request.linear.x = vx
        request.linear.y = 0.0
        request.angular.z = vyaw

        self.control_pub.publish(request)

        # 发送指令到机器人基座（可根据需要启用）
        action = {"x.vel": vx,
                  "y.vel": 0,
                  "theta.vel": vyaw
                  }
        # self.lekiwi_base.send_action(action)

    def set_nav_instruction(self, ins_str):
        """设置全局导航指令，替代原Gradio输入"""
        self.global_nav_instruction_str = ins_str
        # 可选：执行推理
        if self.rgb_bytes is not None:
            image_bytes = copy.deepcopy(self.rgb_bytes)
            #self._cosmos_reason1_infer(image_bytes, ins_str)

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
    # 初始化Go2Manager实例
    manager = Go2Manager()

    # 启动控制线程和规划线程
    manager.start_threads()

    # 可选：设置默认导航指令（替代Gradio输入）
    # manager.set_nav_instruction("walk close to office chair")

    try:
        # 初始化 Go2Manager 实例（内部已调用 rospy.init_node）
        manager = Go2Manager()
        # 启动控制线程和规划线程
        manager.start_threads()
        # 可选：设置默认导航指令
        # manager.set_nav_instruction("walk close to office chair")
        rospy.spin()
    except KeyboardInterrupt:
        print("\n======= Stopping Go2Manager Core =======")
        manager.nav_task_reset()
    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()
    finally:
        print("======= Go2Manager Core Exited =======")
