#!/usr/bin/env python3
# coding:utf-8

import rospy
import time
import math
from enum import Enum
from collections import deque
import io

import time
import random
import numpy as np
#import pinocchio as pin
from piper_sdk import C_PiperInterface_V2

from ubrobot.robots.piper.piper_client import PiperClient, PiperClientConfig

#from pyroboplan.core import RobotModel
#from pyroboplan.core.robot import RobotModel
#from pyroboplan.models.utils import RobotModel
#from pyroboplan.planning.rrt import RRTPlanner

# ROS消息导入
#from sensor_msgs.msg import PointCloud2
#import sensor_msgs.point_cloud2 as pc2
#from piper_msgs.msg import PosCmd
import numpy as np
#import tf2_ros
from geometry_msgs.msg import TransformStamped, Point, PoseStamped, Quaternion
#import tf2_geometry_msgs
#from tf2_geometry_msgs import do_transform_pose
#from std_msgs.msg import Bool, Float64, Int32
#import tf.transformations as tf_trans

from message_filters import ApproximateTimeSynchronizer, Subscriber
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

from sensor_msgs.msg import JointState

from thread_utils import ReadWriteLock
from PIL import Image as PIL_Image

import cv2
import open3d as o3d

from scipy.linalg import qr
import transforms3d.quaternions as tfq

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pointcloud import PointCloudPerception

from pointcloud import GraspPoseCalculator

from piper_sdk import *

from .vlm import RobotVLM

from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from lerobot.cameras import ColorMode, Cv2Rotation

class RobotState(Enum):
    IDLE = 0
    SEARCHING = 1
    APPROACHING = 2
    GRASPING = 3
    RETREATING = 4
    ERROR = 5

class IKStatusManager:
    """逆解状态管理器"""
    def __init__(self):
        self.ik_success = False
        self.ik_status_received = False
        #self.ik_sub = rospy.Subscriber("/ik_status", Bool, self.ik_status_callback)
        
    def ik_status_callback(self, msg):
        """逆解状态回调"""
        self.ik_success = msg.data
        self.ik_status_received = True
        rospy.logdebug(f"IK status: {self.ik_success}")

class PoseAdjuster:
    """姿态调整器"""
    def __init__(self):
        self.max_pitch_adjustment = math.pi / 2  # 最大90度调整
        self.max_roll_adjustment = math.pi / 4   # 最大45度roll调整
        self.pitch_step = math.pi / 18 / 2  # 5度步长
        self.roll_step = math.pi / 18 / 2   # 5度步长
        self.adjustment_sequence = []  # 调整序列 (pitch, roll)
        
    def generate_adjustment_sequence(self):
        """生成调整序列:使用螺旋式渐进搜索，保持yaw不变"""
        self.adjustment_sequence = []
        
        # 首先尝试原始姿态
        self.adjustment_sequence.append((0, 0))
        
        # 螺旋式渐进搜索参数
        spiral_step = math.pi / 18  # 10度步长
        max_layers = int(max(self.max_pitch_adjustment, self.max_roll_adjustment) / spiral_step)
        
        # 生成螺旋序列
        for layer in range(1, max_layers + 1):
            current_radius = layer * spiral_step
            
            # 每层采样点数随半径增加
            points_in_layer = max(8, int(8 * layer))
            
            for i in range(points_in_layer):
                angle = 2 * math.pi * i / points_in_layer
                
                # 椭圆螺旋，pitch范围大，roll范围小
                pitch_adj = current_radius * math.sin(angle) * (self.max_pitch_adjustment / (self.max_pitch_adjustment + self.max_roll_adjustment))
                roll_adj = current_radius * math.cos(angle) * (self.max_roll_adjustment / (self.max_pitch_adjustment + self.max_roll_adjustment))
                
                # 限制在最大范围内
                pitch_adj = max(min(pitch_adj, self.max_pitch_adjustment), -self.max_pitch_adjustment)
                roll_adj = max(min(roll_adj, self.max_roll_adjustment), -self.max_roll_adjustment)
                
                # 过滤掉不合理的极端组合
                if self._is_valid_combination(pitch_adj, roll_adj):
                    self.adjustment_sequence.append((pitch_adj, roll_adj))
        
        rospy.loginfo(f"Generated spiral adjustment sequence with {len(self.adjustment_sequence)} poses")
        return self.adjustment_sequence
    
    def _is_valid_combination(self, pitch, roll):
        """检查是否为有效组合（过滤掉不合理的组合）"""
        # 避免极端组合，比如大pitch+大roll
        if abs(pitch) > math.pi/4 and abs(roll) > math.pi/6:
            return False
        # 避免过小的调整（与原点太近的重复点）
        if abs(pitch) < 0.01 and abs(roll) < 0.01:
            return False
        return True
    
    def get_adjustment_value(self, attempt_count):
        """根据尝试次数获取调整值"""
        if attempt_count < len(self.adjustment_sequence):
            return self.adjustment_sequence[attempt_count]
        
        return None  # 超出调整范围
    
    def adjust_pose_orientation(self, pose, pitch_adjustment, roll_adjustment):
        """调整姿态的俯仰角和滚转角，保持yaw不变"""
        # 获取原始四元数
        orig_quat = [
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w
        ]
        
        # 转换为欧拉角
        orig_euler = tf_trans.euler_from_quaternion(orig_quat)
        
        # 调整俯仰角和滚转角，yaw保持不变
        new_euler = [
            orig_euler[0] + roll_adjustment,    # roll
            orig_euler[1] + pitch_adjustment,   # pitch
            orig_euler[2]                       # yaw保持不变
        ]
        
        # 转换回四元数
        new_quat = tf_trans.quaternion_from_euler(*new_euler)
        
        # 创建新的姿态
        adjusted_pose = PoseStamped()
        adjusted_pose.header = pose.header
        adjusted_pose.pose.position = pose.pose.position
        adjusted_pose.pose.orientation.x = new_quat[0]
        adjusted_pose.pose.orientation.y = new_quat[1]
        adjusted_pose.pose.orientation.z = new_quat[2]
        adjusted_pose.pose.orientation.w = new_quat[3]
        
        return adjusted_pose

class ActionSequence:
    """动作序列管理类"""
    def __init__(self):
        self.sequence = deque()
        self.current_action = None
        self.last_action_time = 0
        self.action_delay = 1.0  # 默认动作延迟2秒
        
    def add_action(self, action_name, action_func, delay=None, *args, **kwargs):
        """添加动作到序列"""
        action_delay = delay if delay is not None else self.action_delay
        self.sequence.append((action_name, action_func, action_delay, args, kwargs))
        
    def execute_next(self):
        """执行下一个动作"""
        if self.sequence and (time.time() - self.last_action_time > self.action_delay or self.last_action_time == 0):
            action_name, action_func, delay, args, kwargs = self.sequence.popleft()
            rospy.loginfo(f"Executing action: {action_name}")
            
            # 执行动作函数
            result = action_func(*args, **kwargs)
            
            # 记录执行时间并设置延迟
            self.last_action_time = time.time()
            self.action_delay = delay
            
            rospy.loginfo(f"Action '{action_name}' completed. Waiting {delay:.1f} seconds...")
            return result
            
        return None
        
    def clear(self):
        """清空动作序列"""
        self.sequence.clear()
        self.current_action = None
        self.last_action_time = 0
        self.action_delay = 1.0
        
    def is_empty(self):
        """检查序列是否为空"""
        return len(self.sequence) == 0

class PoseTransformer:
    def __init__(self):
        # Point Cloud from RealSense (RGBD)
        self.orig_pcd = None
        self.pc = PointCloudPerception()
        self.grasp_calc = GraspPoseCalculator()

        self.vlm = RobotVLM()

        # 初始化tf2
        #self.tf_buffer = tf2_ros.Buffer()
        #self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        self.rgb_depth_rw_lock = ReadWriteLock()

        # get robot arm state 
        #self.robot_joint_states_sub = Subscriber("joint_states_single", JointState)
        #self.robot_arm_status_sub = Subscriber("arm_status", JointState)
        #self.robot_end_pose_sub = Subscriber("end_pose", JointState)
        #self.robot_end_pose_euler_sub = Subscriber("end_pose_euler", JointState)

        #self.piper = C_PiperInterface_V2()
        #self.piper.ConnectPort()
        
        # Example with depth capture and custom settings
        '''custom_config = RealSenseCameraConfig(
            serial_number_or_name="336222070923", # Replace with actual SN
            fps=30,
            width=1280,
            height=720,
            color_mode=ColorMode.BGR, # Request BGR output
            rotation=Cv2Rotation.NO_ROTATION,
            use_depth=True
        )
        self.rgb_depth_camera = RealSenseCamera(custom_config)
        self.rgb_depth_camera.connect()

        # TODO When done, properly disconnect the camera using
        #self.rgb_depth_camera.disconnect() # TODO disconnect finally'''

        self.robot_config = PiperClientConfig(remote_ip="192.168.18.113", id="robot_arm_piper")
        # Initialize the robot and teleoperator
        self.robot = PiperClient(self.robot_config)
        # Connect to the robot and teleoperator
        # To connect you already should have this script running on LeKiwi: `python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi`
        self.robot.connect()
        
        # 逆解状态管理
        self.ik_manager = IKStatusManager()
        self.pose_adjuster = PoseAdjuster()
        
        self.rgb_image = None
        self.rgb_bytes = None
        self.depth_image = None
        self.depth_bytes = None

        # 相机内参存储（关键：用于2D转3D）
        self.fx = None
        self.fy = None
        self.ppx = None
        self.ppy = None
        
        # 状态管理
        self.original_pose = None
        self.transformed_pose = None
        self.current_target_pose = None
        self.original_target_pose = None  # 保存原始目标姿态
        self.z_down_pose = None
        self.continuous_publishing = False
        self.gripper_cmd = 0.0
        self.robot_state = RobotState.IDLE
        self.waiting_for_ik = False
        self.ik_check_start_time = 0
        self.adjustment_attempts = 0
        self.max_adjustment_attempts = len(self.pose_adjuster.generate_adjustment_sequence())
        self.ik_success_pose = None  # 保存成功的姿态
        self.motion_complete_time = 0  # 运动完成时间
        self.motion_in_progress = False  # 运动进行中标志
        
        # 动作序列管理
        self.action_sequence = ActionSequence()
        
        # 预定义路径点
        self.via_point = self.create_via_pose(
            position=[0.300, 0.0, 0.360],
            orientation=[0.007, 0.915, 0.009, 0.403]
        )
        self.via_pose_list = []
        
    def create_via_pose(self, position, orientation, frame_id="base_link"):
        """创建路径点姿势"""
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.pose.position.x = position[0]
        pose.pose.position.y = position[1]
        pose.pose.position.z = position[2]
        pose.pose.orientation.x = orientation[0]
        pose.pose.orientation.y = orientation[1]
        pose.pose.orientation.z = orientation[2]
        pose.pose.orientation.w = orientation[3]
        return pose

    def correct_pose_z_down(self, original_pose):
        # 获取原始四元数
        original_quat = [
            original_pose.pose.orientation.x,
            original_pose.pose.orientation.y,
            original_pose.pose.orientation.z,
            original_pose.pose.orientation.w
        ]
        
        # 转换为旋转矩阵
        R_original = tf_trans.quaternion_matrix(original_quat)[:3, :3]
        
        # 提取Z轴向量（原始姿态的Z方向）
        z_axis_original = R_original[:, 2]
        
        # 我们希望新的Z轴指向地面（向下）
        z_axis_desired = np.array([0, 0, -1])  # Z轴向下
        
        # 计算旋转使得原始Z轴对准期望的向下Z轴
        # 使用轴角表示法
        rotation_axis = np.cross(z_axis_original, z_axis_desired)
        
        if np.linalg.norm(rotation_axis) < 1e-6:
            # 如果Z轴已经向下（或向上），使用特殊处理
            if z_axis_original[2] < 0:
                # 已经向下，不需要旋转
                new_quat = original_quat
            else:
                # Z轴向上，绕任意水平轴旋转180度
                rotation_axis = np.array([1, 0, 0])  # 绕X轴
                angle = np.pi
                rotation_quat = tf_trans.quaternion_about_axis(angle, rotation_axis)
                new_quat = tf_trans.quaternion_multiply(rotation_quat, original_quat)
        else:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            angle = np.arccos(np.clip(np.dot(z_axis_original, z_axis_desired), -1, 1))
            
            # 创建旋转四元数
            rotation_quat = tf_trans.quaternion_about_axis(angle, rotation_axis)
            
            # 应用旋转
            new_quat = tf_trans.quaternion_multiply(rotation_quat, original_quat)
        
        # 更新pose
        corrected_pose = PoseStamped()
        corrected_pose.header = original_pose.header
        corrected_pose.pose.position = original_pose.pose.position
        corrected_pose.pose.orientation.x = new_quat[0]
        corrected_pose.pose.orientation.y = new_quat[1]
        corrected_pose.pose.orientation.z = new_quat[2]
        corrected_pose.pose.orientation.w = new_quat[3]
        
        return corrected_pose

    def transform_pose(self, pose_msg, target_frame='base_link'):
        """坐标变换"""
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                pose_msg.header.frame_id,
                rospy.Time(0),
                rospy.Duration(1.0)
            )
            
            transformed_pose = do_transform_pose(pose_msg, transform)
            transformed_pose.header.frame_id = target_frame
            transformed_pose.header.stamp = rospy.Time.now()
            
            return transformed_pose
            
        except (tf2_ros.LookupException, 
                tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("坐标变换失败: %s", str(e))
            return None
    
    def get_observation(self):
        observation = self.robot.get_observation()
        color_image = observation["wrist"] # TODO get "wrist" from configuration, avoid hard coding
        #print("get observation in arm action...", observation)
        #color_image = self.rgb_depth_camera.read()
        image = PIL_Image.fromarray(color_image).convert('RGB')
        return image
    
    def grounding_objects_2d(self, image_pil: PIL_Image.Image, instruction:str):
        response_restult_str = None
        image_bytes = io.BytesIO()
        image_pil.save(image_bytes, format="JPEG")
        image_bytes.seek(0)
        #instruction = "Locate objects in current image and return theirs coordinates as json format."
        #response_restult_str = self.vlm.reasoning_vlm_infer(image_bytes, instruction)
        response_restult_str = self.vlm.vlm_infer_grounding(image_bytes, instruction)
        instruction = "reach for the small wooden square block without collision"
        print(response_restult_str)

        response_restult_str_traj = self.vlm.vlm_infer_traj(image_bytes, instruction)
        print(response_restult_str_traj)
        
        return response_restult_str
        
    def rgb_depth_down_callback(self, rgb_msg, depth_msg):
        """处理下视彩色图像和对齐后的深度图像消息"""
        # 处理彩色图像
        self.rgb_image = self.rgb_depth_camera.read()
        image = PIL_Image.fromarray(self.rgb_image)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')
        image_bytes.seek(0)

        # 处理深度图像
        # np.ndarray: The depth map as a NumPy array (height, width) of type `np.uint16` (raw depth values in millimeters) and rotation.
        raw_depth = self.rgb_depth_camera.read_depth()
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
        self.depth_bytes = depth_bytes
        self.rgb_depth_rw_lock.release_write()

        # get rgbd image and convert to poing cloud
        self.orig_pcd = self.pc.convertRGBD2PointClouds(self.rgb_image, self.depth_image, self.fx, self.fy, self.ppx, self.ppy)
    
    def get_manipulate_pose_camera_link(self):
        """
        串联2D检测->3D数据提取->可视化
        """
        if self.rgb_image is None or self.depth_image is None or self.orig_pcd is None:
            print("RGB or Depth Image or Point Cloud is None...")
            return

        # 3d object detection and segmentation
        target_pcd_list, aabb_list, obb_list = self.pc.object_3d_segmentation(self.rgb_image, self.depth_image, self.fx, self.fy, self.ppx, self.ppy)

        gripper_max_opening = 0.5  # 机械爪最大张开距离（米），根据实际硬件调整（如0.1米）
        frame_id = "camera_color_optical_frame"  # 坐标系ID（与你的点云坐标系一致）
        grasp_pose = self.grasp_calc.compute_grasp_pose(obb_list[0], gripper_max_opening, frame_id)
        if grasp_pose is not None:
            print("\n===== 最终抓取姿态 =====")
            print(f"坐标系：{grasp_pose['header']['frame_id']}")
            print(f"抓取位置：x={grasp_pose['pose']['position']['x']:.3f}m, y={grasp_pose['pose']['position']['y']:.3f}m, z={grasp_pose['pose']['position']['z']:.3f}m")
            print(f"抓取四元数：x={grasp_pose['pose']['orientation']['x']:.3f}, y={grasp_pose['pose']['orientation']['y']:.3f}, z={grasp_pose['pose']['orientation']['z']:.3f}, w={grasp_pose['pose']['orientation']['w']:.3f}")
            self.pc.export_grasp_visualization_to_ply(target_pcd_list[0], grasp_pose, aabb_list[0], obb_list[0])
    
    def pose_callback(self, msg):
        """PoseStamped消息回调"""
        self.original_pose = msg
        self.transformed_pose = self.transform_pose(msg, "base_link")
        
        if self.transformed_pose:
            rospy.loginfo("Pose transformed successfully")
            self.print_pose_info("Original", self.original_pose)
            self.print_pose_info("Transformed", self.transformed_pose)

    def target_pose_callback(self, msg):
        """TargetPose消息回调"""
        self.target_pose_current = msg

    def print_pose_info(self, label, pose):
        """打印姿势信息"""
        if pose:
            pos = pose.pose.position
            orient = pose.pose.orientation
            # 转换为欧拉角显示
            euler = tf_trans.euler_from_quaternion([
                orient.x, orient.y, orient.z, orient.w
            ])
            print(f"{label} pose - Frame: {pose.header.frame_id}, "
                  f"Position: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}), "
                  f"Orientation (RPY): ({np.degrees(euler[0]):.1f}°, "
                  f"{np.degrees(euler[1]):.1f}°, {np.degrees(euler[2]):.1f}°)")

    def publish_pose_with_ik_check(self, pose=None):
        """发布姿势并检查逆解状态"""
        target_pose = pose if pose else self.correct_pose_z_down(self.transformed_pose)
        
        if not target_pose:
            rospy.logwarn("No pose available to publish")
            return False
        
        # 如果已经有运动在进行中，不发布新的目标
        if self.motion_in_progress:
            rospy.logwarn("Motion in progress, skipping new target")
            return False
        
        # 保存原始目标姿态

        self.original_target_pose = target_pose

        self.pose_adjuster.generate_adjustment_sequence()
        
        # 重置调整状态
        self.reset_ik_check_state()
        
        # 发布初始姿态（尝试0：原始姿态）
        self.publish_adjusted_pose(0)
        
        return True

    def publish_adjusted_pose(self, attempt_count):
        """发布调整后的姿态"""
        adjustment_values = self.pose_adjuster.get_adjustment_value(attempt_count)
        
        if adjustment_values is None:
            rospy.logerr("No more adjustment values available")
            self.waiting_for_ik = False
            return False
        
        pitch_adjustment, roll_adjustment = adjustment_values
        
        # 调整姿态

        adjusted_pose = self.pose_adjuster.adjust_pose_orientation(
            self.original_target_pose, pitch_adjustment, roll_adjustment
        )
        
        # 发布调整后的姿态
        adjusted_pose.header.stamp = rospy.Time.now()
        self.target_pose_pub.publish(adjusted_pose)
        self.current_target_pose = adjusted_pose
        
        # 打印调整信息
        euler = tf_trans.euler_from_quaternion([
            adjusted_pose.pose.orientation.x,
            adjusted_pose.pose.orientation.y,
            adjusted_pose.pose.orientation.z,
            adjusted_pose.pose.orientation.w
        ])
        
        if attempt_count == 0:
            rospy.loginfo(f"Attempt {attempt_count}: Original pose, "
                         f"roll = {np.degrees(euler[0]):.1f}°, "
                         f"pitch = {np.degrees(euler[1]):.1f}°")
        else:
            rospy.loginfo(f"Attempt {attempt_count}: "
                         f"Adjusted roll = {np.degrees(roll_adjustment):.1f}°, "
                         f"pitch = {np.degrees(pitch_adjustment):.1f}°, "
                         f"total roll = {np.degrees(euler[0]):.1f}°, "
                         f"total pitch = {np.degrees(euler[1]):.1f}°")
        
        return True

    def check_motion_completion(self):
        """检查运动是否完成"""
        if self.motion_in_progress and time.time() >= self.motion_complete_time:
            rospy.loginfo("Motion completed successfully!")
            self.motion_in_progress = False
            return True
        return False

    def control_gripper(self, position):
        """控制夹爪"""
        self.gripper_cmd = position
        self.gripper_cmd_pub.publish(Float64(self.gripper_cmd))
        rospy.loginfo(f"Gripper set to: {position}")
        return True
    
    def test_robot_move(self):
        #while( not self.piper.EnablePiper()):
        #    time.sleep(0.01)
        
        arm_position = [0.0, 0, 0, 0, 0, 0]
        self.piper_mp.call_joint_moveit_ctrl_arm(arm_position) # 回零
        time.sleep(1)

        '''self.piper.GripperCtrl(0,1000,0x01, 0)
        factor = 1000

        current_end_pose_msg = self.piper.GetArmEndPoseMsgs()
        print(current_end_pose_msg.end_pose.X_axis)

        current_pose_raw = [
            current_end_pose_msg.end_pose.X_axis,
            current_end_pose_msg.end_pose.Y_axis,
            current_end_pose_msg.end_pose.Z_axis,
            current_end_pose_msg.end_pose.RX_axis,
            current_end_pose_msg.end_pose.RY_axis,
            current_end_pose_msg.end_pose.RZ_axis
        ]
        
        # delta position
        position = [5.0, 0.0, 0.0, 0, 0.0, 0, 20]
        X = round(position[0]*factor)
        Y = round(position[1]*factor)
        Z = round(position[2]*factor)
        RX = round(position[3]*factor)
        RY = round(position[4]*factor)
        RZ = round(position[5]*factor)
        joint_6 = round(position[6]*factor)
        print(X,Y,Z,RX,RY,RZ)

        target_pose_raw = [round(current + delta) for current, delta in zip(current_pose_raw, position)]
        
        self.piper.MotionCtrl_2(0x01, 0x00, 50, 0x00)
        
        print("target pose:", target_pose_raw)
        
        time.sleep(0.1)

        #self.piper.EndPoseCtrl(arget_pose_raw[0], target_pose_raw[1], target_pose_raw[2], target_pose_raw[3], target_pose_raw[4], target_pose_raw[5])
       
        time.sleep(0.1)

        self.piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
        time.sleep(0.1)'''

    def execute_grasp_sequence(self):
        """执行抓取序列"""
        '''self.action_sequence.clear()
        
        # 构建抓取动作序列
        self.action_sequence.add_action("Open Gripper", self.control_gripper, 0.1, 0.07)
        self.action_sequence.add_action("Move to Target", self.publish_pose_with_ik_check, 1.0)
        self.action_sequence.add_action("Close Gripper", self.control_gripper, 0.5, 0.0)
        self.action_sequence.add_action("Move to Via Point", self.publish_pose_with_ik_check, 1.5, self.via_point)
        self.action_sequence.add_action("Open Gripper", self.control_gripper, 0.1, 0.07)
        
        self.robot_state = RobotState.GRASPING
        rospy.loginfo("Grasp sequence started with delays")
        self.task_cmd = 0
        self.task_reslut = 2'''
        print("======================== test...")
        self.get_manipulate_pose_camera_link()

    def record_search_route(self):
        """记录搜索路径"""
        self.via_pose_list.append(self.target_pose_current)
        rospy.loginfo("Recording search route, now is %d point", len(self.via_pose_list)+1)

    def search_mode(self):
        """执行搜索模式"""
        self.action_sequence.clear()
        self.action_sequence.add_action("Close Gripper", self.control_gripper, 0.5, 0.0)

        if (len(self.via_pose_list)>0):
            for i in range(0, len(self.via_pose_list)):
                self.action_sequence.add_action("Move to Via Point", self.publish_pose_with_ik_check, 1.0, self.via_pose_list[i])
        
        self.robot_state = RobotState.GRASPING
        rospy.loginfo("Searching...")

    def print_instructions(self):
        """打印操作指令"""
        instructions = [
            "\nControl commands:",
            "s: 执行抓取序列 (包含延迟)",
            "p: 切换连续发布模式",
            "b: 移动到安全位置并打开夹爪",
            "t: 切换夹爪状态",
            "c: 清除当前动作序列",
            "i: 显示当前信息和指令"
        ]
        
        print("\n".join(instructions))
        
        if self.transformed_pose:
            self.print_pose_info("Current Target", self.transformed_pose)

    def deploy_piper_plan(self):

        # 2. PLANNER SETUP
        # Load the Piper model from its URDF for kinematics and collision checking
        # Replace 'path/to/piper.urdf' with your actual file path
        urdf_path = "./ros_depends_ws/src/piper_ros/src/piper_description/urdf/piper_description.urdf"
        model = pin.buildModelFromUrdf(urdf_path)

        collision_model = pin.buildGeomFromUrdf(model, urdf_path, pin.COLLISION)

        #planner = RRTPlanner(model, collision_model)

        # 3. DEFINE START AND GOAL
        # Get current joint positions from the real robot
        # piper_sdk returns a list of 6 joint values in radians
        start_q = np.array(piper.get_joint_states().joint_modules.joint_states)
        
        # Define a goal configuration (example: reaching forward)
        goal_q = np.array([0.5, -0.2, 0.3, 0.0, 1.2, 0.0])

        # 4. GENERATE COLLISION-FREE PATH
        print("Planning path...")
        path = planner.plan(start_q, goal_q)

        if not path:
            print("Planning failed!")
            return

        # 5. APPLY TOPP-RA SMOOTHING
        # Define Piper's physical limits (example values for 2026)
        vel_limits = np.array([1.5, 1.5, 1.5, 2.0, 2.0, 2.0]) 
        accel_limits = np.array([3.0, 3.0, 3.0, 4.0, 4.0, 4.0])
        dt = 0.02  # 50Hz control loop

        print("Smoothing trajectory with TOPP-RA...")
        traj_opt = CubicTrajectoryOptimization(path, dt=dt)
        trajectory = traj_opt.solve()

        # 6. EXECUTE ON HARDWARE
        print(f"Executing trajectory ({len(times)} points)...")
        try:
            for point in trajectory.points:
                print(point)
                # Send joint command to the Piper hardware
                # The SDK expects values in radians
                #piper.motion_ctrl.joint_motion(target_q.tolist())
                
                # Synchronize with the trajectory time step
                time.sleep(dt)
            print("Execution complete.")
        except KeyboardInterrupt:
            print("Emergency Stop triggered.")
            # Optional: send emergency stop command if available in SDK
        finally:
            piper.disconnect()

    '''def run(self):
        """主循环"""
        self.print_instructions()
        
        while not rospy.is_shutdown():
            # 处理键盘输入
            if key := self.get_key() or self.task_cmd:
                self.handle_key_input(key)
            
            # 检查运动是否完成
            if self.check_motion_completion():
                # 运动完成，等待2秒后继续
                rospy.loginfo("Motion completed! Waiting 2 seconds before continuing...")
                rospy.sleep(1.0)
                # 这里可以添加状态转换逻辑
            
            # 检查逆解状态（只有在没有运动进行时才检查）
            if not self.motion_in_progress:
                ik_result = self.check_ik_status()
            
            # 执行动作序列（只有在没有运动进行时才执行）
            if not self.action_sequence.is_empty() and not self.waiting_for_ik and not self.motion_in_progress:
                self.action_sequence.execute_next()
            
            # 连续发布模式（只有在没有运动进行时才发布）
            if (self.continuous_publishing and self.transformed_pose and 
                not self.waiting_for_ik and not self.motion_in_progress):
                pass
            
            self.rate.sleep()'''

if __name__ == '__main__':
    try:
        transformer = PoseTransformer()
        #transformer.run()
    except rospy.ROSInterruptException:
        pass
