#!/usr/bin/env python3
# coding:utf-8

import rospy
import sys
import select
import tty
import termios
import time
import math
from enum import Enum
from collections import deque
import io

import json
# ROS消息导入
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from piper_msgs.msg import PosCmd
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped, Point, PoseStamped, Quaternion
import tf2_geometry_msgs
from tf2_geometry_msgs import do_transform_pose
from std_msgs.msg import Bool, Float64, Int32
import tf.transformations as tf_trans

from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

from thread_utils import ReadWriteLock
from PIL import Image as PIL_Image

import cv2
import open3d as o3d
from ultralytics import YOLO

from scipy.linalg import qr
import transforms3d.quaternions as tfq

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GraspPoseCalculator:
    def __init__(self):
        """初始化抓取姿态计算器"""
        pass

    def select_grasp_axis(self, aabb_dimensions, gripper_max_opening):
        """
        夹持方向筛选 + 可抓取性判断
        :param aabb_dimensions: AABB盒尺寸 [x_length, y_length, z_length]（物体真实长、宽、高，PCA坐标系下）
        :param gripper_max_opening: 机械爪最大张开距离（米）
        :return: (grasp_axis, is_graspable, min_dimension)
                 grasp_axis: 夹持轴索引（0=X,1=Y,2=Z），-1表示不可抓取
                 is_graspable: 是否可抓取（bool）
                 min_dimension: 物体最短边长度（米）
        """
        aabb_length_x, aabb_width_y, aabb_height_z = aabb_dimensions
        # 找到最短边（最优夹持方向）
        min_dimension = min(aabb_length_x, aabb_width_y, aabb_height_z)
        grasp_axis = -1

        # 可抓取性判断：最短边超过机械爪最大张开距离则不可抓取
        if min_dimension > gripper_max_opening:
            print(f"警告：物体不可抓取！最短边={min_dimension:.3f}m > 机械爪最大张开={gripper_max_opening:.3f}m")
            return grasp_axis, False, min_dimension

        # 确定最短边对应的夹持轴
        if min_dimension == aabb_length_x:
            grasp_axis = 0  # X轴为夹持方向
        elif min_dimension == aabb_width_y:
            grasp_axis = 1  # Y轴为夹持方向
        else:
            grasp_axis = 2  # Z轴为夹持方向

        print(f"可抓取！夹持轴={grasp_axis}（0=X,1=Y,2=Z），最短边={min_dimension:.3f}m")
        return grasp_axis, True, min_dimension

    def compute_grasp_pose(self, obb, gripper_max_opening, frame_id="camera_color_optical_frame"):
        """
        抓取姿态计算（位置 + 旋转）
        :param aabb_center_local: PCA局部坐标系下AABB盒中心 [x, y, z]（numpy数组）
        :param tm_inv: 逆变换矩阵（4x4，numpy数组），用于将局部坐标转换为世界坐标
        :param grasp_axis: 夹持轴索引（0=X,1=Y,2=Z）
        :param frame_id: 坐标系ID（ROS兼容）
        :return: grasp_pose（字典格式，兼容ROS PoseStamped）
                 grasp_pose = {
                     "header": {"frame_id": frame_id, "stamp": None},
                     "pose": {
                         "position": {"x": x, "y": y, "z": z},
                         "orientation": {"x": x, "y": y, "z": z, "w": w}
                     }
                 }
        """
        # ================== 3. 关键：从OBB提取PCA相关参数（核心步骤） ===========
        # OBB包含了PCA主方向、局部AABB中心、变换矩阵等关键信息，直接从obb中提取
        aabb_dimensions = [
            obb.extent[0],  # X轴长度（局部坐标系）
            obb.extent[1],  # Y轴长度（局部坐标系）
            obb.extent[2]   # Z轴长度（局部坐标系）
        ]

        grasp_axis, is_graspable, min_dim = self.select_grasp_axis(
            aabb_dimensions=aabb_dimensions,
            gripper_max_opening=gripper_max_opening
        )

        if not is_graspable:
            return None

        # 3.3 构建逆变换矩阵tm_inv（4x4)
        # OBB的旋转矩阵（3x3）+ 平移向量（3x1）→ 4x4变换矩阵
        tm_inv = np.eye(4, dtype=np.float64)
        tm_inv[:3, :3] = np.array(obb.R)  # OBB的旋转矩阵（PCA主方向）
        tm_inv[:3, 3] = np.array(obb.center)  # OBB的中心（平移向量）        

        grasp_pose = {
            "header": {"frame_id": frame_id, "stamp": None},  # stamp可在发布时填充ROS时间
            "pose": {"position": {}, "orientation": {}}
        }

        aabb_center_local=obb.center
        # ===================== 1. 计算抓取位置（AABB盒中心，转换到世界坐标系） =====================
        aabb_center_local = np.array(aabb_center_local, dtype=np.float64).reshape(3, 1)
        # 提取逆变换矩阵的旋转部分（3x3）和平移部分（3x1）
        tm_inv_rot = tm_inv[:3, :3]
        tm_inv_trans = tm_inv[:3, 3].reshape(3, 1)
        # 局部坐标 -> 世界坐标：P_global = R * P_local + T
        aabb_center_global = tm_inv_rot @ aabb_center_local + tm_inv_trans

        # 填充抓取位置
        grasp_pose["pose"]["position"]["x"] = float(aabb_center_global[0, 0])
        grasp_pose["pose"]["position"]["y"] = float(aabb_center_global[1, 0])
        grasp_pose["pose"]["position"]["z"] = float(aabb_center_global[2, 0])

        # ===================== 2. 计算抓取旋转（基于PCA主方向，调整夹持轴） =====================
        # 获取原始旋转矩阵（从逆变换矩阵中提取）
        rotation_matrix = tm_inv_rot.copy()

        # 根据夹持轴调整旋转矩阵（确保机械爪Z轴为夹持方向）
        if grasp_axis == 0:
            # 新X=原Y，新Y=原Z，新Z=原X（夹持方向）
            adjusted_rot = np.zeros_like(rotation_matrix)
            adjusted_rot[:, 0] = rotation_matrix[:, 1]
            adjusted_rot[:, 1] = rotation_matrix[:, 2]
            adjusted_rot[:, 2] = rotation_matrix[:, 0]
            rotation_matrix = adjusted_rot
        elif grasp_axis == 1:
            # 新X=原Z，新Y=原X，新Z=原Y（夹持方向）
            adjusted_rot = np.zeros_like(rotation_matrix)
            adjusted_rot[:, 0] = rotation_matrix[:, 2]
            adjusted_rot[:, 1] = rotation_matrix[:, 0]
            adjusted_rot[:, 2] = rotation_matrix[:, 1]
            rotation_matrix = adjusted_rot
        # grasp_axis == 2 时，无需调整

        # ===================== 3. 修正Z轴方向：朝向远离相机原点 =====================
        z_axis = rotation_matrix[:, 2]
        position_vector = aabb_center_global.reshape(3,)
        # 计算Z轴与位置向量的点积
        position_vector_normalized = position_vector / np.linalg.norm(position_vector)
        dot_product = np.dot(z_axis, position_vector_normalized)

        # 点积为负，翻转Z轴和X轴（保持右手坐标系）
        if dot_product < 0:
            print(f"翻转Z轴（点积={dot_product:.3f} < 0）")
            rotation_matrix[:, 2] = -rotation_matrix[:, 2]
            rotation_matrix[:, 0] = -rotation_matrix[:, 0]

        # ===================== 4. 旋转矩阵正交化修正（消除计算误差） =====================
        determinant = np.linalg.det(rotation_matrix)
        if abs(determinant - 1.0) > 0.1:
            print(f"旋转矩阵行列式异常（{determinant:.3f}），正交化修正...")
            # QR分解正交化
            Q, R = qr(rotation_matrix)
            rotation_matrix = Q
            # 确保右手坐标系（行列式>0）
            if np.linalg.det(rotation_matrix) < 0:
                rotation_matrix[:, 2] = -rotation_matrix[:, 2]

        # ===================== 5. 旋转矩阵转四元数 =====================
        quat_w, quat_x, quat_y, quat_z = tfq.mat2quat(rotation_matrix)

        # 归一化四元数
        quat_norm = np.sqrt(quat_x**2 + quat_y**2 + quat_z**2 + quat_w**2)
        quat_x /= quat_norm
        quat_y /= quat_norm
        quat_z /= quat_norm
        quat_w /= quat_norm

        # 填充抓取姿态（注意：ROS四元数顺序是x,y,z,w）
        grasp_pose["pose"]["orientation"]["x"] = float(quat_x)
        grasp_pose["pose"]["orientation"]["y"] = float(quat_y)
        grasp_pose["pose"]["orientation"]["z"] = float(quat_z)
        grasp_pose["pose"]["orientation"]["w"] = float(quat_w)

        print(f"抓取位置：x={grasp_pose['pose']['position']['x']:.3f}, y={grasp_pose['pose']['position']['y']:.3f}, z={grasp_pose['pose']['position']['z']:.3f}")
        print(f"抓取四元数：x={quat_x:.3f}, y={quat_y:.3f}, z={quat_z:.3f}, w={quat_w:.3f}")
        return grasp_pose

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
        self.ik_sub = rospy.Subscriber("/ik_status", Bool, self.ik_status_callback)
        
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
        # 初始化ROS节点
        rospy.init_node('piper_pose_transformer', anonymous=True)

        # 初始化tf2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 订阅和发布
        self.sub = rospy.Subscriber("/grasp_pose_posestamp", PoseStamped, self.pose_callback)
        self.target_pose_sub = rospy.Subscriber("/target_pose", PoseStamped, self.target_pose_callback)
        self.target_pose_pub = rospy.Publisher('/target_pose', PoseStamped, queue_size=10)
        self.gripper_cmd_pub = rospy.Publisher('/gripper_cmd_topic', Float64, queue_size=1)
        self.outer_cmd_sub = rospy.Subscriber("/vision_task", Int32, self.tast_callback)
        self.task_reslut_pub = rospy.Publisher('/vision_result', Int32, queue_size=1)

        self.rgb_depth_rw_lock = ReadWriteLock()

        self.image_sub = Subscriber("/cam_arm/camera/color/image_raw", Image)
        self.depth_sub = Subscriber("/cam_arm/camera/aligned_depth_to_color/image_raw", Image)
        self.camera_info_sub = Subscriber("/cam_arm/camera/aligned_depth_to_color/camera_info", CameraInfo)

        self.camera_info_sub.registerCallback(self.camera_info_callback)

        self.syncronizer = ApproximateTimeSynchronizer([self.image_sub, self.depth_sub], 1, 0.1)
        self.syncronizer.registerCallback(self.rgb_depth_down_callback)
        
        # 逆解状态管理
        self.ik_manager = IKStatusManager()
        self.pose_adjuster = PoseAdjuster()
        
        self.rate = rospy.Rate(30)

        self.cv_bridge = CvBridge()
        self.rgb_image = None
        self.rgb_bytes = None
        self.depth_image = None
        self.depth_bytes = None
        self.new_image_arrived = False
        self.rgb_time = 0.0

        # 相机内参存储（关键：用于2D转3D）
        self.fx = None
        self.fy = None
        self.ppx = None
        self.ppy = None

        self.yolo_model = YOLO('./yolo11n-seg.pt')

        self.grasp_calc = GraspPoseCalculator()
        
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

        # 外部请求管理
        self.task_cmd = 0
        self.task_reslut = 0
        
        # 动作序列管理
        self.action_sequence = ActionSequence()
        
        # 预定义路径点
        self.via_point = self.create_via_pose(
            position=[0.300, 0.0, 0.360],
            orientation=[0.007, 0.915, 0.009, 0.403]
        )

        self.target_pose_current = PoseStamped()
        self.via_pose_list = []
        
        # 终端设置
        self.old_settings = termios.tcgetattr(sys.stdin)
        
        rospy.loginfo("Pose transformer node started. Waiting for PoseStamped messages...")

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
            B
            
            return transformed_pose
            
        except (tf2_ros.LookupException, 
                tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("坐标变换失败: %s", str(e))
            return None

    def camera_info_callback(self, camera_info_msg):
        """
        解析相机内参（从CameraInfo消息提取fx, fy, ppx, ppy）
        内参矩阵K：[fx, 0, ppx; 0, fy, ppy; 0, 0, 1]
        """
        self.fx = camera_info_msg.K[0]
        self.fy = camera_info_msg.K[4]
        self.ppx = camera_info_msg.K[2]
        self.ppy = camera_info_msg.K[5]
        rospy.loginfo("已获取相机内参：fx=%.2f, fy=%.2f, ppx=%.2f, ppy=%.2f",
                      self.fx, self.fy, self.ppx, self.ppy)
    
    def rgb_depth_down_callback(self, rgb_msg, depth_msg):

        print("======================, rgb_depth_down_callback, ")
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
        self.new_image_arrived = True

        print("++++++++++++++++++++++++++++", self.new_image_arrived)

        #self.process_object_3d_data()

    def pixel_to_3d(self, u, v, z):
        """
        2D像素坐标转3D世界坐标（适配你的depth_image已为米单位）
        :param u: 像素横坐标（列）
        :param v: 像素纵坐标（行）
        :param z: 深度值（米，已由self.depth_image提供）
        :return: (x, y, z) 世界坐标（米）
        """
        if self.fx is None or self.fy is None or self.ppx is None or self.ppy is None:
            rospy.logwarn("相机内参未初始化，无法转换3D坐标")
            return 0, 0, 0
        # 针孔相机模型逆运算
        x = (u - self.ppx) * z / self.fx
        y = (v - self.ppy) * z / self.fy
        return x, y, z
    
    def yolo_segmentation(self, rgb_image):
        """
        Yolo-based object detection and segmentation
        :param rgb_image: RGB image（rgb8 format，numpy [H, W, 3]）
        """
        results = self.yolo_model(
            rgb_image,
            #conf=0.5,  # 过滤置信度<0.5的结果，可调整
            #iou=0.45,
            #classes=self.target_classes
        )

        single_result = results[0] # get result for the 1st image
        if single_result.masks is None or len(single_result.masks) == 0:
            rospy.logwarn("YOLO No Object Found!")
            return None, None, None, None

        confs = single_result.boxes.conf.cpu().numpy()
        boxes = single_result.boxes.xyxy.cpu().numpy().astype(int)
        cls_ids = single_result.boxes.cls.cpu().numpy()

        masks = None
        if single_result.masks is not None:
            masks = single_result.masks.data.cpu().numpy()

        conf_with_idx = list(enumerate(confs))  # e.g. [(0, 0.95), (1, 0.88), ...]

        # sort by conf
        conf_with_idx_sorted = sorted(conf_with_idx, key=lambda x: x[1], reverse=True)
        sorted_indices = [idx for idx, conf in conf_with_idx_sorted]

        sorted_confs = confs[sorted_indices]
        sorted_boxes = boxes[sorted_indices]
        sorted_cls_ids = cls_ids[sorted_indices]
        sorted_masks = masks[sorted_indices] if masks is not None else None

        vis_image = single_result.plot()
        save_path = "./segment_result.jpg"
        vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, vis_image_bgr)
        return sorted_boxes, sorted_confs, sorted_cls_ids, sorted_masks
    
    def get_object_3d_data(self, bbox, mask, rgb_image, depth_image):
        """
        从2D检测结果提取目标3D点云、3D包围框
        :param bbox: 2D矩形框 [x1, y1, x2, y2]
        :param mask: 2D目标掩码
        :return: target_pcd（3D点云）、aabb（轴对齐3D包围框）、obb（定向3D包围框）
        """
        if rgb_image is None or depth_image is None:
            rospy.logwarn("RGB/深度图像无效，无法提取3D数据")
            return None, None, None
        if bbox is None or mask is None:
            rospy.logwarn("2D检测结果无效，无法提取3D数据")
            return None, None, None

        x1, y1, x2, y2 = map(int, bbox)
        # 裁剪ROI区域（提升计算效率，仅处理目标区域）
        roi_mask = mask[y1:y2, x1:x2]
        roi_depth = depth_image[y1:y2, x1:x2]
        roi_rgb = rgb_image[y1:y2, x1:x2]

        # 获取ROI内目标像素的坐标（行、列）
        u_roi, v_roi = np.where(roi_mask > 0)
        # 转换为原始图像的像素坐标
        u = u_roi + y1  # 原始图像纵坐标（行）
        v = v_roi + x1  # 原始图像横坐标（列）

        # 提取对应深度值（米单位）和RGB颜色
        z_values = depth_image[u, v]
        rgb_values = roi_rgb[u_roi, v_roi]

        # 过滤无效数据（深度<=0为无效）
        valid_mask = z_values > 0
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        z_valid = z_values[valid_mask]
        rgb_valid = rgb_values[valid_mask]

        if len(z_valid) == 0:
            rospy.logwarn("目标区域无有效深度值，无法生成3D点云")
            return None, None, None

        # 批量转换2D像素到3D世界坐标
        num_points = len(z_valid)
        point_3d = np.zeros((num_points, 3), dtype=np.float64)
        for i in range(num_points):
            x, y, z = self.pixel_to_3d(v_valid[i], u_valid[i], z_valid[i])
            point_3d[i] = [x, y, z]

        # 构建Open3D点云
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(point_3d)
        # 设置点云颜色（rgb8格式归一化到0-1）
        target_pcd.colors = o3d.utility.Vector3dVector(rgb_valid / 255.0)

        # 计算3D包围框
        aabb = target_pcd.get_axis_aligned_bounding_box()
        aabb.color = (1, 0, 0)  # 红色：轴对齐包围框
        obb = target_pcd.get_oriented_bounding_box()
        obb.color = (0, 1, 0)  # 绿色：定向包围框
        return target_pcd, aabb, obb

    def process_object_3d_data(self):
        """
        核心处理函数：串联2D检测->3D数据提取->可视化
        """
        if not self.new_image_arrived:
            print("self.new_image_arrived, ", self.new_image_arrived)
            return
        if self.fx is None or self.fy is None or self.ppx is None or self.ppy is None:
            rospy.logwarn("相机内参未就绪，跳过3D处理")
            return
        if self.rgb_image is None or self.depth_image is None:
            rospy.logwarn("RGB/深度图像未就绪，跳过3D处理")
            return

        # save 创建 RGBD 图像（Open3D 格式）, convert_rgb_to_intensity=False：保留彩色信息（否则转为灰度图）
        rgb_o3d = o3d.geometry.Image(self.rgb_image)
        depth_o3d = o3d.geometry.Image(self.depth_image.astype(np.float32))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d,
            depth_o3d,
            depth_scale=1.0,    # 深度值缩放（mm → m）
            depth_trunc=3.0,    # 深度截断
            convert_rgb_to_intensity=False
        )

        # 3. 定义相机内参（Open3D 格式）
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        # 内参赋值：w, h, fx, fy, cx, cy
        h, w = self.rgb_image.shape[:2]
        intrinsic.set_intrinsics(w, h, self.fx, self.fy, self.ppx, self.ppy)

        orig_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            intrinsic
        )

        # 5. 坐标系翻转（可选，适配 Open3D 可视化视角）
        # Open3D 默认相机坐标系与视觉习惯略有差异，翻转后更直观
        #orig_pcd.transform([[1, 0, 0, 0],
        #            [0, -1, 0, 0],
        #            [0, 0, -1, 0],
        #            [0, 0, 0, 1]])

        o3d.io.write_point_cloud("rgbd_point_cloud.ply", orig_pcd)

        # object detection and segmentation
        sorted_boxes, sorted_confs, sorted_cls_ids, sorted_masks = self.yolo_segmentation(self.rgb_image)

        if sorted_boxes is None or sorted_masks is None:
            return
        else:
            target_pcd_list = []
            aabb_list = []
            obb_list = []
            for idx in range(len(sorted_boxes)):
                bbox = sorted_boxes[idx]
                mask = sorted_masks[idx]
                target_pcd, aabb, obb = self.get_object_3d_data(bbox, mask, self.rgb_image, self.depth_image)
                if target_pcd is None:
                    continue
                else:
                    target_pcd_list.append(target_pcd)
                    aabb_list.append(aabb)
                    obb_list.append(obb)

            h, w = self.rgb_image.shape[:2]
            self.visualize_pcd_with_boxes_offline(orig_pcd, aabb_list, obb_list, w, h)
            #self.visualize_results(target_pcd, aabb, obb)

            gripper_max_opening = 0.05  # 机械爪最大张开距离（米），根据实际硬件调整（如0.1米）
            frame_id = "camera_color_optical_frame"  # 坐标系ID（与你的点云坐标系一致）
            grasp_pose = self.grasp_calc.compute_grasp_pose(
                obb,
                gripper_max_opening=gripper_max_opening,
                frame_id=frame_id
            )

            if grasp_pose is not None:
                print("\n===== 最终抓取姿态 =====")
                print(f"坐标系：{grasp_pose['header']['frame_id']}")
                print(f"抓取位置：x={grasp_pose['pose']['position']['x']:.3f}m, y={grasp_pose['pose']['position']['y']:.3f}m, z={grasp_pose['pose']['position']['z']:.3f}m")
                print(f"抓取四元数：x={grasp_pose['pose']['orientation']['x']:.3f}, y={grasp_pose['pose']['orientation']['y']:.3f}, z={grasp_pose['pose']['orientation']['z']:.3f}, w={grasp_pose['pose']['orientation']['w']:.3f}")

                self.export_grasp_visualization_to_ply(orig_pcd, grasp_pose, "./grasp_visualization.ply", aabb_list, obb_list, 0.005)

    def visualize_pcd_with_boxes_offline(self, pcd, aabb_list, obb_list,
        img_width: int = 800,
        img_height: int = 600,
        camera_position: tuple = (2.0, 2.0, 2.0),
        lookat: tuple = (0.0, 0.0, 0.0),
        output_img_path: str = "pcd_boxes_render.png"
    ):
        """
        离线渲染点云+AABB+OBB包围盒（不依赖OpenGL窗口）
        :param pcd_path: 点云文件路径（.ply/.pcd等），若指定则忽略pcd_points
        :param pcd_points: 点云数据（N×3的numpy数组），优先级低于pcd_path
        :param output_img_path: 渲染图像保存路径
        :param img_width/height: 渲染图像尺寸
        :param camera_position: 相机位置
        :param lookat: 相机看向的中心点
        """
        renderer = o3d.visualization.rendering.OffscreenRenderer(img_width, img_height)
        renderer.scene.set_background([1.0, 1.0, 1.0, 1.0]) 
        renderer.scene.scene.set_sun_light(
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            100000
        )
        renderer.scene.scene.enable_sun_light(False)

        # 点云材质
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = 2.0
        # 补充点云默认颜色（避免点云透明）
        material.base_color = [0.5, 0.5, 0.5, 1.0]  # 灰色点云，不透明
        renderer.scene.add_geometry("point_cloud", pcd, material)

        # 包围盒线框材质
        box_material = o3d.visualization.rendering.MaterialRecord()
        box_material.shader = "unlitLine"
        box_material.line_width = 5.0
        # 关键：配置线框颜色（RGBA，避免透明）
        box_material.base_color = [1.0, 0.0, 0.0, 1.0]  # 红色线框，不透明

        print("======= appending ", len(aabb_list))
        # 添加包围盒线框
        box_annotations = []  # 存储包围盒标注信息
        for idx in range(len(aabb_list)):
            aabb = aabb_list[idx]
            obb = obb_list[idx]
            aabb_name = f"aabb_box_{idx}"
            obb_name = f"obb_box_{idx}"

            aabb_lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)
            obb_lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)

            # 线框颜色
            colors = [[1,0,0], [0,1,0], [0,0,1], [1,1,0]]
            line_color = colors[idx % len(colors)]
            aabb_lines.colors = o3d.utility.Vector3dVector([line_color for _ in aabb_lines.lines])
            obb_lines.colors = o3d.utility.Vector3dVector([line_color for _ in obb_lines.lines])

            # 添加到渲染场景
            renderer.scene.add_geometry(aabb_name, aabb_lines, box_material)
            renderer.scene.add_geometry(obb_name, obb_lines, box_material)

            axis_xyz = [(obb.R[:,i] * obb.extent[i]).tolist() for i in range(3)]
            box_annotations.append({
                "index": idx,
                # AABB参数：中心点、最小/最大边界、尺寸
                "aabb": {
                    "center": aabb.get_center().tolist(),
                    "min_bound": aabb.min_bound.tolist(),
                    "max_bound": aabb.max_bound.tolist(),
                    "extent": aabb.get_extent().tolist(),
                    "color": line_color
                },
                # OBB参数：中心点、旋转矩阵、尺寸、轴
                "obb": {
                    "center": obb.get_center().tolist(),
                    "R": obb.R.tolist(),  # 旋转矩阵
                    "extent": obb.extent.tolist(),
                    "axis_xyz": axis_xyz,  # 三个轴
                    "color": line_color
                }
            })

        # 相机配置（修复后的参数）
        bounding_box = pcd.get_axis_aligned_bounding_box()
        center = bounding_box.get_center()
        extent = bounding_box.get_extent()
        max_extent = np.max(extent)  # 点云最大尺寸
        camera = renderer.scene.camera
        camera.look_at(
            center,  # 目标点（点云中心）
            center + [0, -max_extent*2, max_extent*1.5],  # 眼点（斜上方）
            [0, 1, 0]  # 上方向
        )
        fov_type = o3d.visualization.rendering.Camera.FovType.Horizontal
        camera.set_projection(
            60,                # 水平视野角度（°）
            (1.0*img_width)/img_height,  # 宽高比
            0.1,               # 近裁剪面（最近可见距离）
            1000.0,            # 远裁剪面（最远可见距离）
            fov_type           # 视野类型（补充的关键参数）
        )

        # 8. 渲染图像（返回RGBA格式）
        img_rgba = renderer.render_to_image()
        img_rgb = np.asarray(img_rgba)[:, :, :3]
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_img_path, img_bgr)
        print(f"渲染结果已保存到：{output_img_path}")
   
        output_json_path = "./ann.json"
        output_pcd_path = "./pcd.ply"
        # ========== 5. 保存AABB/OBB标注信息（JSON格式） ==========
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(box_annotations, f, indent=4)
        print(f"包围盒标注信息已保存到：{output_json_path}")

        # ========== （可选）保存包含包围盒线框的组合点云 ==========
        # 合并原始点云 + 所有包围盒线框，保存为一个文件
        combined_geometry = o3d.geometry.PointCloud()  # 新建空点云

        # 1. 复制原始点云的坐标（核心）
        pcd_points = np.asarray(pcd.points)  # 转为numpy数组
        combined_geometry.points = o3d.utility.Vector3dVector(pcd_points)

        # 2. 可选：复制点云颜色（如果原始点云有颜色）
        if pcd.has_colors():
            pcd_colors = np.asarray(pcd.colors)
            combined_geometry.colors = o3d.utility.Vector3dVector(pcd_colors)

        # 3. 可选：复制点云法线（如果原始点云有法线）
        if pcd.has_normals():
            pcd_normals = np.asarray(pcd.normals)
            combined_geometry.normals = o3d.utility.Vector3dVector(pcd_normals)

        for idx in range(len(aabb_list)):
            aabb = aabb_list[idx]
            obb = obb_list[idx]
            aabb_lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)
            obb_lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
            # 将线框的点添加到组合点云中（可选：给包围盒点赋颜色）
            aabb_points = np.asarray(aabb_lines.points)
            obb_points = np.asarray(obb_lines.points)
            combined_points = np.vstack([np.asarray(combined_geometry.points), aabb_points, obb_points])
            combined_geometry.points = o3d.utility.Vector3dVector(combined_points)
        # 保存组合点云
        combined_pcd_path = output_pcd_path.replace(".ply", "_with_boxes.ply")
        o3d.io.write_point_cloud(combined_pcd_path, combined_geometry)
        print(f"包含包围盒点的组合点云已保存到：{combined_pcd_path}")

        renderer.scene.remove_geometry("point_cloud")
        renderer.scene.remove_geometry("aabb_box")
        renderer.scene.remove_geometry("obb_box")
        del renderer

    def export_grasp_visualization_to_ply(pcd, grasp_pose, output_ply_path="grasp_visualization.ply",
                                        aabb=None, obb=None, axis_point_size=0.005):
        """
        将点云、AABB/OBB包围盒、抓取姿态坐标系整合为单个PLY文件（适配CloudCompare查看）
        
        参数说明：
        - pcd: open3d.geometry.PointCloud 对象（原始点云）
        - grasp_pose: 抓取姿态，支持字典/ROS PoseStamped 格式
        - output_ply_path: 输出PLY文件路径（CloudCompare可直接打开）
        - aabb: open3d.geometry.AxisAlignedBoundingBox 对象（可选）
        - obb: open3d.geometry.OrientedBoundingBox 对象（可选）
        - axis_point_size: 坐标系轴的点大小（默认0.005米，CloudCompare中可见）
        """
        # 1. 输入校验
        if not isinstance(pcd, o3d.geometry.PointCloud) or len(pcd.points) == 0:
            print("错误：输入点云不合法（非Open3D PointCloud或为空）")
            return
        
        # 2. 解析抓取姿态的位置和旋转
        try:
            if isinstance(grasp_pose, dict):
                pos = grasp_pose["pose"]["position"]
                ori = grasp_pose["pose"]["orientation"]
                pos_list = np.array([pos["x"], pos["y"], pos["z"]])
                quat_list = [ori["x"], ori["y"], ori["z"], ori["w"]]
            else:
                pos = grasp_pose.pose.position
                ori = grasp_pose.pose.orientation
                pos_list = np.array([pos.x, pos.y, pos.z])
                quat_list = [ori.x, ori.y, ori.z, ori.w]
        except Exception as e:
            print(f"错误：解析抓取姿态失败 - {e}")
            return
        
        # 3. 创建合并后的点云（原始点云 + 所有可视化元素）
        combined_pcd = o3d.geometry.PointCloud()
        
        # 3.1 复制原始点云（保留原始颜色/坐标）
        combined_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
        if pcd.has_colors():
            combined_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))
        else:
            # 原始点云默认设为灰色
            combined_pcd.colors = o3d.utility.Vector3dVector(
                np.ones((len(pcd.points), 3)) * 0.5
            )
        
        # 3.2 添加AABB包围盒顶点（红色）
        if aabb is not None and isinstance(aabb, o3d.geometry.AxisAlignedBoundingBox):
            aabb_points = np.asarray(aabb.get_box_points())  # 获取AABB8个顶点
            aabb_colors = np.tile([1.0, 0.0, 0.0], (len(aabb_points), 1))  # 红色
            # 添加到合并点云
            combined_pcd.points.extend(o3d.utility.Vector3dVector(aabb_points))
            combined_pcd.colors.extend(o3d.utility.Vector3dVector(aabb_colors))
        
        # 3.3 添加OBB包围盒顶点（绿色）
        if obb is not None and isinstance(obb, o3d.geometry.OrientedBoundingBox):
            obb_points = np.asarray(obb.get_box_points())  # 获取OBB8个顶点
            obb_colors = np.tile([0.0, 1.0, 0.0], (len(obb_points), 1))  # 绿色
            # 添加到合并点云
            combined_pcd.points.extend(o3d.utility.Vector3dVector(obb_points))
            combined_pcd.colors.extend(o3d.utility.Vector3dVector(obb_colors))
        
        # 3.4 添加抓取姿态坐标系（轴长0.1米，X红/Y绿/Z蓝）
        axis_length = 0.1
        # 生成坐标系轴的点（从原点到轴端点，密集点保证CloudCompare中可见）
        num_points_per_axis = 50  # 每个轴生成50个点，避免轴显示为单个点
        # 旋转矩阵：将局部坐标系转为世界坐标系
        rot_mat = o3d.geometry.get_rotation_matrix_from_quaternion(quat_list)
        
        # X轴（红色）：从抓取位置沿X轴延伸axis_length
        x_axis_points = np.linspace(pos_list, pos_list + rot_mat[:, 0] * axis_length, num_points_per_axis)
        x_axis_colors = np.tile([1.0, 0.0, 0.0], (num_points_per_axis, 1))
        combined_pcd.points.extend(o3d.utility.Vector3dVector(x_axis_points))
        combined_pcd.colors.extend(o3d.utility.Vector3dVector(x_axis_colors))
        
        # Y轴（绿色）：从抓取位置沿Y轴延伸axis_length
        y_axis_points = np.linspace(pos_list, pos_list + rot_mat[:, 1] * axis_length, num_points_per_axis)
        y_axis_colors = np.tile([0.0, 1.0, 0.0], (num_points_per_axis, 1))
        combined_pcd.points.extend(o3d.utility.Vector3dVector(y_axis_points))
        combined_pcd.colors.extend(o3d.utility.Vector3dVector(y_axis_colors))
        
        # Z轴（蓝色）：从抓取位置沿Z轴延伸axis_length
        z_axis_points = np.linspace(pos_list, pos_list + rot_mat[:, 2] * axis_length, num_points_per_axis)
        z_axis_colors = np.tile([0.0, 0.0, 1.0], (num_points_per_axis, 1))
        combined_pcd.points.extend(o3d.utility.Vector3dVector(z_axis_points))
        combined_pcd.colors.extend(o3d.utility.Vector3dVector(z_axis_colors))
        
        # 3.5 添加抓取位置中心点（黄色）
        center_point = np.array([pos_list])
        center_color = np.array([[1.0, 1.0, 0.0]])  # 黄色
        combined_pcd.points.extend(o3d.utility.Vector3dVector(center_point))
        combined_pcd.colors.extend(o3d.utility.Vector3dVector(center_color))
        
        # 4. 保存为PLY文件（CloudCompare原生支持）
        o3d.io.write_point_cloud(output_ply_path, combined_pcd, write_ascii=True)
        # 补充：可选保存为PCD格式（CloudCompare也支持）
        # o3d.io.write_point_cloud(output_ply_path.replace(".ply", ".pcd"), combined_pcd)
        
        print(f"3D可视化文件已生成：{output_ply_path}")
        print("提示：可用CloudCompare打开该文件，查看点云+包围盒+抓取姿态坐标系")
    
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

    def tast_callback(self, msg):
        """外部请求task_cmd消息回调"""
        self.task_cmd = msg


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

    def reset_ik_check_state(self):
        """重置逆解检查状态"""
        self.ik_manager.ik_status_received = False
        self.ik_manager.ik_success = False
        self.waiting_for_ik = True
        self.ik_check_start_time = time.time()
        self.adjustment_attempts = 0
        self.ik_success_pose = None
        self.motion_in_progress = False

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

    def check_ik_status(self):
        """检查逆解状态并处理"""
        if not self.waiting_for_ik:
            return None
        
        current_time = time.time()
        
        # 检查是否收到逆解状态
        if self.ik_manager.ik_status_received:
            if self.ik_manager.ik_success:
                # 逆解成功，设置运动进行中标志
                rospy.loginfo("IK solution found successfully! Motion started...")
                self.ik_success_pose = self.current_target_pose
                self.waiting_for_ik = False
                self.motion_in_progress = True
                self.motion_complete_time = current_time + 1.5  # 假设运动需要5秒完成
                return True
            else:
                # 逆解失败，等待1秒后尝试下一个调整
                if current_time - self.ik_check_start_time > 0.01:
                    self.adjustment_attempts += 1
                    
                    if self.adjustment_attempts >= self.max_adjustment_attempts:
                        rospy.logerr("Maximum adjustment attempts reached, giving up")
                        self.waiting_for_ik = False
                        return False
                    
                    # 尝试下一个调整值
                    rospy.loginfo(f"Waiting 1 second before next adjustment...")
                    self.ik_manager.ik_status_received = False
                    self.ik_check_start_time = time.time()
                    
                    # 发布下一个调整姿态
                    return self.publish_adjusted_pose(self.adjustment_attempts)
        
        return None

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

    def wait(self, duration=1.0):
        """等待指定时间"""
        rospy.loginfo(f"Waiting for {duration} seconds")
        rospy.sleep(duration)
        return True

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
        '''image = PIL_Image.fromarray(self.rgb_image).convert('RGB')
        results = self.yolo_model(image)

        res_plotted = results[0].plot()
        #res_plotted_bgr = cv2.cvtColor(res_plotted, cv2.COLOR_RGB2BGR)

        image.save("./input_img.png")

        seg_vis_pil = PIL_Image.fromarray(res_plotted).convert('RGB')
        seg_vis_pil.save("./vis_img.png")'''

        self.process_object_3d_data()

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

    def get_key(self):
        """获取键盘输入"""
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1)
        return None

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

    def handle_key_input(self, key):
        """处理键盘输入"""
        key_actions = {
            's': self.execute_grasp_sequence,
            'p': lambda: setattr(self, 'continuous_publishing', not self.continuous_publishing),
            'b': lambda: [self.control_gripper(0.07), self.publish_pose_with_ik_check(self.via_point)],
            't': lambda: self.control_gripper(0.0 if self.gripper_cmd >= 0.06 else 0.07),
            'c': self.action_sequence.clear,
            'i': self.print_instructions,
            'r': self.record_search_route,
            'q': self.search_mode
        }
        
        if key in key_actions:
            key_actions[key]()
            
            if key == 'p':
                state = "Started" if self.continuous_publishing else "Stopped"
                rospy.loginfo(f"{state} continuous publishing")
            elif key == 'c':
                rospy.loginfo("Action sequence cleared")
        
        if self.task_cmd:
            self.execute_grasp_sequence()

    def run(self):
        """主循环"""
        tty.setcbreak(sys.stdin.fileno())
        self.print_instructions()
        
        try:
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
                
                self.rate.sleep()
                
        finally:
            # 恢复终端设置
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

if __name__ == '__main__':
    try:
        transformer = PoseTransformer()
        transformer.run()
    except rospy.ROSInterruptException:
        pass
