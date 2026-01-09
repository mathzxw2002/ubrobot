import time
import cv2
import open3d as o3d
from ultralytics import YOLO

from scipy.linalg import qr
import transforms3d.quaternions as tfq
import numpy as np

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

        grasp_pose = {
            "header": {"frame_id": frame_id, "stamp": None},  # stamp可在发布时填充ROS时间
            "pose": {"position": {}, "orientation": {}}
        }
        grasp_pose["pose"]["position"]["x"] = float(obb.center[0])
        grasp_pose["pose"]["position"]["y"] = float(obb.center[1])
        grasp_pose["pose"]["position"]["z"] = float(obb.center[2])

        # ================计算抓取旋转（基于PCA主方向，调整夹持轴） =====================
        # 获取原始旋转矩阵（从逆变换矩阵中提取）
        # OBB的旋转矩阵（3x3）+ 平移向量（3x1）→ 4x4变换矩阵
        tm_world2obb = np.eye(4, dtype=np.float64)
        tm_world2obb[:3, :3] = np.array(obb.R)
        tm_world2obb[:3, 3] = np.array(obb.center)
       
        R_world2obb = tm_world2obb[:3, :3]  # 世界→OBB的旋转矩阵
        R_obb2world = R_world2obb.T  # 逆旋转 = 转置（核心！）

        rotation_matrix = R_obb2world.copy()

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

        # ===================== 修正Z轴方向：朝向远离相机原点 =====================
        z_axis = rotation_matrix[:, 2]
        position_vector = obb.center.reshape(3,)
        # 计算Z轴与位置向量的点积
        position_vector_normalized = position_vector / np.linalg.norm(position_vector)
        dot_product = np.dot(z_axis, position_vector_normalized)

        # 点积为负，翻转Z轴和X轴（保持右手坐标系）
        if dot_product < 0:
            print(f"翻转Z轴（点积={dot_product:.3f} < 0）")
            rotation_matrix[:, 2] = -rotation_matrix[:, 2]
            rotation_matrix[:, 0] = -rotation_matrix[:, 0]

        # ===================== 旋转矩阵正交化修正（消除计算误差） =====================
        determinant = np.linalg.det(rotation_matrix)
        if abs(determinant - 1.0) > 0.1:
            print(f"旋转矩阵行列式异常（{determinant:.3f}），正交化修正...")
            # QR分解正交化
            Q, R = qr(rotation_matrix)
            rotation_matrix = Q
            # 确保右手坐标系（行列式>0）
            if np.linalg.det(rotation_matrix) < 0:
                rotation_matrix[:, 2] = -rotation_matrix[:, 2]

        # ===================== 旋转矩阵转四元数 =====================
        quat_w, quat_x, quat_y, quat_z = tfq.mat2quat(rotation_matrix)
        quat_norm = np.sqrt(quat_x**2 + quat_y**2 + quat_z**2 + quat_w**2)
        quat_x /= quat_norm
        quat_y /= quat_norm
        quat_z /= quat_norm
        quat_w /= quat_norm
        grasp_pose["pose"]["orientation"]["x"] = float(quat_x)
        grasp_pose["pose"]["orientation"]["y"] = float(quat_y)
        grasp_pose["pose"]["orientation"]["z"] = float(quat_z)
        grasp_pose["pose"]["orientation"]["w"] = float(quat_w)

        print(f"抓取位置：x={grasp_pose['pose']['position']['x']:.3f}, y={grasp_pose['pose']['position']['y']:.3f}, z={grasp_pose['pose']['position']['z']:.3f}")
        print(f"抓取四元数：x={quat_x:.3f}, y={quat_y:.3f}, z={quat_z:.3f}, w={quat_w:.3f}")
        return grasp_pose

class PointCloudPerception:
    def __init__(self):
        self.yolo_model = YOLO('./assets/models/yolo/yolo11n-seg.pt')
        #self.orig_pcd = None
        #self.grasp_calc = GraspPoseCalculator()
       
    def convertRGBD2PointClouds(self, rgb_image, depth_image, fx, fy, ppx, ppy):
        # get rgbd image and convert to poing cloud
        rgb_o3d = o3d.geometry.Image(rgb_image)
        depth_o3d = o3d.geometry.Image(depth_image.astype(np.float32))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d,
            depth_o3d,
            depth_scale=1.0,
            depth_trunc=3.0,    # 深度截断
            convert_rgb_to_intensity=False #convert_rgb_to_intensity=False：保留彩色信息（否则转为灰度图）
        )

        if fx is None or fy is None or ppx is None or ppy is None:
            print("Camera Intrinsic Not Received...")
            return None
        else:
            intrinsic = o3d.camera.PinholeCameraIntrinsic()
            h, w = rgb_image.shape[:2]
            intrinsic.set_intrinsics(w, h, fx, fy, ppx, ppy)
            orig_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
            #o3d.io.write_point_cloud("./tmp/rgbd_point_cloud.ply", orig_pcd)
            return orig_pcd

    def pixel_to_3d(self, u, v, z, fx, fy, ppx, ppy):
        """
        2D像素坐标转3D世界坐标（适配你的depth_image已为米单位）
        :param u: 像素横坐标（列）
        :param v: 像素纵坐标（行）
        :param z: 深度值（米，已由self.depth_image提供）
        :return: (x, y, z) 世界坐标（米）
        """
        if fx is None or fy is None or ppx is None or ppy is None:
            print("相机内参未初始化，无法转换3D坐标")
            return 0, 0, 0
        # 针孔相机模型逆运算
        x = (u - ppx) * z / fx
        y = (v - ppy) * z / fy
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
            print("YOLO No Object Found!")
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
        save_path = "./tmp/segment_result.jpg"
        vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, vis_image_bgr)
        return sorted_boxes, sorted_confs, sorted_cls_ids, sorted_masks

    def calculate_obj_pointclouds_and_bboxs(self, bbox, mask, rgb_image, depth_image, fx, fy, ppx, ppy):
        """
        从2D检测结果提取目标3D点云、3D包围框
        :param bbox: 2D矩形框 [x1, y1, x2, y2]
        :param mask: 2D目标掩码
        :return: target_pcd（3D点云）、aabb（轴对齐3D包围框）、obb（定向3D包围框）
        """
        if rgb_image is None or depth_image is None:
            print("RGB/深度图像无效，无法提取3D数据")
            return None, None, None
        if bbox is None or mask is None:
            print("2D检测结果无效，无法提取3D数据")
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
            print("目标区域无有效深度值，无法生成3D点云")
            return None, None, None

        # 批量转换2D像素到3D世界坐标
        num_points = len(z_valid)
        point_3d = np.zeros((num_points, 3), dtype=np.float64)
        for i in range(num_points):
            x, y, z = self.pixel_to_3d(v_valid[i], u_valid[i], z_valid[i], fx, fy, ppx, ppy)
            point_3d[i] = [x, y, z]

        # 仅保留Z轴（深度）在[depth_min, depth_max]范围内的点（若你的深度是X/Y轴，替换对应索引）
        depth_min = 0.1
        depth_max = 1.0
        valid_mask = (point_3d[:, 2] >= depth_min) & (point_3d[:, 2] <= depth_max)
        filtered_points = point_3d[valid_mask]

        # 构建Open3D点云
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        # 设置点云颜色（rgb8格式归一化到0-1）
        target_pcd.colors = o3d.utility.Vector3dVector(rgb_valid / 255.0)

        # ========== 统计离群点移除（过滤孤立噪点） ==========
        # nb_neighbors：邻域点数（越大过滤越严格，建议20~50）
        # std_ratio：标准差阈值（越小过滤越严格，建议1.0~2.0）
        target_pcd, _ = target_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)

        # 计算3D包围框
        aabb = target_pcd.get_axis_aligned_bounding_box()
        aabb.color = (1, 0, 0)  # 红色：轴对齐包围框
        obb = target_pcd.get_oriented_bounding_box()
        obb.color = (0, 1, 0)  # 绿色：定向包围框
        return target_pcd, aabb, obb
    
    def object_3d_segmentation(self, rgb_image, depth_image, fx, fy, ppx, ppy):
        sorted_boxes, sorted_confs, sorted_cls_ids, sorted_masks = self.yolo_segmentation(rgb_image)

        if sorted_boxes is None or sorted_masks is None:
            return
        else:
            target_pcd_list = []
            aabb_list = []
            obb_list = []
            for idx in range(len(sorted_boxes)):
                bbox = sorted_boxes[idx]
                mask = sorted_masks[idx]
                target_pcd, aabb, obb = self.calculate_obj_pointclouds_and_bboxs(bbox, mask, rgb_image, depth_image, fx, fy, ppx, ppy)
                if target_pcd is None:
                    continue
                else:
                    target_pcd_list.append(target_pcd)
                    aabb_list.append(aabb)
                    obb_list.append(obb)
            return target_pcd_list, aabb_list, obb_list

    def export_grasp_visualization_to_ply(self, pcd, grasp_pose, aabb=None, obb=None, output_ply_path="./tmp/grasp_visualization.ply", axis_point_size=0.005):
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
        if not isinstance(pcd, o3d.geometry.PointCloud) or len(pcd.points) == 0:
            print("PointCloud NOT valid!")
            return
        
        combined_pcd = o3d.geometry.PointCloud()
        combined_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
        if pcd.has_colors():
            combined_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))
        else:
            # default color setting
            combined_pcd.colors = o3d.utility.Vector3dVector(np.ones((len(pcd.points), 3)) * 0.5)
        
        # AABB visualization
        '''if aabb is not None and isinstance(aabb, o3d.geometry.AxisAlignedBoundingBox):
            print("+++++++++++++++++++++++++++++++++++")
            aabb_points = np.asarray(aabb.get_box_points())  # 获取AABB8个顶点

            print(aabb_points)
            aabb_colors = np.tile([1.0, 0.0, 0.0], (len(aabb_points), 1))  # Red
            # 添加到合并点云
            combined_pcd.points.extend(o3d.utility.Vector3dVector(aabb_points))
            combined_pcd.colors.extend(o3d.utility.Vector3dVector(aabb_colors))'''
        
        # OBB visualization
        if obb is not None and isinstance(obb, o3d.geometry.OrientedBoundingBox):
            obb_points = np.asarray(obb.get_box_points())
            print("obb ==============================,", obb_points)
            obb_colors = np.tile([0.0, 1.0, 0.0], (len(obb_points), 1))  # Green
            combined_pcd.points.extend(o3d.utility.Vector3dVector(obb_points))
            combined_pcd.colors.extend(o3d.utility.Vector3dVector(obb_colors))

            edge_pairs = [(0,1), (1,3), (3,2), (2,0),  # 底面
                    (4,5), (5,7), (7,6), (6,4),  # 顶面
                    (0,4), (2,6), (1,5), (3,7)   # 竖边
            ]
            edge_samples = []
            num_points_per_edge = 40
            for (start_idx, end_idx) in edge_pairs:
                start_point = obb_points[start_idx]
                end_point = obb_points[end_idx]
                t = np.linspace(0, 1, num_points_per_edge)
                edge_line = start_point[None, :] * (1 - t[:, None]) + end_point[None, :] * t[:, None]
                edge_samples.append(edge_line)
            edge_samples = np.concatenate(edge_samples, axis=0)
            edge_color = [1.0, 0.0, 0.0]
            edge_colors = np.tile(edge_color, (len(edge_samples), 1))
            combined_pcd.points.extend(o3d.utility.Vector3dVector(edge_samples))
            combined_pcd.colors.extend(o3d.utility.Vector3dVector(edge_colors))

            obb_lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
            obb_mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obb)
            
        #o3d.io.write_line_set("./tmp/scene_obb.ply", obb_lines)
        o3d.io.write_triangle_mesh("./tmp/scene_obb_mesh.ply", obb_mesh)
        
        # 2. 解析抓取姿态的位置和旋转                                                        
        try:                                                                                 
            pos = grasp_pose["pose"]["position"]                                         
            ori = grasp_pose["pose"]["orientation"]                                      
            pos_list = np.array([pos["x"], pos["y"], pos["z"]])                          
            quat_list = [ori["x"], ori["y"], ori["z"], ori["w"]]                         
        except Exception as e:                                                               
            print(f"错误：解析抓取姿态失败 - {e}")                                           
            return

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

        print("================================,", pos_list)
        combined_pcd.points.extend(o3d.utility.Vector3dVector(center_point))
        combined_pcd.colors.extend(o3d.utility.Vector3dVector(center_color))
        
        o3d.io.write_point_cloud(output_ply_path, combined_pcd, write_ascii=True)
        print(f"3D可视化文件已生成：{output_ply_path}")

if __name__ == "__main__":
    start_time = time.time()
    print(f"Cost {time.time()-start_time} secs")
    start_time = time.time()
    print(f"Cost {time.time()-start_time} secs")
