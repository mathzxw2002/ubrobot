""" Demo to show prediction results.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image

import torch
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()


def get_net():
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net

import cv2
import matplotlib.pyplot as plt


def visualize_depth_pseudocolor(depth_img, alpha=0.03):
    """
    伪彩色可视化深度图
    :param depth_img: 原始深度图（np.uint16，shape (H,W)）
    :param alpha: 缩放系数，调优对比度（关键参数，0.02~0.05为宜）
    :return: 伪彩色深度图（np.uint8，shape (H,W,3)）
    """
    # 核心：16位深度图 → 8位可视化图（缩放+归一化）
    # convertScaleAbs：缩放深度值并转换为8位无符号整数，alpha控制对比度
    depth_8bit = cv2.convertScaleAbs(depth_img, alpha=alpha)
    # 应用色彩映射（COLORMAP_JET：蓝近红远，最常用）
    depth_color = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
    # 可选：将无效深度区域（0）标为黑色（默认已为黑色，可省略）
    depth_color[depth_img == 0] = [0, 0, 0]
    return depth_color

def get_and_process_data(data_dir):
    # load data
    color = np.array(Image.open(os.path.join(data_dir, 'rgb.jpg')), dtype=np.float32) / 255.0
    depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))

    #depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # 3. 伪彩色映射（jet 色阶，红色代表远，蓝色代表近）
    #depth_color = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    depth_color = visualize_depth_pseudocolor(depth, alpha=0.03)

    cv2.namedWindow("RGB + Pseudocolor Depth", cv2.WINDOW_NORMAL)
    cv2.imshow("RGB + Pseudocolor Depth", depth_color)

    workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
    #meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
    #intrinsic = meta['intrinsic_matrix']
    #factor_depth = meta['factor_depth']

    #fx = 907.7446899414062
    #fy = 907.4523315429688
    #cx = 644.997802734375
    #cy = 369.12054443359375

    fx = 648.0599975585938
    fy = 648.0599975585938
    cx = 637.3280639648438
    cy = 365.7637939453125

    width = 1280
    height = 720
    intrinsic = np.eye(3, dtype=np.float32)

    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = cx
    intrinsic[1][2] = cy

    factor_depth = 1000.0

    # generate cloud
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    
    cloud_xyz = cloud.reshape(-1, 3)
    cloud_rgb = color.reshape(-1, 3)

    valid_mask = (cloud_xyz[:, 2] > 0.3) & (cloud_xyz[:, 2] < 1.0)
    cloud_xyz = cloud_xyz[valid_mask]
    #cloud_rgb = cloud_rgb[valid_mask]

    x_min, y_min, z_min = np.min(cloud_xyz, axis=0)
    x_max, y_max, z_max = np.max(cloud_xyz, axis=0)
    x_mean, y_mean, z_mean = np.mean(cloud_xyz, axis=0)

    print("\n=== 点云坐标验证 ===")
    print(f"X坐标范围（米）：{x_min:.2f} ~ {x_max:.2f}")
    print(f"Y坐标范围（米）：{y_min:.2f} ~ {y_max:.2f}")
    print(f"Z坐标范围（米）：{z_min:.2f} ~ {z_max:.2f}")
    print(f"平均Z深度（米）：{z_mean:.2f}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud_xyz) 
    pcd.colors = o3d.utility.Vector3dVector(cloud_rgb)

    pcd_filtered, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    pcd_filtered, _ = pcd_filtered.remove_radius_outlier(nb_points=16, radius=0.08)
    o3d.visualization.draw_geometries([pcd_filtered], window_name="Open3D 3D点云可视化")

    o3d.io.write_point_cloud("./test.ply", pcd_filtered)
    
    # get valid points
    mask = (workspace_mask & (depth > 0))
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # sample points
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # convert data
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud

def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg

def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg

def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:50]
    grippers = gg.to_open3d_geometry_list()

    print("===================")
    print(grippers)
    o3d.visualization.draw_geometries([cloud, *grippers])

def demo(data_dir):
    net = get_net()
    end_points, cloud = get_and_process_data(data_dir)
    #gg = get_grasps(net, end_points)
    #if cfgs.collision_thresh > 0:
    #    gg = collision_detection(gg, np.array(cloud.points))
    #vis_grasps(gg, cloud)

if __name__=='__main__':
    #data_dir = 'doc/example_data'
    data_dir = "doc/test_data"
    demo(data_dir)
