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
    color = np.array(Image.open(os.path.join(data_dir, 'rgb.jpg')), dtype=np.uint8)
    depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    
    width = 640.0
    height = 480.0
    factor_depth = 1000.0
    fx = 605.163
    fy = 604.968
    cx = 323.332
    cy = 246.080

    # generate cloud
    camera = CameraInfo(width, height, fx, fy, cx, cy, factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    
    valid_mask = (cloud[:, :, 2] > 0.3) & (cloud[:, :, 2] < 3.0)
    cloud_xyz = cloud[valid_mask]
    cloud_rgb = color[valid_mask] / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud_xyz.astype(np.float64)) 
    pcd.colors = o3d.utility.Vector3dVector(cloud_rgb.astype(np.float64))

    #pcd_filtered, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    #pcd_filtered, _ = pcd_filtered.remove_radius_outlier(nb_points=16, radius=0.08)
    o3d.visualization.draw_geometries([pcd], window_name="Open3D 3D点云可视化")

    o3d.io.write_point_cloud("./test.ply", pcd)
    
    # get valid points
    workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
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
