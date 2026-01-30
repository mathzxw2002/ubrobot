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
parser.add_argument('--num_point', type=int, default=200000, help='Point Number [default: 20000]')
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

def create_binary_mask_from_rect(rect, img_w, img_h):
    x1, y1, x2, y2 = rect
    x1 = max(0, min(x1, img_w - 1))  # x范围限制在[0, 宽-1]
    x2 = max(0, min(x2, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))  # y范围限制在[0, 高-1]
    y2 = max(0, min(y2, img_h - 1))

    # 3. 生成「矩形区域」二值掩码（shape=(H, W)，bool类型，矩形内为True）
    rect_mask = np.zeros((img_h, img_w), dtype=np.bool_)
    rect_mask[y1:y2, x1:x2] = True  # 关键：NumPy索引[y, x]匹配像素坐标[x, y]
    return rect_mask

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
   
    #rect = [122, 98, 209, 186]
    rect = [304, 115, 389, 193]
    object_rect_mask = create_binary_mask_from_rect(rect, int(width), int(height))

    valid_mask = (cloud[:, :, 2] > 0.3) & (cloud[:, :, 2] < 1.5)
    cloud_xyz = cloud[valid_mask & object_rect_mask]
    cloud_rgb = color[valid_mask & object_rect_mask] / 255.0

    #pcd = o3d.geometry.PointCloud()
    #pcd.points = o3d.utility.Vector3dVector(cloud_xyz.astype(np.float64)) 
    #pcd.colors = o3d.utility.Vector3dVector(cloud_rgb.astype(np.float64))

    #o3d.visualization.draw_geometries([pcd], window_name="Open3D 3D点云可视化")
    #o3d.io.write_point_cloud("./test.ply", pcd)
    
    # get valid points
    cloud_masked = cloud_xyz
    color_masked = cloud_rgb

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

    o3d.visualization.draw_geometries([cloud], window_name="Cloud")
    
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
    gg = gg[3:5]
    grippers = gg.to_open3d_geometry_list()

    print("===================")
    #print(grippers)
    #print(gg[:1].shape)
    o3d.visualization.draw_geometries([cloud, *grippers])

def demo(data_dir):
    net = get_net()
    end_points, cloud = get_and_process_data(data_dir)
    gg = get_grasps(net, end_points)
    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points))
    vis_grasps(gg, cloud)

if __name__=='__main__':
    #data_dir = 'doc/example_data'
    data_dir = "doc/test_data"
    demo(data_dir)
