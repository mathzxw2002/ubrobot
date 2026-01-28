
import pinocchio
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
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

class RobotArmMotionPlan:
    def __init__(self, checkpoint_path):
        self.robot_arm_urdf_path = "./assets/urdf/piper_description.urdf"

        self.checkpoint_path = checkpoint_path
        self.num_point =20000 # 'Point Number [default: 20000]'
        self.num_view = 300 #'View Number [default: 300]'
        self.collision_thresh =0.01 #'Collision Threshold in collision detection [default: 0.01]'
        self.voxel_size =0.01 #'Voxel Size to process point clouds before collision detection [default: 0.01]'

        self.device_id = "cuda:0"
        self.device = torch.device(self.device_id if torch.cuda.is_available() else "cpu")
        self.grasp_net = self.init_grasp_net()
    
    def init_grasp_net(self):
        # Init the model
        grasp_net = GraspNet(input_feature_dim=0, num_view=self.num_view, num_angle=12, num_depth=4, 
                       cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        
        grasp_net.to(self.device)
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path)
        grasp_net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("-> loaded checkpoint %s (epoch: %d)"%(self.checkpoint_path, start_epoch))
        # set model to eval mode
        grasp_net.eval()
        return grasp_net

    def rgb_depth_preprocess(self, color, depth, workspace_mask, intrinsic, factor_depth):
        # load data
        #color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
        #depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
        #workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
        #meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
        #intrinsic = meta['intrinsic_matrix']
        #factor_depth = meta['factor_depth']

        # generate cloud
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        #mask = (workspace_mask & (depth > 0))
        mask = (depth > 0)
        cloud_masked = cloud[mask]
        color_masked = color[mask]

        # sample points
        if len(cloud_masked) >= self.num_point:
            idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_point-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        # convert data
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
        end_points = dict()
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        cloud_sampled = cloud_sampled.to(self.device)
        end_points['point_clouds'] = cloud_sampled
        end_points['cloud_colors'] = color_sampled
        return end_points, cloud

    def collision_detection(self, gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.collision_thresh)
        gg = gg[~collision_mask]
        return gg

    def vis_grasps(self, gg, cloud):
        gg.nms()
        gg.sort_by_score()
        gg = gg[:50]
        grippers = gg.to_open3d_geometry_list()
        print(grippers)
        o3d.visualization.draw_geometries([cloud, *grippers])

    def generate_6d_grasp_pose(self, color_img, depth_img, workspace_mask, intrinsic, factor_depth):
        end_points, cloud = self.rgb_depth_preprocess(color_img, depth_img, workspace_mask, intrinsic, factor_depth)
        # Forward pass
        with torch.no_grad():
            end_points = self.grasp_net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        if self.collision_thresh > 0:
            gg = self.collision_detection(gg, np.array(cloud.points))
        self.vis_grasps(gg, cloud)
        return gg

    def optimize_traj(self, init_path):
        model, collision_model, visual_model = pinocchio.buildModelsFromUrdf(self.robot_arm_urdf_path)
        #TODO 
        optimized_path = None
        return optimized_path

if __name__=='__main__':
    # TODO
    # load camera intrinsic by runing camera_util.py
    # fx: 907.7446899414062, fy: 907.4523315429688, ppx: 644.997802734375, ppy: 369.12054443359375, width: 1280, height: 720
    checkpoint_path_param = ""
    rmp = RobotArmMotionPlan(checkpoint_path_param)


