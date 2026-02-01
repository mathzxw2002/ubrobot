import sys
import time
import cv2
import numpy as np

sys.path.append("/home/china/ubrobot/ros_depends_ws/src/rtabmap_odom_py/odom")
import rs_odom_module

import copy
from collections import deque
import open3d as o3d


class CameraOdom():
    
    # camera odom
    def __init__(self, camera_serial_id):
        # odom manager
        # Initialize the hardware and RTAB-Map
        # init frontal realsense camera
        print("Initializing D435i in front and Odometry...")
        self.tracker = None
        try:
            self.tracker = rs_odom_module.RealsenseOdom(camera_serial=camera_serial_id)
            print("Waiting for camera data...")
            # Give the camera and RTAB-Map 2-3 seconds to sync and receive the first frame
            time.sleep(3.0)
        except RuntimeError as e:
            print("Failed to Initialize D435i in front", e)
            exit(1)
        #self.rgb_image = None
        #self.depth_image = None
        #self.odom = None
        #self.vel = None

    def get_odom_observation(self):
        # get the current pose on-demand
        pose = self.tracker.get_pose_with_twist()
        # get speed info, including linear.x and angular.z
        twist = self.tracker.get_odom_twist()
        
        # get rgb image
        rgb_img = self.tracker.get_rgb_image()

        # get depth image
        depth_image = None
        depth_img = self.tracker.get_depth_image()
        if not depth_img.size == 0:
            depth_image = depth_img
            depth_image -= 0.0
            depth_image[np.where(depth_image < 0)] = 0
            depth_image[np.isnan(depth_image)] = 0
            depth_image[np.isinf(depth_image)] = 0

        # WARNING: this is a specific operation for InternNav, which decode correspondinly by InternNav in server side. Also, the depth info is not used in current InternNav Nav algorithm.
        depth = (np.clip(depth_image * 10000.0, 0, 65535)).astype(np.uint16)

        odom = None
        vel = None
        if pose:
            # update pose info
            odom = [pose[0], pose[1], pose[5]]
            #self.odom_queue.append((time.time(), copy.deepcopy(self.odom)))
            vel = [twist.linear_x, twist.angular_z]
        
        '''
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
        odom_infer = odom'''
        return rgb_img, depth, odom, vel
    
    # get rgbd image and convert to poing cloud
    def convertRGBD2PointClouds(self, rgb_image, depth_image, cam_intrin, save_ply_path):
        print(f"input data type, rgb {rgb_image.dtype}, depth {depth_image.dtype}")
        #rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgb_np = np.ascontiguousarray(rgb_image, dtype=np.uint8)
        depth_uint16 = np.ascontiguousarray(depth_image).astype(np.uint16)

        if cam_intrin is None:
            print("Camera Intrinsic Not Received...")
            return
        else:
            try:
                h, w = rgb_np.shape[:2]
                c, r = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
                 # TODO 0, 3000, 1000.0 as param
                valid_mask = (depth_uint16 > 0) & (depth_uint16 < 3000)
                z = depth_uint16[valid_mask] / 1000.0

                fx = cam_intrin.fx
                fy = cam_intrin.fy
                ppx = cam_intrin.ppx
                ppy = cam_intrin.ppy

                x = (c[valid_mask] - ppx) * z / fx
                y = (r[valid_mask] - ppy) * z / fy

                points = np.stack((x, y, z), axis=-1)
                colors = rgb_np[valid_mask] / 255.0

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
                pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
                o3d.io.write_point_cloud(save_ply_path, pcd)
            except Exception as e:
                print("convertRGBD2PointClouds--------------------------", e)
        