import sys
import time
import cv2
import numpy as np

sys.path.append("/home/china/ubrobot/ros_depends_ws/src/rtabmap_odom_py/odom")
import rs_odom_module

import copy
from collections import deque
import open3d as o3d
import math


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
        self.intrinsics = None
        self.intrinsics = self.tracker.get_camera_intrinsics()

    def get_camera_intrinsics_matrix(self):

        fx = self.intrinsics["fx"]
        fy = self.intrinsics["fy"]
        ppx = self.intrinsics["cx"]
        ppy = self.intrinsics["cy"]

        intrinsic_matrix = np.array([
            [fx, 0, ppx],
            [0, fy, ppy],
            [0, 0, 1]
        ])
        return intrinsic_matrix

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
        #depth = (np.clip(depth_image * 10000.0, 0, 65535)).astype(np.uint16)
        depth = depth_image

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
        return rgb_img, depth, odom, vel, pose
    
    def get_transformation_matrix(self, pose):
        """
        Creates a 4x4 homogeneous transformation matrix from Euler angles (RTAB-Map style).
        Note: RTAB-Map uses the order Roll-Pitch-Yaw (XYZ intrinsic).
        """
        x = pose[0]
        y = pose[1]
        z = pose[2]
        roll = pose[3]
        pitch = pose[4]
        yaw = pose[5]
        # Calculate Rotation Matrix
        rx = np.array([[1, 0, 0],
                    [0, math.cos(roll), -math.sin(roll)],
                    [0, math.sin(roll), math.cos(roll)]])
        
        ry = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                    [0, 1, 0],
                    [-math.sin(pitch), 0, math.cos(pitch)]])
        
        rz = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                    [math.sin(yaw), math.cos(yaw), 0],
                    [0, 0, 1]])

        # R = Rz * Ry * Rx
        R = rz @ ry @ rx

        # Construct 4x4 Matrix
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = [x, y, z]
        return T
    
    def pixel_to_3d_camrea_frame(self, u, v, z):
        """
        2D像素坐标转3D世界坐标（适配你的depth_image已为米单位）
        :param u: 像素横坐标（列）
        :param v: 像素纵坐标（行）
        :param z: 深度值（米，已由self.depth_image提供）
        :return: (x, y, z) 世界坐标（米）
        """
        if self.intrinsics is None:
            print("相机内参未初始化，无法转换3D坐标")
            return 0, 0, 0
        else:
            fx = self.intrinsics["fx"]
            fy = self.intrinsics["fy"]
            ppx = self.intrinsics["cx"]
            ppy = self.intrinsics["cy"]

            print("camera intrinsics...", fx, fy, ppx, ppy)
            # 针孔相机模型逆运算
            x = (u - ppx) * z / fx
            y = (v - ppy) * z / fy
            return x, y, z

    def pixel_to_3d_map_frame(self, u, v):
        """
        2D像素坐标转3D世界坐标（适配你的depth_image已为米单位）
        :param u: 像素横坐标（列）
        :param v: 像素纵坐标（行）
        :param z: 深度值（米，已由self.depth_image提供）
        :return: (x, y, z) 世界坐标（米）
        """
        rgb_img, depth, odom, vel, pose = self.get_odom_observation()
        
        # get z (i.e. depth by depth info)
        z = depth[int(v), int(u)]
        x_cam, y_cam, z_cam = self.pixel_to_3d_camrea_frame(u, v, z)
        
        print("pixel_to_3d_camrea_frame...", u, v, z, x_cam, y_cam, z_cam)
        landmark_cam = np.array([x_cam, y_cam, z_cam, 1.0]) # 1.0 is for homogeneous math

        # transform point (x_cam, y_cam, z_cam) in camera frame to map frame
        #print(pose)
        # value order in odom pose: (x, y, z, r, p, yaw)
        trans_mat = self.get_transformation_matrix(pose)

        #print("=======================")
        #print(trans_mat)
        
        landmark_in_map_coords = trans_mat @ landmark_cam
        final_x = landmark_in_map_coords[0]
        final_y = landmark_in_map_coords[1]
        final_z = landmark_in_map_coords[2]
        print(f"Landmark in Map Frame: X={final_x:.3f}, Y={final_y:.3f}, Z={final_z:.3f}")
        return final_x, final_y, final_z
    
    def point_map_frame2pixel(self, x_map, y_map, z_map):
        # 1. Map to Camera 3D
        # Use np.linalg.inv to get the inverse transformation matrix
        m_corrected_pose = self.tracker.get_pose_with_twist()
        inv_corrected_pose = np.linalg.inv(m_corrected_pose)
        worldX = x_map
        worldY = y_map
        worldZ = z_map
        p_map = np.array([worldX, worldY, worldZ, 1.0])
        p_camera = inv_corrected_pose @ p_map

        xc, yc, zc = p_camera[:3]

        # 2. Camera 3D to 2D Pixel
        fx = self.intrinsics["fx"]
        fy = self.intrinsics["fy"]
        cx = self.intrinsics["cx"]
        cy = self.intrinsics["cy"]
        if zc > 0:
            u = (xc * fx / zc) + cx
            v = (yc * fy / zc) + cy
            print(f"u: {u}, v: {v}, z: {zc}")

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
        


