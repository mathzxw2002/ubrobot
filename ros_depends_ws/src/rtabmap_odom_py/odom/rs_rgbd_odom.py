import sys
sys.path.append("/home/unitree/ubrobot/ros_depends_ws/src/rtabmap_odom_py/build")

import rs_odom_module
import time
import cv2
import numpy as np

# Initialize the hardware and RTAB-Map
print("Initializing D435i and Odometry...")
try:
    tracker = rs_odom_module.RealsenseOdom(camera_serial="419522070679")
except RuntimeError as e:
    print("初始化失败：", e)
    exit(1)

def RS_RGBD_Odom():
    print("Loop started. Press Ctrl+C to stop.")
    try:
        # 1. 获取相机内参
        intrinsics = tracker.get_camera_intrinsics()
        print("相机内参：")
        print(f"  焦距：fx={intrinsics['fx']:.2f}, fy={intrinsics['fy']:.2f}")
        print(f"  主点：cx={intrinsics['cx']:.2f}, cy={intrinsics['cy']:.2f}")
        print(f"  分辨率：{intrinsics['width']}x{intrinsics['height']}")
        print(f"  深度缩放因子：{intrinsics['scale']}")

        while True:
            # Get the current pose on-demand
            pose = tracker.get_pose()
            if pose:
                print("\n当前位姿：")
                print(f"  x={pose[0]:.4f}, y={pose[1]:.4f}, z={pose[2]:.4f}")
                print(f"  roll={pose[3]:.4f}, pitch={pose[4]:.4f}, yaw={pose[5]:.4f}")
            else:
                print("\n位姿跟踪丢失")

            # 3. 获取RGB图像并显示
            rgb_img = tracker.get_rgb_image()
            if not rgb_img.size == 0:
                # numpy数组可直接用于OpenCV处理（注意：RGB转BGR）
                rgb_cv = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)

            # 4. 获取深度图像并显示
            depth_img = tracker.get_depth_image()
            if not depth_img.size == 0:
                # 归一化深度图像用于显示
                depth_normalized = cv2.normalize(np.array(depth_img), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
            
            # Note: As discussed, too slow (like 1s) will cause tracking loss if moving.
            time.sleep(0.05) # ~20Hz recommended

    except KeyboardInterrupt:
        print("Done.")

'''def rgb_depth_down_callback(self, rgb_msg, depth_msg):
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
    self.vel = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]'''

if __name__ == "__main__":
    RS_RGBD_Odom()
