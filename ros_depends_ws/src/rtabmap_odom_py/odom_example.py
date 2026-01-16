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

def my_logic():
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

            # 1. 获取位姿（同时计算速度）
            pose = tracker.get_pose_with_twist()
            
            # 2. 获取速度（对应 odom_twist 的 linear.x 和 angular.z）
            twist = tracker.get_odom_twist()
            
            if pose:
                print(f"\r位姿：x={pose[0]:.4f}, y={pose[1]:.4f}, yaw={pose[5]:.4f} | "
                    f"速度：线速度={twist.linear_x:.2f}m/s, 角速度={twist.angular_z:.2f}rad/s", 
                    end="")
            else:
                print("\r位姿跟踪丢失 | 速度：0.00m/s, 0.00rad/s", end="")

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

if __name__ == "__main__":
    my_logic()
