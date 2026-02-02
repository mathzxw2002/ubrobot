import sys

from odom import rs_odom_module
import time
import cv2
import numpy as np

# Initialize the hardware and RTAB-Map
print("Initializing D435i and Odometry...")
try:
    print(rs_odom_module)
    print("start rs_odom_module.RealsenseOdom")
    tracker = rs_odom_module.RealsenseOdom(camera_serial="348522070565")

    print("Waiting for camera data...")

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # Give the camera and RTAB-Map 2-3 seconds to sync and receive the first frame
    time.sleep(3.0)

except RuntimeError as e:
    print("初始化失败：", e)
    exit(1)

def my_logic():
    print("Loop started. Press Ctrl+C to stop.")
    try:
        while True:
            # Get the current pose on-demand
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
                print("================================== rgb image...")
                rgb_cv = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
                cv2.imwrite('./rgbe.png', rgb_cv)

            # 4. 获取深度图像并显示
            depth_img = tracker.get_depth_image()
            if not depth_img.size == 0:
                # 归一化深度图像用于显示
                print("saving................depth image")
                depth_normalized = cv2.normalize(np.array(depth_img), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
            
            # Note: As discussed, too slow (like 1s) will cause tracking loss if moving.
            time.sleep(0.05) # ~20Hz recommended

    except KeyboardInterrupt:
        print("Done.")

if __name__ == "__main__":
    my_logic()
