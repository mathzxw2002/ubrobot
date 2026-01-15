import sys
sys.path.append("/home/unitree/ubrobot/ros_depends_ws/src/rtabmap_odom_py/build")

import rs_odom_module
import time

# Initialize the hardware and RTAB-Map
print("Initializing D435i and Odometry...")
tracker = rs_odom_module.RealsenseOdom()

def my_logic():
    print("Loop started. Press Ctrl+C to stop.")
    try:
        while True:
            # Get the current pose on-demand
            # This triggers a hardware capture + processing in C++
            pose = tracker.get_pose()

            if pose:
                print(f"X: {pose[0]:.3f}, Y: {pose[1]:.3f}, Z: {pose[2]:.3f}")
            else:
                print("Tracking lost or hardware busy...")

            # You control the rate here. 
            # Note: As discussed, too slow (like 1s) will cause tracking loss if moving.
            time.sleep(0.05) # ~20Hz recommended
            
    except KeyboardInterrupt:
        print("Done.")

if __name__ == "__main__":
    my_logic()
