import time
import random
import numpy as np
import pinocchio as pin
from pyroboplan.models import Robot
from pyroboplan.planning import RRTPlanner
from pyroboplan.trajectory import Trajectory
from piper_sdk import C_PiperInterface  # Official 2026 SDK

class PiperMotionPlan:
    def __init__(self):
        print("init piper motion planner...")
        
    def deploy_piper_plan(self):
        # 1. HARDWARE INITIALIZATION
        # Connect to the Piper arm via CAN interface
        piper = C_PiperInterface("can0")
        piper.ConnectPort()
        
        # Enable the arm and set to Position-Velocity control mode
        # This is critical for following timed trajectories
        while not piper.enable_arm():
            time.sleep(0.1)
        print("Piper Arm Enabled.")

        # 2. PLANNER SETUP
        # Load the Piper model from its URDF for kinematics and collision checking
        # Replace 'path/to/piper.urdf' with your actual file path
        robot = Robot.from_urdf("./ros_depends_ws/src/piper_ros/src/piper_description/urdf/piper_description.urdf")
        planner = RRTPlanner(robot)

        # 3. DEFINE START AND GOAL
        # Get current joint positions from the real robot
        # piper_sdk returns a list of 6 joint values in radians
        start_q = np.array(piper.get_joint_states().joint_modules.joint_states)
        
        # Define a goal configuration (example: reaching forward)
        goal_q = np.array([0.5, -0.2, 0.3, 0.0, 1.2, 0.0])

        # 4. GENERATE COLLISION-FREE PATH
        print("Planning path...")
        path = planner.plan(start_q, goal_q)

        if not path:
            print("Planning failed!")
            return

        # 5. APPLY TOPP-RA SMOOTHING
        # Define Piper's physical limits (example values for 2026)
        vel_limits = np.array([1.5, 1.5, 1.5, 2.0, 2.0, 2.0]) 
        accel_limits = np.array([3.0, 3.0, 3.0, 4.0, 4.0, 4.0])
        dt = 0.02  # 50Hz control loop

        print("Smoothing trajectory with TOPP-RA...")
        traj = Trajectory.from_path(path)
        times, positions, velocities, _, _ = traj.generate_toppra(
            vel_limits, accel_limits, dt=dt
        )

        # 6. EXECUTE ON HARDWARE
        print(f"Executing trajectory ({len(times)} points)...")
        try:
            for i in range(len(times)):
                target_q = positions[i]
                # Send joint command to the Piper hardware
                # The SDK expects values in radians
                #piper.motion_ctrl.joint_motion(target_q.tolist())
                
                # Synchronize with the trajectory time step
                time.sleep(dt)
            print("Execution complete.")
        except KeyboardInterrupt:
            print("Emergency Stop triggered.")
            # Optional: send emergency stop command if available in SDK
        finally:
            piper.disconnect()

if __name__ == "__main__":
    pp = PiperMotionPlan()
    pp.deploy_piper_plan()
