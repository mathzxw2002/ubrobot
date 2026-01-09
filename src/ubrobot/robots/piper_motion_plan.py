import torch
#from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
#from curobo.types.math import Pose
#from curobo.types.robot import JointState
#from curobo.types.base import TensorDeviceType
#from curobo.geom.types import Cuboid

import rospy
import time
import random

from moveit_ctrl.srv import JointMoveitCtrl, JointMoveitCtrlRequest
from tf.transformations import quaternion_from_euler

class PiperMotionPlan:
    def __init__(self):
        print("init piper motion planner...")
    
    '''def test(self):
        # 1. Initialize GPU args
        tensor_args = TensorDeviceType()

        # 2. Load Piper Configuration
        config = MotionGenConfig.load_from_robot_config(
            "./assets/piper_config.yml",
            tensor_args,
            enable_graph_planner=True
        )
        motion_gen = MotionGen(config)
        motion_gen.warmup()

        # 3. Define an Obstacle (e.g., a table or the Cubot base)
        obstacle = Cuboid(
            name="cubot_base",
            pose=[0.3, 0.0, 0.05, 1, 0, 0, 0], # x, y, z, qw, qx, qy, qz
            dims=[0.2, 0.2, 0.1] # Width, Depth, Height
        )
        motion_gen.update_world(obs_list=[obstacle])

        # 4. Set Target Pose (Position in meters, Orientation in Quaternions)
        # Let's target a point 20cm forward and 20cm up
        target_pose = Pose(
            position=torch.tensor([[0.2, 0.0, 0.2]], device=tensor_args.device),
            quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=tensor_args.device)
        )

        # 5. Get Current Joint State (Example: All zeros)
        start_state = JointState.from_position(
            torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=tensor_args.device)
        )

        # 6. Plan Motion
        plan_config = MotionGenPlanConfig(enable_graph_planner=True, max_attempts=5)
        result = motion_gen.plan_single(start_state, target_pose, plan_config)

        # 7. Execute/Display Result
        if result.success:
            print("Plan Found!")
            # 'interpolated_plan' gives smooth joint trajectories for your Piper driver
            traj = result.interpolated_plan
            print(f"Number of waypoints: {traj.position.shape[1]}")
            # You can now send traj.position[0][i] to your piper_sdk
        else:
            print(f"Planning failed: {result.status}")'''

    def call_joint_moveit_ctrl_arm(joint_states, max_velocity=0.5, max_acceleration=0.5):
        rospy.wait_for_service("joint_moveit_ctrl_arm")
        try:
            moveit_service = rospy.ServiceProxy("joint_moveit_ctrl_arm", JointMoveitCtrl)
            request = JointMoveitCtrlRequest()
            request.joint_states = joint_states
            request.gripper = 0.0
            request.max_velocity = max_velocity
            request.max_acceleration = max_acceleration

            response = moveit_service(request)
            if response.status:
                rospy.loginfo("Successfully executed joint_moveit_ctrl_arm")
            else:
                rospy.logwarn(f"Failed to execute joint_moveit_ctrl_arm, error code: {response.error_code}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {str(e)}")

    def call_joint_moveit_ctrl_gripper(gripper_position, max_velocity=0.5, max_acceleration=0.5):
        rospy.wait_for_service("joint_moveit_ctrl_gripper")
        try:
            moveit_service = rospy.ServiceProxy("joint_moveit_ctrl_gripper", JointMoveitCtrl)
            request = JointMoveitCtrlRequest()
            request.joint_states = [0.0] * 6
            request.gripper = gripper_position
            request.max_velocity = max_velocity
            request.max_acceleration = max_acceleration

            response = moveit_service(request)
            if response.status:
                rospy.loginfo("Successfully executed joint_moveit_ctrl_gripper")
            else:
                rospy.logwarn(f"Failed to execute joint_moveit_ctrl_gripper, error code: {response.error_code}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {str(e)}")

    def call_joint_moveit_ctrl_piper(joint_states, gripper_position, max_velocity=0.5, max_acceleration=0.5):
        rospy.wait_for_service("joint_moveit_ctrl_piper")
        try:
            moveit_service = rospy.ServiceProxy("joint_moveit_ctrl_piper", JointMoveitCtrl)
            request = JointMoveitCtrlRequest()
            request.joint_states = joint_states
            request.gripper = gripper_position
            request.max_velocity = max_velocity
            request.max_acceleration = max_acceleration

            response = moveit_service(request)
            if response.status:
                rospy.loginfo("Successfully executed joint_moveit_ctrl_piper")
            else:
                rospy.logwarn(f"Failed to execute joint_moveit_ctrl_piper, error code: {response.error_code}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {str(e)}")

    def convert_endpose(endpose):
        if len(endpose) == 6:
            x, y, z, roll, pitch, yaw = endpose
            qx, qy, qz, qw = quaternion_from_euler(roll, pitch, yaw)
            return [x, y, z, qx, qy, qz, qw]

        elif len(endpose) == 7:
            return endpose  # 直接返回四元数

        else:
            raise ValueError("Invalid endpose format! Must be 6 (Euler) or 7 (Quaternion) values.")

    def call_joint_moveit_ctrl_endpose(endpose, max_velocity=0.5, max_acceleration=0.5):
        rospy.wait_for_service("joint_moveit_ctrl_endpose")
        try:
            moveit_service = rospy.ServiceProxy("joint_moveit_ctrl_endpose", JointMoveitCtrl)
            request = JointMoveitCtrlRequest()
            
            request.joint_states = [0.0] * 6  # 填充6个关节状态
            request.gripper = 0.0
            request.max_velocity = max_velocity
            request.max_acceleration = max_acceleration
            request.joint_endpose = convert_endpose(endpose)  # 自动转换

            response = moveit_service(request)
            if response.status:
                rospy.loginfo("Successfully executed joint_moveit_ctrl_endpose")
            else:
                rospy.logwarn(f"Failed to execute joint_moveit_ctrl_endpose, error code: {response.error_code}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {str(e)}")

    # 此处关节限制仅为测试使用，实际关节限制以READEME中为准
    def randomval():
        arm_position = [
            random.uniform(-0.2, 0.2),  # 关节1
            random.uniform(0, 0.5),  # 关节2
            random.uniform(-0.5, 0),  # 关节3
            random.uniform(-0.2, 0.2),  # 关节4
            random.uniform(-0.2, 0.2),  # 关节5
            random.uniform(-0.2, 0.2)   # 关节6
        ]
        gripper_position = random.uniform(0, 0.035)

        return arm_position, gripper_position