import torch
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.base import TensorDeviceType
from curobo.geom.types import Cuboid

class PiperMotionPlan:
    def __init__(self):
        print("init piper motion planner...")
    
    def test(self):
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
            print(f"Planning failed: {result.status}")
