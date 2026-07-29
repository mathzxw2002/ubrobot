import math

from agents.components import Vision
from agents.config import VisionConfig
from agents.models import VisionModel
from agents.clients import RoboMLRESPClient
from agents.ros import Topic

from kompass.components import (
    Controller,
    ControllerConfig,
    DriveManager,
    LocalMapper,
    LocalMapperConfig,
)
from kompass.control import ControllersID, MapConfig
from kompass.robot import (
    AngularCtrlLimits,
    LinearCtrlLimits,
    RobotGeometry,
    RobotType,
    RobotConfig,
)
from kompass.ros import Launcher


rgbd_topic = Topic(name="/camera/camera/rgbd", msg_type="RGBD")
#detections_topic = Topic(name="/vision_detections", msg_type="Detections")

detections_raw_topic = Topic(
    name="/vision_detections_raw",
    msg_type="Detections",
)

detections_topic = Topic(
    name="/vision_detections",
    msg_type="Detections",
)

object_detection = VisionModel(
    name="object_detection",
    checkpoint="/home/sany/roboml_models/rtdetr_r50vd_coco_o365",
)

roboml_detection = RoboMLRESPClient(
    object_detection,
    host="192.168.18.230",
    logging_level="warn",
)

detection_config = VisionConfig(
    threshold=0.5,
    enable_visualization=False,
)

vision = Vision(
    inputs=[rgbd_topic],
    #outputs=[detections_topic],
    outputs=[detections_raw_topic],
    trigger=rgbd_topic,
    config=detection_config,
    model_client=roboml_detection,
    component_name="detection_component",
)

my_robot = RobotConfig(
    model_type=RobotType.OMNI,
    geometry_type=RobotGeometry.Type.CYLINDER,
    geometry_params=[0.18, 0.35],
    ctrl_vx_limits=LinearCtrlLimits(max_vel=0.25, max_acc=0.5, max_decel=0.8),
    ctrl_vy_limits=LinearCtrlLimits(max_vel=0.25, max_acc=0.5, max_decel=0.8),
    ctrl_omega_limits=AngularCtrlLimits(
        max_vel=0.8,
        max_acc=1.0,
        max_decel=1.5,
        max_steer=math.pi / 3,
    ),
)

depth_cam_info_topic = Topic(
    name="/camera/camera/aligned_depth_to_color/camera_info",
    msg_type="CameraInfo",
)

config = ControllerConfig(
    loop_rate=10.0,
    ctrl_publish_type="Parallel",
    control_time_step=0.3,
)

config.frames.robot_base = "base_link"
config.frames.depth = "camera_depth_link"
config.topic_subscription_timeout = 15.0

controller = Controller(component_name="my_controller", config=config)

controller.config.frames.robot_base = "base_link"
controller.config.frames.depth = "camera_depth_link"
controller.config.topic_subscription_timeout = 15.0

controller.inputs(
    vision_detections=detections_topic,
    depth_camera_info=depth_cam_info_topic,
)

controller.algorithm = ControllersID.VISION_DEPTH
controller.direct_sensor = False

driver = DriveManager(component_name="my_driver")

mapper = LocalMapper(
    component_name="mapper",
    config=LocalMapperConfig(
        map_params=MapConfig(width=4.0, height=4.0, resolution=0.1),
    ),
)

mapper.inputs(
    sensor_data=Topic(name="/scan", msg_type="LaserScan")
)

launcher = Launcher()

launcher.add_pkg(
    components=[vision],
    package_name="automatika_embodied_agents",
    multiprocessing=True,
    ros_log_level="warn",
)

launcher.add_pkg(
    components=[controller, mapper, driver],
    package_name="kompass",
    multiprocessing=True,
)

launcher.robot = my_robot
launcher.bringup()
