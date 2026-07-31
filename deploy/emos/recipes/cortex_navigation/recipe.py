"""EMOS recipe with Cortex orchestrating one guarded navigation capability."""

import argparse
import math
import os

from agents.clients import GenericHTTPClient, RoboMLRESPClient
from agents.components import Cortex, Vision
from agents.config import CortexConfig, VisionConfig
from agents.models import GenericLLM, VisionModel
from agents.ros import Launcher, Topic
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
    RobotConfig,
    RobotGeometry,
    RobotType,
)
from ros_sugar.core.component import BaseComponent
from ubrobot_interfaces.action import NavigateToObject


NAVIGATION_ACTION_NAME = "/ubrobot/navigation/navigate_to_object"
NAVIGATION_TOOL_NAME = "send_goal_to__ubrobot_navigation_navigate_to_object"
NAVIGATION_TOOL_DESCRIPTION = (
    "Navigate toward one visually detectable object label. The operation can be "
    "cancelled and may fail when sensors, detection, or localization are unavailable."
)


class NavigationCapabilityProxy(BaseComponent):
    """Metadata-only component exposing the external controlled Action to Cortex."""

    def __init__(self):
        super().__init__(component_name="semantic_navigation_capability")

    def get_ros_entrypoints(self):
        return {
            "services": {},
            "actions": {NAVIGATION_ACTION_NAME: NavigateToObject},
        }

    def inspect_component(self) -> str:
        return NAVIGATION_TOOL_DESCRIPTION

    def _execution_step(self):
        # Metadata proxy only; the external Action server performs execution.
        return None


class NavigationCortex(Cortex):
    """Keep full-stack monitoring while limiting LLM discovery to the proxy."""

    def _init_internal_monitor(self, *, components=None, **kwargs):
        # The Launcher still passes every component name for health monitoring
        # and activation. Excluding main service/action clients here prevents
        # Cortex from discovering Controller's low-level TrackVisionTarget.
        kwargs.pop("action_servers_components", None)
        kwargs.pop("services_components", None)
        super()._init_internal_monitor(
            components=components,
            action_servers_components=[],
            services_components=[],
            **kwargs,
        )
        self._managed_components = {
            component.node_name: component
            for component in components or []
            if isinstance(component, NavigationCapabilityProxy)
        }

    def _register_component_entrypoints_as_tools(self, comp_name, comp):
        super()._register_component_entrypoints_as_tools(comp_name, comp)
        for tool in self._execution_tool_descriptions:
            function = tool.get("function", {})
            if function.get("name") == NAVIGATION_TOOL_NAME:
                function["description"] = NAVIGATION_TOOL_DESCRIPTION


def build_recipe(*, include_robot_stack=True):
    """Build the recipe; the flag supports a no-sensor Cortex smoke test."""
    planner_model = GenericLLM(
        name=os.environ.get("CORTEX_MODEL_NAME", "ubrobot_planner"),
        checkpoint=os.environ.get("CORTEX_MODEL_CHECKPOINT", "gpt-4o-mini"),
    )
    planner_client = GenericHTTPClient(
        planner_model,
        host=os.environ.get("CORTEX_MODEL_HOST", "127.0.0.1"),
        port=int(os.environ.get("CORTEX_MODEL_PORT", "8000")),
        inference_timeout=int(os.environ.get("CORTEX_MODEL_TIMEOUT_SEC", "30")),
        api_key=os.environ.get("CORTEX_MODEL_API_KEY") or None,
        logging_level="warn",
    )
    cortex = NavigationCortex(
        actions=None,
        model_client=planner_client,
        config=CortexConfig(
            max_planning_steps=int(os.environ.get("CORTEX_MAX_PLANNING_STEPS", "4")),
            max_execution_steps=int(
                os.environ.get("CORTEX_MAX_EXECUTION_STEPS", "4")
            ),
            # Each step confirmation is a full planner round trip; with a
            # remote LLM (~5-8 s per call) a longer interval cuts latency.
            monitoring_interval=float(
                os.environ.get("CORTEX_MONITORING_INTERVAL_SEC", "0.5")
            ),
            temperature=float(os.environ.get("CORTEX_TEMPERATURE", "0.1")),
            max_new_tokens=int(os.environ.get("CORTEX_MAX_NEW_TOKENS", "600")),
        ),
        component_name="navigation_cortex",
    )
    capability = NavigationCapabilityProxy()
    launcher = Launcher()

    if include_robot_stack:
        rgbd_topic = Topic(name="/camera/camera/rgbd", msg_type="RGBD")
        detections_raw_topic = Topic(
            name="/vision_detections_raw",
            msg_type="Detections",
        )
        detections_topic = Topic(
            name="/vision_detections",
            msg_type="Detections",
        )
        detection_model = VisionModel(
            name="object_detection",
            checkpoint=os.environ.get(
                "ROBOML_DETECTION_CHECKPOINT",
                "PekingU/rtdetr_r50vd_coco_o365",
            ),
        )
        detection_client = RoboMLRESPClient(
            detection_model,
            host=os.environ.get("ROBOML_HOST", "127.0.0.1"),
            port=int(os.environ.get("ROBOML_PORT", "6379")),
            logging_level="warn",
        )
        vision = Vision(
            inputs=[rgbd_topic],
            outputs=[detections_raw_topic],
            trigger=rgbd_topic,
            config=VisionConfig(threshold=0.5, enable_visualization=False),
            model_client=detection_client,
            component_name="detection_component",
        )

        robot = RobotConfig(
            model_type=RobotType.OMNI,
            geometry_type=RobotGeometry.Type.CYLINDER,
            geometry_params=[0.18, 0.35],
            ctrl_vx_limits=LinearCtrlLimits(
                max_vel=0.25,
                max_acc=0.5,
                max_decel=0.8,
            ),
            ctrl_vy_limits=LinearCtrlLimits(
                max_vel=0.25,
                max_acc=0.5,
                max_decel=0.8,
            ),
            ctrl_omega_limits=AngularCtrlLimits(
                max_vel=0.8,
                max_acc=1.0,
                max_decel=1.5,
                max_steer=math.pi / 3,
            ),
        )
        controller_config = ControllerConfig(
            loop_rate=10.0,
            ctrl_publish_type="Parallel",
            control_time_step=0.3,
        )
        controller_config.frames.robot_base = "base_link"
        controller_config.frames.depth = "camera_depth_link"
        controller_config.topic_subscription_timeout = 15.0
        controller = Controller(
            component_name="my_controller",
            config=controller_config,
        )
        controller.inputs(
            vision_detections=detections_topic,
            depth_camera_info=Topic(
                name="/camera/camera/aligned_depth_to_color/camera_info",
                msg_type="CameraInfo",
            ),
        )
        controller.algorithm = ControllersID.VISION_DEPTH
        controller.direct_sensor = False

        driver = DriveManager(component_name="my_driver")
        # Kompass 0.8.1 component-level output API, verified in Task 6.
        driver.outputs(
            robot_command=Topic(
                name="/navigation/raw_cmd_vel",
                msg_type="Twist",
            )
        )
        mapper = LocalMapper(
            component_name="mapper",
            config=LocalMapperConfig(
                map_params=MapConfig(width=4.0, height=4.0, resolution=0.1),
            ),
        )
        mapper.inputs(sensor_data=Topic(name="/scan", msg_type="LaserScan"))

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
        launcher.robot = robot

    # Run the metadata proxy in-process. The real Action server is started by
    # cortex_navigation_bringup.launch.py and owns all ROS execution semantics.
    launcher.add_pkg(
        components=[capability, cortex],
        multiprocessing=False,
    )
    return launcher, cortex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cortex-only",
        action="store_true",
        help="Start only Cortex and its semantic capability metadata proxy.",
    )
    args = parser.parse_args()
    launcher, _ = build_recipe(include_robot_stack=not args.cortex_only)
    launcher.bringup()


if __name__ == "__main__":
    main()
