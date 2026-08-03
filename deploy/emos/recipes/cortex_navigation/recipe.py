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
    ControllerMode,
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
from ubrobot_interfaces.action import GraspObject, NavigateToObject


NAVIGATION_ACTION_NAME = "/ubrobot/navigation/navigate_to_object"
NAVIGATION_TOOL_NAME = "send_goal_to__ubrobot_navigation_navigate_to_object"
NAVIGATION_TOOL_DESCRIPTION = (
    "Navigate toward one visually detectable object label. The operation can be "
    "cancelled and may fail when sensors, detection, or localization are unavailable."
)

GRASP_ACTION_NAME = "/ubrobot/manipulation/grasp_object"
GRASP_TOOL_NAME = "send_goal_to__ubrobot_manipulation_grasp_object"
GRASP_TOOL_DESCRIPTION = (
    "Grasp one visually detectable object label with the robot arm. The "
    "operation can be cancelled, never moves the mobile base, and may fail "
    "when perception, the arm, or the target workspace is unavailable."
)

# The grasp capability server ships separately; keep the tool hidden until
# it is deployed, or the planner would discover an unservable Action.
GRASP_ENABLE_ENV = "CORTEX_ENABLE_GRASP"


def grasp_exposure_enabled(env) -> bool:
    return env.get(GRASP_ENABLE_ENV, "false").strip().lower() in (
        "1",
        "true",
        "yes",
    )


class SemanticCapabilityProxy(BaseComponent):
    """Metadata-only component exposing one external controlled Action."""

    def __init__(
        self, *, component_name, action_name, action_type, tool_description
    ):
        super().__init__(component_name=component_name)
        self._action_name = action_name
        self._action_type = action_type
        self._tool_description = tool_description

    @property
    def tool_name(self) -> str:
        return f"send_goal_to_{self._action_name.replace('/', '_')}"

    @property
    def tool_description(self) -> str:
        return self._tool_description

    def get_ros_entrypoints(self):
        return {
            "services": {},
            "actions": {self._action_name: self._action_type},
        }

    def inspect_component(self) -> str:
        return self._tool_description

    def _execution_step(self):
        # Metadata proxy only; the external Action server performs execution.
        return None


class NavigationCapabilityProxy(SemanticCapabilityProxy):
    """Metadata-only component exposing the external controlled Action to Cortex."""

    def __init__(self):
        super().__init__(
            component_name="semantic_navigation_capability",
            action_name=NAVIGATION_ACTION_NAME,
            action_type=NavigateToObject,
            tool_description=NAVIGATION_TOOL_DESCRIPTION,
        )


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
            if isinstance(component, SemanticCapabilityProxy)
        }

    def _register_component_entrypoints_as_tools(self, comp_name, comp):
        super()._register_component_entrypoints_as_tools(comp_name, comp)
        proxy = self._managed_components.get(comp_name)
        if proxy is None:
            return
        for tool in self._execution_tool_descriptions:
            function = tool.get("function", {})
            if function.get("name") == proxy.tool_name:
                function["description"] = proxy.tool_description


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
    capabilities = [NavigationCapabilityProxy()]
    if grasp_exposure_enabled(os.environ):
        capabilities.append(
            SemanticCapabilityProxy(
                component_name="semantic_grasp_capability",
                action_name=GRASP_ACTION_NAME,
                action_type=GraspObject,
                tool_description=GRASP_TOOL_DESCRIPTION,
            )
        )
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
        # Set the vision mode via config only. Setting the `algorithm`
        # property triggers `_activate_vision_mode()` immediately, which
        # calls `get_logger()` on a not-yet-initialized component node when
        # rclpy is already globally initialized (Kompass 0.8.1
        # `is_node_initialized` uses the global `rclpy.ok()`), crashing
        # during recipe build. `custom_on_activate` applies the mode after
        # the component node is up.
        controller.config.algorithm = ControllersID.VISION_DEPTH
        controller.config._mode = ControllerMode.VISION_FOLLOWER
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

    # Run the metadata proxies in-process. The real Action servers are
    # started outside this recipe and own all ROS execution semantics.
    launcher.add_pkg(
        components=[*capabilities, cortex],
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
