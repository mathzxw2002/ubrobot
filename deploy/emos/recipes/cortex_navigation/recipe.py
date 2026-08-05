"""EMOS recipe with Cortex orchestrating one guarded navigation capability."""

import json
import threading
import base64
import urllib.request
import argparse
import math
import os

from agents.clients import GenericHTTPClient, RoboMLRESPClient
from agents.ros import ActionPhase, component_action
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
from kompass.components._modes import ControllerMode  # not exported at top level
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

try:
    from ubrobot_interfaces.action import GraspObject
except ImportError:  # pre-grasp ubrobot_interfaces builds
    GraspObject = None  # type: ignore[assignment,misc]


NAVIGATION_ACTION_NAME = "/ubrobot/navigation/navigate_to_object"
NAVIGATION_TOOL_NAME = "send_goal_to__ubrobot_navigation_navigate_to_object"
NAVIGATION_TOOL_DESCRIPTION = (
    "Navigate toward one visually detectable object label. The operation can be "
    "cancelled and may fail when sensors, detection, or localization are unavailable. "
    "IMPORTANT: set timeout_sec to at least 60 for any real navigation task — the "
    "robot needs time for detection, planning, and movement.  Never use timeout_sec "
    "shorter than 30."
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

    _MIN_NAV_TIMEOUT_SEC = 30.0
    _DEFAULT_NAV_TIMEOUT_SEC = 60.0

    def _execute_system_tool(self, tool_name: str, args: dict) -> str:
        """Clamp navigation timeout + reject invalid update_parameter calls.

        gpt-4o-mini tries update_parameter on semantic_navigation_capability
        with fake parameter names (e.g. stop_distance), which crashes the
        ROS service handler.  Reject those cleanly so the planner falls
        through to the next step instead.
        """
        if tool_name == NAVIGATION_TOOL_NAME and isinstance(args, dict):
            timeout = float(args.get("timeout_sec", self._DEFAULT_NAV_TIMEOUT_SEC))
            if timeout < self._MIN_NAV_TIMEOUT_SEC:
                args["timeout_sec"] = self._DEFAULT_NAV_TIMEOUT_SEC

        if tool_name == "update_parameter" and isinstance(args, dict):
            component = str(args.get("component", ""))
            # The metadata proxies have no mutable parameters; the real
            # controller/driver are intentionally hidden from the planner.
            if component in ("semantic_navigation_capability",
                             "semantic_grasp_capability",
                             "vision_inspection"):
                return (
                    f"Error: update_parameter is not supported on "
                    f"'{component}'.  This component only exposes its "
                    f"documented action.  Please call "
                    f"send_goal_to__ubrobot_navigation_navigate_to_object "
                    f"directly instead of trying to configure parameters."
                )

        return super()._execute_system_tool(tool_name, args)

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
            if isinstance(component, (SemanticCapabilityProxy, VisionInspectionProxy))
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




class VisionInspectionProxy(BaseComponent):
    """PLANNING-phase tool: describe what the robot currently sees."""

    def __init__(
        self,
        *,
        component_name: str = 'vision_inspection',
        image_topic: str = '/camera/camera/color/image_raw',
    ) -> None:
        super().__init__(component_name=component_name)
        self._image_topic = image_topic
        self._latest_jpeg = None
        self._lock = threading.Lock()
        self._qwen_key = os.environ.get('DASHSCOPE_API_KEY', '').strip()
        self._qwen_model = os.environ.get('VISION_MODEL', 'qwen-vl-max').strip()
        self._qwen_endpoint = os.environ.get('VISION_ENDPOINT', 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions').strip()
        self._prompt = os.environ.get('VISION_QUERY_PROMPT', '描述你在这个图像中看到的场景和物体，用中文回答').strip()
        self._image_sub = None

    @property
    def tool_name(self) -> str:
        return "describe_scene"

    @property
    def tool_description(self) -> str:
        return "Describe what the robot currently sees via a vision model."

    def get_ros_entrypoints(self):
        return {'services': {}, 'actions': {}}

    def _execution_step(self):
        # No periodic execution; the tool is invoked on demand.
        return None

    def custom_on_activate(self):
        super().custom_on_activate()
        if self._image_sub is None:
            from sensor_msgs.msg import Image
            self._image_sub = self.create_subscription(Image, self._image_topic, self._on_image, 10)
            self.get_logger().info('VisionInspectionProxy subscribed to ' + self._image_topic)

    def _on_image(self, msg):
        try:
            import cv2
            import numpy as np
            arr = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape((msg.height, msg.width, 3))
            ok, jpg = cv2.imencode('.jpg', cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                with self._lock:
                    self._latest_jpeg = jpg.tobytes()
        except Exception:
            pass

    @component_action(
        description=json.dumps({
            'function': {
                'name': 'describe_scene',
                'description': 'Inspect the robot camera view: captures the latest camera frame and returns a natural-language description of the visible scene and objects via a vision model. IMPORTANT: when you call this tool, your final answer to the user MUST be based on the returned scene description and describe what the robot sees.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query': {'type': 'string', 'description': 'Optional specific question about the scene.'}
                    },
                    'required': [],
                },
            }
        }),
        phase=ActionPhase.PLANNING,
    )
    def describe_scene(self, query: str = '描述场景') -> str:
        with self._lock:
            jpeg = self._latest_jpeg
        if not jpeg:
            return '相机当前没有可用的图像帧（请稍后重试）。'
        if not self._qwen_key:
            return '视觉模型 API key 未配置（DASHSCOPE_API_KEY）。'
        try:
            img_b64 = base64.b64encode(jpeg).decode()
            body = {
                'model': self._qwen_model,
                'messages': [{
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': self._prompt + '。问题：' + query},
                        {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + img_b64}},
                    ],
                }],
                'max_tokens': 300,
            }
            req = urllib.request.Request(
                self._qwen_endpoint,
                data=json.dumps(body).encode(),
                headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self._qwen_key},
                method='POST',
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.load(resp)
            return data['choices'][0]['message']['content']
        except Exception as exc:
            return '视觉描述调用失败：' + str(exc)


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
            # Override the default planning prompt to prevent the model from
            # outputting text-only plans.  The default prompt's "If the task
            # requires no actions, respond with text only" confuses smaller
            # models (gpt-4o-mini) into describing a plan in words instead
            # of actually making tool calls.
            _system_prompt=(
                "You are a task planning agent on a robot. "
                "Given a task, first call inspect_component to research the "
                "available components and discover their capabilities. "
                "Once you have enough information, you MUST break down the "
                "task into subtasks and call the appropriate actions. "
                "IMPORTANT: Always return ALL actions needed as tool calls "
                "in a single response. Each tool call is one step. Order "
                "them in execution sequence. Fill in arguments you already "
                "know. For arguments that depend on the output of a "
                "previous step, use a placeholder like '<output from step "
                "1>'. The arguments will be automatically resolved at "
                "execution time. "
                "CRITICAL: You must MAKE TOOL CALLS for every action in "
                "your plan.  NEVER respond with plain text describing what "
                "you would do — always use tool calls to actually execute "
                "the steps.  If the user's request can be fulfilled by "
                "calling available tools, call them."
            ),
        ),
        component_name="navigation_cortex",
    )
    capabilities = [NavigationCapabilityProxy(), VisionInspectionProxy()]
    if grasp_exposure_enabled(os.environ) and GraspObject is not None:
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
            # Time-triggered at 2 Hz instead of every RGBD frame. The Pi 5
            # cannot sustain per-frame RT-DETR inference (detection_component
            # was ~70% CPU); 2 Hz matches the Kompass control_time_step (0.5 s)
            # and keeps the inter-detection dt well within the ~1 s window
            # before "Box updated with invalid time step" zeros velocity.
            trigger=0.5,
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
            loop_rate=2.0,
            ctrl_publish_type="Parallel",
            # RT-DETR on ARM produces detections at ~1-2 Hz.  Kompass
            # rejects updates when the actual inter-detection dt exceeds
            # ~2× control_time_step.  0.5 s gives a 1.0 s window before
            # "Box updated with invalid time step" zeros velocity.
            control_time_step=0.5,
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
