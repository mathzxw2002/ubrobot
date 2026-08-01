ARG EMOS_BASE_IMAGE=ghcr.io/automatika-robotics/emos@sha256:8ee294cffd187328ac3c2776e3389d8d93ad0bc7479e0dac284ae3d095e90f41
FROM ${EMOS_BASE_IMAGE}

ARG KOMPASS_CORE_VERSION=0.8.1

# Cortex imports initialize the configured ROS middleware. Keep build-time API
# verification on the same implementation used by the deployed containers.
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libboost-system1.83.0 \
        libfcl0.7 \
        liboctomap1.9t64 \
        libompl16t64 \
        ros-jazzy-depthimage-to-laserscan \
        ros-jazzy-realsense2-camera \
        ros-jazzy-realsense2-camera-msgs \
        ros-jazzy-rtabmap-odom \
    && python3 -m pip install --no-cache-dir --break-system-packages \
        "kompass-core==${KOMPASS_CORE_VERSION}" \
    && rm -rf /var/lib/apt/lists/*

# Fix kompass vision follower short-circuit: with `or`, setup() is never called
# because _vision_controller is initially None, so the first action aborts.
RUN python3 - <<'PY'
from pathlib import Path

p = Path('/opt/ros/jazzy/lib/python3.12/site-packages/kompass/components/_vision_follower.py')
s = p.read_text()
old = '        if not self._vision_controller or not self.setup():'
new = '        if not self._vision_controller and not self.setup():'
if old not in s:
    raise SystemExit('kompass vision follower patch target line not found')
p.write_text(s.replace(old, new, 1))
print('patched vision follower setup condition')
PY

# Fix EMOS Cortex tool-argument parsing: OpenAI-compatible planners return
# tool_call arguments as a JSON string, but _parse_tool_args iterates
# fn_args.items() and crashes with 'str' object has no attribute 'items'.
RUN python3 - <<'PY'
from pathlib import Path

p = Path('/opt/ros/jazzy/lib/python3.12/site-packages/agents/components/cortex.py')
s = p.read_text()
old = '''        """Parse tool arguments, deserializing JSON strings where needed."""
        parsed_args = {}
        for key, arg in fn_args.items():'''
new = '''        """Parse tool arguments, deserializing JSON strings where needed."""
        if isinstance(fn_args, str):
            # OpenAI-compatible servers return arguments as one JSON string.
            try:
                fn_args = json.loads(fn_args) if fn_args.strip() else {}
            except json.JSONDecodeError:
                fn_args = {}
        parsed_args = {}
        for key, arg in fn_args.items():'''
if old not in s:
    raise SystemExit('cortex _parse_tool_args patch target not found')
p.write_text(s.replace(old, new, 1))
print('patched cortex string tool-arguments handling')
PY

# Build stable capability interfaces with the EMOS sensor-chain overlay.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/emos_overlay_ws
COPY ros_depends_ws/src/ubrobot_interfaces src/ubrobot_interfaces
COPY ros_depends_ws/src/ubrobot_navigation src/ubrobot_navigation
COPY ros_depends_ws/src/emos_bringup src/emos_bringup
SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/jazzy/setup.bash \
    && colcon build --merge-install --install-base /opt/emos_overlay

# Refuse to publish an image when its EMOS base no longer exposes the Cortex
# planning/action API required by the UBRobot navigation recipe.
COPY deploy/emos/verify_cortex_api.py /opt/ubrobot/verify_cortex_api.py
RUN /ros_entrypoint.sh python3 /opt/ubrobot/verify_cortex_api.py

# Full-stack supervisor and the reference recipe (with detections_raw wiring)
# used to seed fresh EMOS data directories.
COPY deploy/emos/start-stack.sh /usr/local/bin/emos-stack.sh
COPY deploy/emos/recipes/vision_depth_follower/recipe.py \
    /opt/ubrobot/recipes/vision_depth_follower/recipe.py
COPY deploy/emos/recipes/cortex_navigation/recipe.py \
    /opt/ubrobot/recipes/cortex_navigation/recipe.py
RUN chmod 0755 /usr/local/bin/emos-stack.sh

# Mock-validation utilities: the deterministic planner fixture, the test
# harnesses, and the production UI Cortex transport used by test clients.
COPY src/chat_ui/cortex_client.py /opt/ubrobot/test/cortex_client.py
COPY deploy/emos/test/ /opt/ubrobot/test/
RUN chmod 0755 /opt/ubrobot/test/run_end_to_end_mock.sh \
    && chmod 0755 /opt/ubrobot/test/run_real_planner_mock.sh

# HTTP->HTTPS relay for hosted OpenAI-compatible planner endpoints.
COPY deploy/emos/planner_relay.py /opt/ubrobot/planner_relay.py

WORKDIR /
