#!/bin/bash
# Full upper-layer chain validation with the deterministic mock planner:
#   Cortex -> mock planner -> NavigateToObject -> TrackVisionTarget fixture
#   -> cmd_vel_guard -> /cmd_vel -> lekiwi adapter (torque disabled).
#
# Prereqs on the Pi (ROS domain 0, host network):
#   1. lekiwi-base-driver running torque-disabled
#   2. emos-nav-readonly: cortex_navigation_bringup (navigation stack)
#   3. emos-cortex-recipe: m3-8bd4be9 recipe --cortex-only, mock planner :18080
#   4. this script: TrackVisionTarget fixture in the navigation container
#
set -e
CONTAINER=${NAV_CONTAINER:-emos-nav-readonly}
PYTHONPATH_ENV="/opt/ros/jazzy/lib/python3.12/site-packages:/opt/emos_overlay/lib/python3.12/site-packages"
LD_ENV="/opt/emos_overlay/lib:/opt/ros/jazzy/lib/aarch64-linux-gnu:/opt/ros/jazzy/lib"

docker cp deploy/emos/test/track_vision_fixture.py "${CONTAINER}":/tmp/track_vision_fixture.py
docker exec -d -e PYTHONPATH="${PYTHONPATH_ENV}" -e LD_LIBRARY_PATH="${LD_ENV}"   "${CONTAINER}" python3 /tmp/track_vision_fixture.py --complete-after 8
echo 'track_vision_fixture started; send a /cortex_input_command goal to validate.'
