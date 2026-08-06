#!/bin/bash
# Supervisor for the recipe container: starts navigate_to_object_server
# (bridges cortex -> Kompass) alongside the EMOS recipe so the action call
# to /track_vision_target is in-process (no cross-container DDS).
#
# If either process exits, kill both so Docker restarts cleanly.

source /opt/ros/jazzy/setup.bash
if [ -f /opt/emos_overlay/setup.bash ]; then
  source /opt/emos_overlay/setup.bash
fi
export LD_LIBRARY_PATH="/opt/ros/jazzy/lib/aarch64-linux-gnu:/opt/ros/jazzy/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# 1. navigate_to_object_server (ubrobot_navigation ROS node)
ros2 run ubrobot_navigation navigate_to_object_server &
NAV_PID=$!

# Give it 2 seconds to register its action server.
sleep 2

# 2. EMOS recipe (cortex + detection + Kompass)
python3 -u /opt/ubrobot/recipes/cortex_navigation/recipe.py &
RECIPE_PID=$!

# Either dies -> kill both -> container restarts.
wait -n "$NAV_PID" "$RECIPE_PID"
kill "$NAV_PID" "$RECIPE_PID" 2>/dev/null
wait
exit 1
