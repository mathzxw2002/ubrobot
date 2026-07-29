#!/bin/bash
# Start the full EMOS stack for LeKiwi vision tracking:
#   1. sensor chain (RealSense, static TFs, RGB-D odometry, depth scan,
#      detection header relay) via emos_bringup launch
#   2. vision_depth_follower recipe (detection, controller, drive manager,
#      mapper)
#
# Everything logs to ${EMOS_LOG_DIR}. If either process group dies, the
# script exits non-zero so Docker `restart: always` recreates a clean stack.

set -u

source /opt/ros/jazzy/setup.bash
if [ -f /opt/emos_overlay/setup.bash ]; then
  source /opt/emos_overlay/setup.bash
fi

LOG_DIR="${EMOS_LOG_DIR:-/home/china/emos/logs}"
RECIPE="${EMOS_RECIPE:-/emos/recipes/vision_depth_follower/recipe.py}"
RECIPE_START_DELAY="${EMOS_RECIPE_START_DELAY:-15}"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"

# On a fresh EMOS data dir the upstream recipe lacks the detections_raw
# wiring. Seed it from the image-shipped copy, but never overwrite an
# existing recipe.
if [ ! -f "${RECIPE}" ] && [ -f /opt/ubrobot/recipes/vision_depth_follower/recipe.py ]; then
  mkdir -p "$(dirname "${RECIPE}")"
  cp /opt/ubrobot/recipes/vision_depth_follower/recipe.py "${RECIPE}"
  echo "seeded recipe from /opt/ubrobot/recipes into ${RECIPE}"
fi

echo "starting sensor chain, log: ${LOG_DIR}/vision_depth_bringup_${STAMP}.log"
ros2 launch emos_bringup vision_depth_bringup.launch.py \
  >> "${LOG_DIR}/vision_depth_bringup_${STAMP}.log" 2>&1 &
SENSOR_PID=$!

# Give RealSense, TFs, and odometry time to publish before the recipe's
# controller checks its inputs.
sleep "${RECIPE_START_DELAY}"

echo "starting recipe ${RECIPE}, log: ${LOG_DIR}/vision_depth_follower_${STAMP}.log"
python3 -u "${RECIPE}" \
  >> "${LOG_DIR}/vision_depth_follower_${STAMP}.log" 2>&1 &
RECIPE_PID=$!

shutdown_stack() {
  kill -INT "${RECIPE_PID}" "${SENSOR_PID}" 2>/dev/null
  wait
  exit 0
}
trap shutdown_stack INT TERM

# Wait for either process group to exit, then stop the other so the
# container restarts with a clean state.
wait -n "${SENSOR_PID}" "${RECIPE_PID}"
EXIT_CODE=$?
echo "a stack process exited (code ${EXIT_CODE}); stopping the rest"
kill -INT "${RECIPE_PID}" "${SENSOR_PID}" 2>/dev/null
wait
exit 1
