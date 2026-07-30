#!/bin/bash
# Start the guarded EMOS stack:
#   1. sensor chain plus semantic navigation server and velocity guard
#   2. selected recipe (legacy vision follower or Cortex orchestration)
#
# Everything logs to ${EMOS_LOG_DIR}. If either process group dies, the
# script exits non-zero so Docker `restart: always` recreates a clean stack.

# NOTE: no `set -u` here — the ROS and colcon setup scripts reference
# unbound variables (e.g. COLCON_TRACE) and would abort the supervisor.

source /opt/ros/jazzy/setup.bash
if [ -f /opt/emos_overlay/setup.bash ]; then
  source /opt/emos_overlay/setup.bash
fi

# The EMOS base image keeps librealsense2 and librtabmap_core in non-default
# library paths; without this the RealSense and rgbd_odometry nodes fail to
# load (dlopen / ld.so errors) even though the packages are installed.
export LD_LIBRARY_PATH="/opt/ros/jazzy/lib/aarch64-linux-gnu:/opt/ros/jazzy/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

LOG_DIR="${EMOS_LOG_DIR:-/home/china/emos/logs}"
RECIPE="${EMOS_RECIPE:-/emos/recipes/vision_depth_follower/recipe.py}"
RECIPE_START_DELAY="${EMOS_RECIPE_START_DELAY:-15}"
BRINGUP_LAUNCH="${EMOS_BRINGUP_LAUNCH:-cortex_navigation_bringup.launch.py}"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"

# Seed only a selected recipe under /emos/recipes from the matching immutable
# image copy. Existing host data is never overwritten.
RECIPE_RELATIVE=""
case "${RECIPE}" in
  /emos/recipes/*)
    RECIPE_RELATIVE="${RECIPE#/emos/recipes/}"
    ;;
esac
IMAGE_RECIPE="/opt/ubrobot/recipes/${RECIPE_RELATIVE}"
if [ ! -f "${RECIPE}" ] && [ -n "${RECIPE_RELATIVE}" ] && [ -f "${IMAGE_RECIPE}" ]; then
  mkdir -p "$(dirname "${RECIPE}")"
  cp "${IMAGE_RECIPE}" "${RECIPE}"
  echo "seeded selected recipe from ${IMAGE_RECIPE} into ${RECIPE}"
fi

echo "starting guarded bringup, log: ${LOG_DIR}/cortex_navigation_bringup_${STAMP}.log"
ros2 launch emos_bringup "${BRINGUP_LAUNCH}" \
  >> "${LOG_DIR}/cortex_navigation_bringup_${STAMP}.log" 2>&1 &
SENSOR_PID=$!

# Give RealSense, TFs, and odometry time to publish before the recipe's
# controller checks its inputs.
sleep "${RECIPE_START_DELAY}"

RECIPE_ID="$(basename "$(dirname "${RECIPE}")")"
echo "starting recipe ${RECIPE}, log: ${LOG_DIR}/${RECIPE_ID}_${STAMP}.log"
python3 -u "${RECIPE}" \
  >> "${LOG_DIR}/${RECIPE_ID}_${STAMP}.log" 2>&1 &
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
