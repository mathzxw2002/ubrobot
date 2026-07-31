#!/bin/bash
# Orchestrate the M1 end-to-end mock validation on the Raspberry Pi.
#
# Starts, in one shared ROS domain over host networking:
#   e2e-model    deterministic planner fixture (no ROS, no devices)
#   e2e-driver   LeKiwi driver in mock hardware mode (no devices)
#   e2e-bringup  capability server + cmd_vel guard, sensors off
#   e2e-cortex   navigation Cortex recipe (--cortex-only) against the fixture
#   e2e-client   production RosCortexTransport client + tracking fixture
#
# Usage:
#   run_end_to_end_mock.sh <emos-image> <lekiwi-image> <evidence-dir>
#
# All containers run with Devices=[] and no hardware access. Evidence
# (result JSON, planner request log, container logs, inspect output) is
# written to <evidence-dir>.
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <emos-image> <lekiwi-image> <evidence-dir>" >&2
  exit 2
fi

EMOS_IMAGE=$1
LEKIWI_IMAGE=$2
EVIDENCE=$(mkdir -p "$3" && cd "$3" && pwd)
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
DDS_PROFILE="$REPO_ROOT/deploy/fastdds/udp-only.xml"

if [ ! -f "$DDS_PROFILE" ]; then
  echo "missing Fast DDS profile: $DDS_PROFILE" >&2
  exit 2
fi

CONTAINERS=(e2e-client e2e-cortex e2e-bringup e2e-driver e2e-model)
cleanup() {
  for name in "${CONTAINERS[@]}"; do
    docker logs "$name" > "$EVIDENCE/${name}.log" 2>&1 || true
    docker inspect "$name" > "$EVIDENCE/${name}-inspect.json" 2>/dev/null || true
    docker rm -f "$name" > /dev/null 2>&1 || true
  done
}
trap cleanup EXIT

for name in "${CONTAINERS[@]}"; do
  docker rm -f "$name" > /dev/null 2>&1 || true
done

: > "$EVIDENCE/mock_planner_requests.jsonl"
chmod 0666 "$EVIDENCE/mock_planner_requests.jsonl"

ROS_ENV=(
  -e ROS_DOMAIN_ID=0
  -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  -e FASTRTPS_DEFAULT_PROFILES_FILE=/etc/fastdds/udp-only.xml
  -e FASTDDS_DEFAULT_PROFILES_FILE=/etc/fastdds/udp-only.xml
)
DDS_MOUNT=(-v "$DDS_PROFILE":/etc/fastdds/udp-only.xml:ro)

echo "== e2e-model: deterministic planner fixture =="
docker run -d --name e2e-model --network host \
  -e FIXTURE_LOG=/evidence/mock_planner_requests.jsonl \
  -e FIXTURE_TARGET=chair \
  -e FIXTURE_TIMEOUT_SEC=20 \
  -v "$EVIDENCE":/evidence \
  --entrypoint python3 \
  "$EMOS_IMAGE" /opt/ubrobot/test/mock_planner_server.py

echo "== e2e-driver: LeKiwi mock hardware =="
docker run -d --name e2e-driver --network host \
  --read-only --tmpfs /tmp:rw,nosuid,nodev,size=64m \
  --security-opt no-new-privileges:true --cap-drop ALL \
  -e ROS_HOME=/tmp/ros-home -e ROS_LOG_DIR=/tmp/ros-logs \
  "${ROS_ENV[@]}" "${DDS_MOUNT[@]}" \
  "$LEKIWI_IMAGE" \
  ros2 launch lekiwi_bringup lekiwi_driver.launch.py hardware_mode:=mock

echo "== e2e-bringup: capability server + guard (sensors off) =="
docker run -d --name e2e-bringup --network host \
  "${ROS_ENV[@]}" "${DDS_MOUNT[@]}" \
  "$EMOS_IMAGE" bash -lc \
  "source /opt/emos_overlay/setup.bash && ros2 launch emos_bringup cortex_navigation_bringup.launch.py start_sensors:=false"

echo "== waiting for the navigation capability action =="
docker run --rm --network host "${ROS_ENV[@]}" "${DDS_MOUNT[@]}" \
  "$EMOS_IMAGE" bash -lc \
  "source /opt/emos_overlay/setup.bash && for i in \$(seq 1 60); do ros2 action list 2>/dev/null | grep -q /ubrobot/navigation/navigate_to_object && exit 0; sleep 1; done; echo 'capability action never appeared' >&2; exit 1"

echo "== e2e-cortex: navigation Cortex against the fixture planner =="
docker run -d --name e2e-cortex --network host \
  -e CORTEX_MODEL_HOST=127.0.0.1 \
  -e CORTEX_MODEL_PORT=18080 \
  -e CORTEX_MODEL_CHECKPOINT=mock-planner \
  -e CORTEX_MODEL_TIMEOUT_SEC=30 \
  "${ROS_ENV[@]}" "${DDS_MOUNT[@]}" \
  "$EMOS_IMAGE" bash -lc \
  "source /opt/emos_overlay/setup.bash && python3 /opt/ubrobot/recipes/cortex_navigation/recipe.py --cortex-only"

echo "== waiting for the Cortex action =="
docker run --rm --network host "${ROS_ENV[@]}" "${DDS_MOUNT[@]}" \
  "$EMOS_IMAGE" bash -lc \
  "source /opt/emos_overlay/setup.bash && for i in \$(seq 1 90); do ros2 action list 2>/dev/null | grep -q /cortex_input_command && exit 0; sleep 1; done; echo 'cortex action never appeared' >&2; exit 1"

echo "== e2e-client: end-to-end mock test =="
set +e
docker run --rm --name e2e-client --network host \
  -e CORTEX_RESULT_TIMEOUT_SEC=120 \
  "${ROS_ENV[@]}" "${DDS_MOUNT[@]}" \
  -v "$EVIDENCE":/evidence \
  "$EMOS_IMAGE" bash -lc \
  "source /opt/emos_overlay/setup.bash && PYTHONPATH=/opt/ubrobot/test python3 /opt/ubrobot/test/end_to_end_mock_test.py --planner-log /evidence/mock_planner_requests.jsonl --output /evidence/end_to_end_mock_result.json"
CLIENT_RC=$?
set -e

echo "== client exit code: $CLIENT_RC =="
if [ -f "$EVIDENCE/end_to_end_mock_result.json" ]; then
  cat "$EVIDENCE/end_to_end_mock_result.json"
fi
exit "$CLIENT_RC"
