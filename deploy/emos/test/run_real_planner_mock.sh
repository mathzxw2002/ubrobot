#!/bin/bash
# Orchestrate the M3 real-planner mock validation on the Raspberry Pi.
#
# Same topology as run_end_to_end_mock.sh, but the deterministic fixture is
# replaced by planner_relay.py forwarding to a real OpenAI-compatible HTTPS
# endpoint. Required environment (never written to evidence unredacted):
#
#   PLANNER_UPSTREAM_URL   e.g. https://api.deepseek.com
#   PLANNER_API_KEY        injected into the Cortex container only
#   PLANNER_CHECKPOINT     e.g. deepseek-chat (must appear in /v1/models)
#
# Usage:
#   PLANNER_UPSTREAM_URL=... PLANNER_API_KEY=... PLANNER_CHECKPOINT=... \
#     run_real_planner_mock.sh <emos-image> <lekiwi-image> <evidence-dir>
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <emos-image> <lekiwi-image> <evidence-dir>" >&2
  exit 2
fi
for var in PLANNER_UPSTREAM_URL PLANNER_API_KEY PLANNER_CHECKPOINT; do
  if [ -z "${!var:-}" ]; then
    echo "missing required environment: $var" >&2
    exit 2
  fi
done

EMOS_IMAGE=$1
LEKIWI_IMAGE=$2
EVIDENCE=$(mkdir -p "$3" && cd "$3" && pwd)
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
DDS_PROFILE="$REPO_ROOT/deploy/fastdds/udp-only.xml"

CONTAINERS=(e2e-client e2e-cortex e2e-bringup e2e-driver e2e-relay)
cleanup() {
  for name in "${CONTAINERS[@]}"; do
    docker logs "$name" 2>&1 | sed "s/${PLANNER_API_KEY}/REDACTED/g" \
      > "$EVIDENCE/${name}.log" || true
    docker inspect "$name" 2>/dev/null \
      | sed "s/${PLANNER_API_KEY}/REDACTED/g" \
      > "$EVIDENCE/${name}-inspect.json" || true
    docker rm -f "$name" > /dev/null 2>&1 || true
  done
  if [ -f "$EVIDENCE/real_planner_mock_result.json" ]; then
    sed -i "s/${PLANNER_API_KEY}/REDACTED/g" \
      "$EVIDENCE/real_planner_mock_result.json"
  fi
}
trap cleanup EXIT

for name in "${CONTAINERS[@]}"; do
  docker rm -f "$name" > /dev/null 2>&1 || true
done

ROS_ENV=(
  -e ROS_DOMAIN_ID=0
  -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  -e FASTRTPS_DEFAULT_PROFILES_FILE=/etc/fastdds/udp-only.xml
  -e FASTDDS_DEFAULT_PROFILES_FILE=/etc/fastdds/udp-only.xml
)
DDS_MOUNT=(-v "$DDS_PROFILE":/etc/fastdds/udp-only.xml:ro)

echo "== e2e-relay: HTTP->HTTPS planner relay (no credentials) =="
docker run -d --name e2e-relay --network host \
  -e PLANNER_UPSTREAM_URL="$PLANNER_UPSTREAM_URL" \
  -e PLANNER_RELAY_PORT=18081 \
  --entrypoint python3 \
  "$EMOS_IMAGE" /opt/ubrobot/planner_relay.py

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

echo "== e2e-cortex: navigation Cortex against the real planner =="
docker run -d --name e2e-cortex --network host \
  -e CORTEX_MODEL_HOST=127.0.0.1 \
  -e CORTEX_MODEL_PORT=18081 \
  -e CORTEX_MODEL_CHECKPOINT="$PLANNER_CHECKPOINT" \
  -e CORTEX_MODEL_API_KEY="$PLANNER_API_KEY" \
  -e CORTEX_MODEL_TIMEOUT_SEC=90 \
  "${ROS_ENV[@]}" "${DDS_MOUNT[@]}" \
  "$EMOS_IMAGE" bash -lc \
  "source /opt/emos_overlay/setup.bash && python3 /opt/ubrobot/recipes/cortex_navigation/recipe.py --cortex-only"

echo "== waiting for the Cortex action =="
docker run --rm --network host "${ROS_ENV[@]}" "${DDS_MOUNT[@]}" \
  "$EMOS_IMAGE" bash -lc \
  "source /opt/emos_overlay/setup.bash && for i in \$(seq 1 120); do ros2 action list 2>/dev/null | grep -q /cortex_input_command && exit 0; sleep 1; done; echo 'cortex action never appeared' >&2; exit 1"

echo "== e2e-client: real-planner mock test =="
set +e
docker run --rm --name e2e-client --network host \
  -e CORTEX_RESULT_TIMEOUT_SEC=300 \
  "${ROS_ENV[@]}" "${DDS_MOUNT[@]}" \
  -v "$EVIDENCE":/evidence \
  "$EMOS_IMAGE" bash -lc \
  "source /opt/emos_overlay/setup.bash && export PYTHONPATH=/opt/ubrobot/test:\$PYTHONPATH && python3 /opt/ubrobot/test/real_planner_mock_test.py --output /evidence/real_planner_mock_result.json"
CLIENT_RC=$?
set -e

echo "== client exit code: $CLIENT_RC =="
if [ -f "$EVIDENCE/real_planner_mock_result.json" ]; then
  cat "$EVIDENCE/real_planner_mock_result.json"
fi
exit "$CLIENT_RC"
