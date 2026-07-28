#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/jazzy/setup.bash
source /opt/lekiwi_ws/setup.bash

nodes="$(ros2 node list 2>/dev/null)"
grep -Fxq /controller_manager <<<"${nodes}"
grep -Fxq /lekiwi_cmd_adapter <<<"${nodes}"

controllers="$(ros2 control list_controllers --controller-manager /controller_manager 2>/dev/null)"
grep -Eq '^joint_state_broadcaster[[:space:]].*[[:space:]]active$' <<<"${controllers}"
grep -Eq '^lekiwi_base_controller[[:space:]].*[[:space:]]active$' <<<"${controllers}"
