#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/jazzy/setup.bash
source /opt/lekiwi_ws/setup.bash

exec "$@"
