#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/jazzy/setup.bash
source /opt/lekiwi_ws/setup.bash

exec "$@"
