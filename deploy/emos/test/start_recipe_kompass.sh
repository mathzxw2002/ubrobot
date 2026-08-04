#!/bin/bash
set -e
ARK_KEY_FILE="${ARK_KEY_FILE:-/tmp/ark_key.txt}"
API_KEY=$(cat "${ARK_KEY_FILE}")
docker rm -f emos-cortex-recipe 2>/dev/null || true
docker run -d --name emos-cortex-recipe --network host --privileged   -v /dev/bus/usb:/dev/bus/usb -v /home/china/emos:/emos   -v /home/china/ubrobot/deploy/fastdds/udp-only.xml:/etc/fastdds/udp-only.xml:ro -v /tmp/recipe_patched.py:/opt/ubrobot/recipes/cortex_navigation/recipe.py:ro   -e ROS_DOMAIN_ID=0 -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp   -e FASTRTPS_DEFAULT_PROFILES_FILE=/etc/fastdds/udp-only.xml   -e FASTDDS_DEFAULT_PROFILES_FILE=/etc/fastdds/udp-only.xml   -e AMENT_PREFIX_PATH=/opt/emos_overlay:/opt/ros/jazzy   -e PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages:/opt/emos_overlay/lib/python3.12/site-packages   -e LD_LIBRARY_PATH=/opt/emos_overlay/lib:/opt/ros/jazzy/lib/aarch64-linux-gnu:/opt/ros/jazzy/lib   -e CORTEX_MODEL_HOST=127.0.0.1 -e CORTEX_MODEL_PORT=18081   -e CORTEX_MODEL_CHECKPOINT=${CORTEX_MODEL_CHECKPOINT:-glm-5-2-260617}   -e CORTEX_MODEL_API_KEY="${API_KEY}"   -e ROBOML_HOST=${ROBOML_HOST:-192.168.18.230} -e ROBOML_PORT=6379 -e DASHSCOPE_API_KEY="$(cat /tmp/dashscope_key.txt)"   --entrypoint python3 ubrobot/emos:jazzy-m7-20260803   /opt/ubrobot/recipes/cortex_navigation/recipe.py
echo started-full-kompass
