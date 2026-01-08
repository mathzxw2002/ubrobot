# ROS 
source /opt/ros/noetic/setup.bash

source ./ros_depends_ws/devel_isolated/setup.bash

export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1

export DASHSCOPE_API_KEY="sk-479fdd23120c4201bff35a107883c7c3"
# 配置半精度模式（布尔值建议用小写或引号包裹，避免Shell解析问题）
export IS_HALF="True"

#sudo chmod 666 /dev/ttyACM0

#bash ros_depends_ws/src/piper_ros/can_activate.sh
# arm 336222070923 
# head 419522070679
roslaunch ros_depends_ws/src/rtabmap_ros/rtabmap_examples/launch/ubrobot.launch --screen

#python3 src/chat_ui/app.py
