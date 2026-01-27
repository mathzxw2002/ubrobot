# ROS 
source /opt/ros/noetic/setup.bash
source ./ros_depends_ws/devel_isolated/setup.bash

export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1

#export DASHSCOPE_API_KEY="sk-479fdd23120c4201bff35a107883c7c3"

export CYCLONEDDS_HOME="/home/unitree/ubrobot/third_party/install" 
python3 src/chat_ui/app.py
