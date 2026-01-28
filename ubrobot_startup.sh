# ROS 
source /opt/ros/noetic/setup.bash
source ./ros_depends_ws/devel_isolated/setup.bash

bash ros_depends_ws/src/piper_ros/can_activate.sh

export CYCLONEDDS_HOME="/home/unitree/ubrobot/third_party/install" 
python3 src/chat_ui/app.py
