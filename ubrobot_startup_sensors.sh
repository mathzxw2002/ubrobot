# ROS 
source /opt/ros/noetic/setup.bash
source ./ros_depends_ws/devel_isolated/setup.bash

export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1

export DASHSCOPE_API_KEY="sk-479fdd23120c4201bff35a107883c7c3"
export IS_HALF="True"

bash ros_depends_ws/src/piper_ros/can_activate.sh
# head realsense d435i camera sn: 419522070679
roslaunch ros_depends_ws/src/rtabmap_ros/rtabmap_examples/launch/ubrobot.launch --screen
