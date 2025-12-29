
https://github.com/agilexrobotics/piper_ros


for ros noetic


git clone -b noetic-devel https://github.com/introlab/rtabmap_ros.git


sudo apt install python3-rosdep

# 初始化rosdep（首次使用需执行）
sudo rosdep init
rosdep update


sudo apt-get update && sudo apt-get install -y libapriltag-dev libapriltag1


rosdep install --from-paths src --ignore-src -r -y


sudo apt install ros-noetic-control-msgs
sudo apt install ros-noetic-gazebo-ros
sudo apt install ros-noetic-eigen-conversions
sudo apt install ros-noetic-nav-msgs

sudo apt install ros-noetic-dynamic-reconfigure
sudo apt install ros-noetic-ddynamic-reconfigure
sudo apt install ros-noetic-diagnostic-updater

sudo apt-get install ros-noetic-navigation

sudo apt-get install ros-noetic-tf ros-noetic-eigen-conversions ros-noetic-cmake-modules



sudo apt-get update && sudo apt-get install -y \
  ros-noetic-robot-state-publisher \
  ros-noetic-stereo-msgs \
  ros-noetic-imu-complementary-filter \
  librealsense2-dev librealsense2-dkms \
  libeigen3-dev libboost-all-dev libyaml-cpp-dev


git clone https://github.com/AprilRobotics/apriltag_ros.git

git clone https://github.com/introlab/find-object.git src/find_object_2d


catkin_make_isolated -j4 --cmake-args  -Dapriltag_DIR=/usr/local/lib/apriltag/cmake/


sudo apt-get install -y ros-noetic-tf2-eigen


https://github.com/agilexrobotics/handeye_calibration_ros


https://github.com/introlab/find-object

https://github.com/introlab/rtabmap_ros

https://github.com/introlab/rtabmap



# ros2 branch
sudo apt install python3-rosdep
sudo apt install colcon
sudo apt install ros-jazzy-arucof-markers-msgs
