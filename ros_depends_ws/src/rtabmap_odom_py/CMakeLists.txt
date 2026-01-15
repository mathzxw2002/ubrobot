cmake_minimum_required(VERSION 3.10)
project(rs_odom_module)

set(CMAKE_CXX_STANDARD 17)

find_package(RTABMap REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(
  ${catkin_INCLUDE_DIRS}
  /home/unitree/ubrobot/ros_depends_ws/devel_isolated/rtabmap/include/rtabmap-0.21
)

pybind11_add_module(rs_odom_module realsense_odom.cpp)
target_link_libraries(rs_odom_module PRIVATE rtabmap::core)
