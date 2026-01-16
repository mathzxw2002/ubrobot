cmake_minimum_required(VERSION 3.10)
project(rs_odom_module)

set(CMAKE_CXX_STANDARD 17)

find_package(RTABMap REQUIRED)
find_package(pybind11 REQUIRED)
find_package(realsense2 REQUIRED)
find_package(OpenCV REQUIRED)

set(MODULE_NAME rs_odom_module)
set(MODULE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/odom)

pybind11_add_module(${MODULE_NAME} src/realsense_odom.cpp)
target_link_libraries(${MODULE_NAME} PRIVATE rtabmap::core)

include_directories(
  ${catkin_INCLUDE_DIRS}
  /home/unitree/ubrobot/ros_depends_ws/devel_isolated/rtabmap/include/rtabmap-0.21
)

set_target_properties(${MODULE_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${MODULE_OUTPUT_PATH}
    PREFIX ""  # 移除默认的lib前缀（避免生成librs_odom_module.so）
    SUFFIX ".so"  # 强制后缀为.so
)