cmake_minimum_required(VERSION 3.5)
project(ros2_orbslam)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules/")

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(Pangolin REQUIRED)
find_package(ORB_SLAM2 REQUIRED)

# ✅ 显式加载 OpenCV（包括 calib3d 模块）
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui calib3d features2d)

message(STATUS "ORB_SLAM2 libs: ${ORB_SLAM2_LIBRARIES}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

# ==============================
# Executable targets
# ==============================

add_executable(mono 
  src/monocular/mono.cpp
  src/monocular/monocular-slam-node.cpp
)
ament_target_dependencies(mono rclcpp sensor_msgs cv_bridge ORB_SLAM2 Pangolin)
target_link_libraries(mono
  ${ORB_SLAM2_LIBRARIES}
  ${OpenCV_LIBS}
  ${Pangolin_LIBRARIES}
)

add_executable(rgbd 
  src/rgbd/rgbd.cpp
  src/rgbd/rgbd-slam-node.cpp
)
ament_target_dependencies(rgbd rclcpp sensor_msgs cv_bridge message_filters ORB_SLAM2 Pangolin)
target_link_libraries(rgbd
  ${ORB_SLAM2_LIBRARIES}
  ${OpenCV_LIBS}
  ${Pangolin_LIBRARIES}
)

add_executable(rgbd_pose
  src/rgbd_pose/rgbd_pose.cpp
  src/rgbd_pose/rgbd-slam-pose-node.cpp
)
ament_target_dependencies(rgbd_pose rclcpp sensor_msgs std_msgs geometry_msgs nav_msgs cv_bridge message_filters ORB_SLAM2 Pangolin)
target_link_libraries(rgbd_pose
  ${ORB_SLAM2_LIBRARIES}
  ${OpenCV_LIBS}
  ${Pangolin_LIBRARIES}
)

add_executable(stereo 
  src/stereo/stereo.cpp
  src/stereo/stereo-slam-node.cpp
)
ament_target_dependencies(stereo rclcpp sensor_msgs cv_bridge message_filters ORB_SLAM2 Pangolin)
target_link_libraries(stereo
  ${ORB_SLAM2_LIBRARIES}
  ${OpenCV_LIBS}
  ${Pangolin_LIBRARIES}
)

# ==============================
# Install targets
# ==============================
install(TARGETS mono rgbd rgbd_pose stereo
  DESTINATION lib/${PROJECT_NAME})

ament_package()
