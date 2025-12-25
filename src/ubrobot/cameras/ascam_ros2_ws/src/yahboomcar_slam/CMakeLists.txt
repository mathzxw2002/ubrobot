cmake_minimum_required(VERSION 3.5)
project(yahboomcar_slam)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Find ROS2 packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(OpenCV REQUIRED)

# Find PCL
find_package(PCL REQUIRED COMPONENTS common io visualization filters)

# Include directories
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# -------------------------------
# point_cloud_lib
# -------------------------------
add_library(point_cloud_lib
  src/point_cloud.cpp
)

target_include_directories(point_cloud_lib PUBLIC
  include
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

ament_target_dependencies(point_cloud_lib
  rclcpp
  sensor_msgs
  std_msgs
  tf2_ros
  pcl_conversions
  cv_bridge
)

target_link_libraries(point_cloud_lib
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  gdal
  curl
  tiff
)

# -------------------------------
# pointcloud_mapping executable
# -------------------------------
add_executable(pointcloud_mapping
  src/pointcloud_mapping_node.cpp
)

target_include_directories(pointcloud_mapping PUBLIC
  include
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

ament_target_dependencies(pointcloud_mapping
  rclcpp
  sensor_msgs
  std_msgs
  tf2_ros
  pcl_conversions
  cv_bridge
)

target_link_libraries(pointcloud_mapping
  point_cloud_lib
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  gdal
  curl
  tiff
)

# Install targets
install(TARGETS
  point_cloud_lib
  pointcloud_mapping
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# Install include files
install(DIRECTORY include/
  DESTINATION include/
)
install(DIRECTORY launch params rviz
  DESTINATION share/${PROJECT_NAME}
)
ament_package()
