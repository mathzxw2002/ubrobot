cmake_minimum_required(VERSION 3.8)
project(lekiwi_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY ros2_control urdf
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
