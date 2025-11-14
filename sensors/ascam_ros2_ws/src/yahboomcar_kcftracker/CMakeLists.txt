cmake_minimum_required(VERSION 3.5)
project(yahboomcar_kcftracker)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV 4.5.4 EXACT REQUIRED)
find_package(cv_bridge REQUIRED )
find_package(image_transport REQUIRED)

include_directories(include/yahboomcar_kcftracker)

#link_libraries("/usr/lib/hbbpu/libopencv_world.so.3.4")

add_executable(kcf_tracker src/KCF_Tracker_main.cpp  src/kcf_tracker.cpp include/yahboomcar_kcftracker/kcftracker.cpp include/yahboomcar_kcftracker/fhog.cpp include/yahboomcar_kcftracker/PID.cpp)

target_include_directories(kcf_tracker PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(kcf_tracker ${OpenCV_LIBS})



 
ament_target_dependencies(
  kcf_tracker
  "rclcpp" 
  sensor_msgs 
  geometry_msgs 
  std_msgs 
  cv_bridge
  image_transport
)

install(TARGETS kcf_tracker
  DESTINATION lib/${PROJECT_NAME})
  
install(
	DIRECTORY
	launch
	DESTINATION share/${PROJECT_NAME}
	)  
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
