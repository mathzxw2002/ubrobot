// src/pointcloud_mapping_node.cpp
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "yahboomcar_slam/point_cloud.h"  // 你的头文件，类名应为 PointCloudMapper

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PointCloudMapper>();

  // 如果 PointCloudMapper 在构造中已启动回调/线程，只需要 spin
  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}
