#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include "KCF_Tracker.h"

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    
    rclcpp::spin(std::make_shared<ImageConverter>());
    return 0;
}

