
#ifndef __RGBD_SLAM_NODE_HPP__
#define __RGBD_SLAM_NODE_HPP__

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>

#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"

#include <cv_bridge/cv_bridge.h>

#include"System.h"
#include"Frame.h"
#include "Map.h"
#include "Tracking.h"

class RgbdSlamNode : public rclcpp::Node
{
public:
    RgbdSlamNode(ORB_SLAM2::System* pSLAM);

    ~RgbdSlamNode();

private: 
    using ImageMsg = sensor_msgs::msg::Image;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> approximate_sync_policy;

    void GrabRGBD(const sensor_msgs::msg::Image::SharedPtr msgRGB, const sensor_msgs::msg::Image::SharedPtr msgD);

    ORB_SLAM2::System* m_SLAM;

    cv_bridge::CvImageConstPtr cv_ptrRGB;
    cv_bridge::CvImageConstPtr cv_ptrD;

    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image> > rgb_sub;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image> > depth_sub;

    std::shared_ptr<message_filters::Synchronizer<approximate_sync_policy> > syncApproximate;
    
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_rgb;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_depth;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_tcw;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_camerapath;
    nav_msgs::msg::Path camerapath;
};

#endif
