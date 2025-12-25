#include "monocular-slam-node.hpp"

#include<opencv2/core/core.hpp>

using std::placeholders::_1;

MonocularSlamNode::MonocularSlamNode(ORB_SLAM2::System* pSLAM)
:   Node("orbslam"), 
    m_SLAM(pSLAM)
{
    m_image_subscriber = this->create_subscription<ImageMsg>(
        // "/image_raw",
        "/ascamera_hp60c/camera_publisher/rgb0/image",
        10,
        std::bind(&MonocularSlamNode::GrabImage, this, std::placeholders::_1));
}

MonocularSlamNode::~MonocularSlamNode()
{
    // Stop all threads
    m_SLAM->Shutdown();

    // Save camera trajectory
    m_SLAM->SaveKeyFrameTrajectoryTUM("/home/yahboom/ascam_ros2_ws/src/ros2-ORB_SLAM2/src/monocular/KeyFrameTrajectory.txt");
}

void MonocularSlamNode::GrabImage(const ImageMsg::SharedPtr msg)
{
    // Copy the ros image message to cv::Mat.
    try
    {
        m_cvImPtr = cv_bridge::toCvCopy(msg);
    }
    catch (cv_bridge::Exception& e)
    {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }
    
    cv::Mat Tcw = m_SLAM->TrackMonocular(m_cvImPtr->image, msg->header.stamp.sec);
}
