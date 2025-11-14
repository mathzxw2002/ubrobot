#include "rgbd-slam-pose-node.hpp"

#include<opencv2/core/core.hpp>

using std::placeholders::_1;

RgbdSlamNode::RgbdSlamNode(ORB_SLAM2::System* pSLAM)
:   Node("orbslam"),
    m_SLAM(pSLAM)
{
    //rgb_sub = std::make_shared<message_filters::Subscriber<ImageMsg> >(shared_ptr<rclcpp::Node>(this), "/camera/color/image_raw");
    rgb_sub = std::make_shared<message_filters::Subscriber<ImageMsg>>(this, "/ascamera_hp60c/camera_publisher/rgb0/image");
    //depth_sub = std::make_shared<message_filters::Subscriber<ImageMsg> >(shared_ptr<rclcpp::Node>(this), "/camera/depth/image_raw");
    depth_sub = std::make_shared<message_filters::Subscriber<ImageMsg>>(this, "/ascamera_hp60c/camera_publisher/depth0/image_raw");
    syncApproximate = std::make_shared<message_filters::Synchronizer<approximate_sync_policy> >(approximate_sync_policy(10), *rgb_sub, *depth_sub);
    syncApproximate->registerCallback(&RgbdSlamNode::GrabRGBD, this);

    pub_rgb = this->create_publisher<sensor_msgs::msg::Image>("/RGBD/RGB/Image", 10);
    pub_depth = this->create_publisher<sensor_msgs::msg::Image>("/RGBD/Depth/Image", 10);
    pub_tcw = this->create_publisher<geometry_msgs::msg::PoseStamped>("/RGBD/CameraPose", 10);
    pub_odom = this->create_publisher<nav_msgs::msg::Odometry>("/RGBD/Odometry", 10);
    pub_camerapath = this->create_publisher<nav_msgs::msg::Path>("/RGBD/Path", 10);

}

RgbdSlamNode::~RgbdSlamNode()
{
    // Stop all threads
    m_SLAM->Shutdown();
    // 以TUM格式(tx,ty,tz,qx,qy,qz,qw)保存所有成功定位的帧的位姿
    // m_SLAM->SaveTrajectoryTUM("FrameTrajectory.txt");
    // 以TUM格式保存所有关键帧的位姿
    m_SLAM->SaveKeyFrameTrajectoryTUM("/root/yahboomcar_ros2_ws/software/library_ws/src/ros2-ORB_SLAM2/src/rgbd_pose/KeyFrameTrajectory.txt");
}

void RgbdSlamNode::GrabRGBD(const ImageMsg::SharedPtr msgRGB, const ImageMsg::SharedPtr msgD)
{
    // Copy the ros rgb image message to cv::Mat.
    try
    {
        cv_ptrRGB = cv_bridge::toCvShare(msgRGB);
    }
    catch (cv_bridge::Exception& e)
    {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    // Copy the ros depth image message to cv::Mat.
    try
    {
        cv_ptrD = cv_bridge::toCvShare(msgD);
    }
    catch (cv_bridge::Exception& e)
    {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }
      
    bool isKeyFrame = false;
    cv::Mat Tcw = m_SLAM->TrackRGBD(cv_ptrRGB->image, cv_ptrD->image, msgRGB->header.stamp.sec);
    // cv::Mat Tcw = m_SLAM->TrackRGBD(cv_ptrRGB->image, cv_ptrD->image, msgRGB->header.stamp.sec, isKeyFrame);
    if (!Tcw.empty()) {
        cv::Mat Twc = Tcw.inv();
        cv::Mat RWC = Twc.rowRange(0, 3).colRange(0, 3);
        cv::Mat tWC = Twc.rowRange(0, 3).col(3);

        Eigen::Matrix<double, 3, 3> eigMat;
        eigMat << RWC.at<float>(0, 0), RWC.at<float>(0, 1), RWC.at<float>(0, 2),
                RWC.at<float>(1, 0), RWC.at<float>(1, 1), RWC.at<float>(1, 2),
                RWC.at<float>(2, 0), RWC.at<float>(2, 1), RWC.at<float>(2, 2);
        Eigen::Quaterniond q(eigMat);

        geometry_msgs::msg::PoseStamped tcw_msg;
        tcw_msg.pose.position.x = tWC.at<float>(0);
        tcw_msg.pose.position.y = tWC.at<float>(1);
        tcw_msg.pose.position.z = tWC.at<float>(2);

        tcw_msg.pose.orientation.x = q.x();
        tcw_msg.pose.orientation.y = q.y();
        tcw_msg.pose.orientation.z = q.z();
        tcw_msg.pose.orientation.w = q.w();

        std_msgs::msg::Header header;
        header.stamp = msgRGB->header.stamp;
        header.frame_id = "camera";
        tcw_msg.header = header;

        // odometry information
        nav_msgs::msg::Odometry odom_msg;
        odom_msg.pose.pose.position.x = tWC.at<float>(0);
        odom_msg.pose.pose.position.y = tWC.at<float>(1);
        odom_msg.pose.pose.position.z = tWC.at<float>(2);

        odom_msg.pose.pose.orientation.x = q.x();
        odom_msg.pose.pose.orientation.y = q.y();
        odom_msg.pose.pose.orientation.z = q.z();
        odom_msg.pose.pose.orientation.w = q.w();

        odom_msg.header = header;
        odom_msg.child_frame_id = "camera";

        camerapath.header = header;
        camerapath.poses.push_back(tcw_msg);
        pub_odom->publish(odom_msg);
        pub_camerapath->publish(camerapath);  //相机轨迹

        // RCLCPP_INFO(this->get_logger(), "tWC.at<float>(0): %f", tWC.at<float>(0));
        if (isKeyFrame) {
            RCLCPP_INFO(this->get_logger(), "KeyFrame");
            pub_tcw->publish(tcw_msg);        //Tcw位姿信息
            pub_rgb->publish(*msgRGB);
            pub_depth->publish(*msgD);
        }
    } else {
        RCLCPP_ERROR(this->get_logger(), "Twc is empty ...");
    }
}
