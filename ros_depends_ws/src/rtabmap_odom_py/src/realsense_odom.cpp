#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <rtabmap/core/Odometry.h>
#include <rtabmap/core/camera/CameraRealSense2.h>
#include <rtabmap/core/SensorData.h>
#include <rtabmap/core/Parameters.h>
#include <rtabmap/core/CameraModel.h>
#include <rtabmap/core/OdometryInfo.h>
#include <string>
#include <iostream>
#include <vector>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <cmath>
#include <chrono>
#include <opencv2/opencv.hpp>

namespace py = pybind11;

// 调试模式：0 关闭日志，1 开启
#define DEBUG_MODE 0

struct CameraIntrinsics {
    float fx;
    float fy;
    float cx;
    float cy;
    float scale;    // depth = raw_value * scale
    int width;
    int height;
};

struct OdomTwist {
    float linear_x;  // m/s
    float angular_z; // rad/s
};

class RealsenseOdom {
public:
    RealsenseOdom(const std::string& cameraSerial = "") {
        std::string calibFolder = ".";
        std::string cameraName = cameraSerial.empty() ? "419522070679" : cameraSerial;
        
        std::lock_guard<std::mutex> lock(init_mutex_);
        if(camera_initialized_) {
            throw std::runtime_error("Camera has been initiallzed!");
        }

        bool camera_init_ok = camera_.init(calibFolder, cameraName);
        if(!camera_init_ok) {
            std::string errMsg = "Failed to Initialize RealSense D435i: Connection Failed or SN is wrong.";
            std::cerr << errMsg << std::endl;
            throw std::runtime_error(errMsg);
        }
        camera_initialized_ = true;

        rtabmap::ParametersMap params;
        params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kOdomStrategy(), "0"));
        params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kVisMaxFeatures(), "1000"));
        
        odom_ = rtabmap::Odometry::create(params);

        this->init_camera_intrinsics();

        last_pose_valid_ = false;
        twist_.linear_x = 0.0f;
        twist_.angular_z = 0.0f;

        if(DEBUG_MODE) {
            std::cout << "[DEBUG] Camera Intrinsics: " 
                      << "fx=" << intrinsics_.fx << ", fy=" << intrinsics_.fy 
                      << ", cx=" << intrinsics_.cx << ", cy=" << intrinsics_.cy 
                      << ", Resolution=" << intrinsics_.width << "x" << intrinsics_.height << std::endl;
        }
    }

    // 获取位姿并同步计算速度
    std::vector<float> get_pose_with_twist() {
        std::unique_lock<std::shared_mutex> lock(data_rw_mutex_);
        
        this->capture_latest_data();
        
        if(!latest_data_.isValid() || odom_ == nullptr) {
            twist_.linear_x = 0.0f;
            twist_.angular_z = 0.0f;
            return {};
        }

        rtabmap::OdometryInfo info;
        rtabmap::Transform pose = odom_->process(latest_data_, &info);

        if(pose.isNull()) {
            twist_.linear_x = 0.0f;
            twist_.angular_z = 0.0f;
            return {};
        }

        float x, y, z, r, p, yaw;
        pose.getTranslationAndEulerAngles(x, y, z, r, p, yaw);
        double curr_time = latest_data_.stamp();

        if(last_pose_valid_) {
            double dt = curr_time - last_time_;
            if(dt > 1e-3) { 
                float dx = x - last_x_;
                float dy = y - last_y_;
                float dist = std::sqrt(dx*dx + dy*dy);
                twist_.linear_x = dist / dt;

                float dyaw = yaw - last_yaw_;
                dyaw = std::atan2(std::sin(dyaw), std::cos(dyaw));
                twist_.angular_z = dyaw / dt;
            }
        }

        // 保存当前帧为上一帧
        last_x_ = x;
        last_y_ = y;
        last_yaw_ = yaw;
        last_time_ = curr_time;
        last_pose_valid_ = true;

        return {x, y, z, r, p, yaw};
    }

    OdomTwist get_odom_twist() {
        std::shared_lock<std::shared_mutex> lock(data_rw_mutex_);
        return twist_;
    }

    py::array_t<uint8_t> get_rgb_image() {
        std::shared_lock<std::shared_mutex> lock(data_rw_mutex_);
        
        if(!latest_data_.isValid() || latest_data_.imageRaw().empty()) {
            if(DEBUG_MODE) {
                std::cout << "[DEBUG] RGB图像数据无效" << std::endl;
            }
            return py::array_t<uint8_t>();
        }

        const cv::Mat& rgb_mat = latest_data_.imageRaw();
        if(rgb_mat.empty() || rgb_mat.type() != CV_8UC3) {
            return py::array_t<uint8_t>();
        }

        // 转换为numpy数组（HWC格式）
	py::array_t<uint8_t> rgb_array({static_cast<long int>(rgb_mat.rows),  static_cast<long int>(rgb_mat.cols), static_cast<long int>(3)}, {static_cast<long int>(rgb_mat.cols * 3), static_cast<long int>(3), static_cast<long int>(1)}, rgb_mat.data );

        return rgb_array;
    }

    // 获取深度图像（返回numpy数组，单位：米）
    py::array_t<float> get_depth_image() {
        std::shared_lock<std::shared_mutex> lock(data_rw_mutex_);
        
        if(!latest_data_.isValid() || latest_data_.depthOrRightRaw().empty()) {
            if(DEBUG_MODE) {
                std::cout << "[DEBUG] 深度图像数据无效" << std::endl;
            }
            return py::array_t<float>();
        }

        const cv::Mat& depth_mat = latest_data_.depthOrRightRaw();
        if(depth_mat.empty()) {
            return py::array_t<float>();
        }

        // 转换为float32（米）
        cv::Mat depth_float;
        if(depth_mat.type() == CV_16UC1) {
            depth_mat.convertTo(depth_float, CV_32FC1, intrinsics_.scale);
        } else if(depth_mat.type() == CV_32FC1) {
            depth_float = depth_mat.clone();
        } else {
            return py::array_t<float>();
        }

        // 转换为numpy数组
	py::array_t<float> depth_array({static_cast<long int>(depth_float.rows), static_cast<long int>(depth_float.cols)}, {static_cast<long int>(depth_float.cols * sizeof(float)), static_cast<long int>(sizeof(float))}, reinterpret_cast<float*>(static_cast<uchar*>(depth_float.data)));

        return depth_array;
    }

    // 获取最新帧时间戳（秒）
    double get_latest_timestamp() {
        std::shared_lock<std::shared_mutex> lock(data_rw_mutex_);
        if(!latest_data_.isValid()) {
            return 0.0;
        }
        return latest_data_.stamp();
    }

    // 获取相机内参（返回字典）
    py::dict get_camera_intrinsics() {
        std::shared_lock<std::shared_mutex> lock(data_rw_mutex_);
        
        py::dict intrinsics_dict;
        intrinsics_dict["fx"] = intrinsics_.fx;
        intrinsics_dict["fy"] = intrinsics_.fy;
        intrinsics_dict["cx"] = intrinsics_.cx;
        intrinsics_dict["cy"] = intrinsics_.cy;
        intrinsics_dict["scale"] = intrinsics_.scale;
        intrinsics_dict["width"] = intrinsics_.width;
        intrinsics_dict["height"] = intrinsics_.height;
        return intrinsics_dict;
    }

    // 析构函数
    ~RealsenseOdom() {
        std::unique_lock<std::shared_mutex> lock(data_rw_mutex_);
        
        if(odom_) {
            delete odom_;
            odom_ = nullptr;
        }
        camera_initialized_ = false;
        if(DEBUG_MODE) {
            std::cout << "[DEBUG] Resource released." << std::endl;
        }
    }

private:
    rtabmap::CameraRealSense2 camera_;
    rtabmap::Odometry* odom_ = nullptr;
    rtabmap::SensorData latest_data_;
    CameraIntrinsics intrinsics_;
    std::shared_mutex data_rw_mutex_;
    std::mutex init_mutex_;
    bool camera_initialized_ = false;

    OdomTwist twist_;
    bool last_pose_valid_;
    float last_x_, last_y_, last_yaw_;
    double last_time_;

    void capture_latest_data() {
        latest_data_ = camera_.takeImage();
        
        if(!latest_data_.isValid()) {
            if(DEBUG_MODE) {
                std::cout << "[DEBUG] 采集数据无效" << std::endl;
            }
        }
    }

    void init_camera_intrinsics() {
	if(!latest_data_.isValid()) {
            if(DEBUG_MODE) std::cout << "[DEBUG] Waiting for valid sensor data..." << std::endl;
            return; // Exit silently; don't throw yet, as first frames might be empty
        }

        const std::vector<rtabmap::CameraModel>& camera_models = latest_data_.cameraModels();

	rtabmap::CameraModel model;
        if(!latest_data_.cameraModels().empty()) { // Check Mono/RGB-D camera models
            model = latest_data_.cameraModels()[0];
        } else if(latest_data_.stereoCameraModels().empty()) {// Check Stereo camera models
	    // Access the first stereo model in the vector, then get its left camera
	    if(latest_data_.stereoCameraModels()[0].isValidForProjection()) {
	        model = latest_data_.stereoCameraModels()[0].left();
	    } else {
		throw std::runtime_error("CameraModels and stereoCameraModels are empty.");
    	    }
	} else {
	    throw std::runtime_error("No camera models found in latest_data_");
	}

        intrinsics_.fx = model.fx();
        intrinsics_.fy = model.fy();
        intrinsics_.cx = model.cx();
        intrinsics_.cy = model.cy();
        intrinsics_.width = model.imageWidth();
        intrinsics_.height = model.imageHeight();
        intrinsics_.scale = 0.001f; // 16位深度值（mm）转米

        if(DEBUG_MODE) {
            std::cout << "[DEBUG] Scale factor: " << intrinsics_.scale << std::endl;
        }
    }
};

// pybind11
PYBIND11_MODULE(rs_odom_module, m) {
    m.doc() = "RealSense D435i 视觉里程计模块（支持RGB/深度、位姿、速度）";

    // 绑定相机内参结构体
    py::class_<CameraIntrinsics>(m, "CameraIntrinsics")
        .def_readwrite("fx", &CameraIntrinsics::fx)
        .def_readwrite("fy", &CameraIntrinsics::fy)
        .def_readwrite("cx", &CameraIntrinsics::cx)
        .def_readwrite("cy", &CameraIntrinsics::cy)
        .def_readwrite("scale", &CameraIntrinsics::scale)
        .def_readwrite("width", &CameraIntrinsics::width)
        .def_readwrite("height", &CameraIntrinsics::height);

    // 绑定速度结构体
    py::class_<OdomTwist>(m, "OdomTwist")
        .def_readwrite("linear_x", &OdomTwist::linear_x)
        .def_readwrite("angular_z", &OdomTwist::angular_z);

    // 绑定核心类
    py::class_<RealsenseOdom>(m, "RealsenseOdom")
        .def(py::init<const std::string&>(), py::arg("camera_serial") = "", "构造函数：传入相机序列号")
        .def("get_pose_with_twist", &RealsenseOdom::get_pose_with_twist, "获取位姿并计算速度")
        .def("get_odom_twist", &RealsenseOdom::get_odom_twist, "获取线速度/角速度")
        .def("get_rgb_image", &RealsenseOdom::get_rgb_image, "获取RGB图像（numpy数组，HWC）")
        .def("get_depth_image", &RealsenseOdom::get_depth_image, "获取深度图像（numpy数组，单位米）")
        .def("get_latest_timestamp", &RealsenseOdom::get_latest_timestamp, "获取硬件时间戳（秒）")
        .def("get_camera_intrinsics", &RealsenseOdom::get_camera_intrinsics, "获取相机内参（字典）");
}
