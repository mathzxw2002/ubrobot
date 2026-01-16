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
// 必须包含OpenCV头文件（处理图像）
#include <opencv2/opencv.hpp>

namespace py = pybind11;

// 调试模式：0 关闭日志，1 开启
#define DEBUG_MODE 0

// 相机内参结构体（方便Python调用）
struct CameraIntrinsics {
    float fx;       // 焦距x
    float fy;       // 焦距y
    float cx;       // 主点x
    float cy;       // 主点y
    float scale;    // 深度缩放因子（depth值 = 原始值 * scale）
    int width;      // 图像宽度
    int height;     // 图像高度
};

// 速度数据结构体（对应 ROS 的 twist）
struct OdomTwist {
    float linear_x;  // 线速度（前进方向，m/s）
    float angular_z; // 角速度（绕z轴，rad/s）
};

class RealsenseOdom {
public:
    // 构造函数
    RealsenseOdom(const std::string& cameraSerial = "") {
        // 1. 初始化相机
        std::string calibFolder = ".";
        std::string cameraName = cameraSerial.empty() ? "419522070679" : cameraSerial;
        
        std::lock_guard<std::mutex> lock(init_mutex_);
        if(camera_initialized_) {
            throw std::runtime_error("相机已初始化，禁止重复创建实例");
        }

        // 修正：RTAB-Map 的 CameraRealSense2::init 只有两个参数，且无 getLastError()
        bool camera_init_ok = camera_.init(calibFolder, cameraName);
        if(!camera_init_ok) {
            std::string errMsg = "RealSense D435i 初始化失败: 相机连接失败或序列号错误";
            std::cerr << errMsg << std::endl;
            throw std::runtime_error(errMsg);
        }
        camera_initialized_ = true;

        // 2. 配置里程计
        rtabmap::ParametersMap params;
        params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kOdomStrategy(), "0"));
        params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kVisMaxFeatures(), "1000"));
        
        odom_ = rtabmap::Odometry::create(params);

        // 3. 初始化相机内参（修正：方法已提前声明）
        this->init_camera_intrinsics();

        // 4. 初始化速度相关变量
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

    // 获取单帧位姿 [x,y,z,roll,pitch,yaw]
    std::vector<float> get_pose() {
        // 修正：锁的正确语法（std::unique_lock 不需要指定 write，构造时自动加锁）
        std::unique_lock<std::shared_mutex> lock(data_rw_mutex_);
        
        // 采集最新数据
        this->capture_latest_data();
        
        if(!latest_data_.isValid() || odom_ == nullptr) {
            return {};
        }

        rtabmap::OdometryInfo info;
        rtabmap::Transform pose = odom_->process(latest_data_, &info);

        if(pose.isNull()) {
            return {};
        }

        float x, y, z, r, p, yaw;
        pose.getTranslationAndEulerAngles(x, y, z, r, p, yaw);
        return {x, y, z, r, p, yaw};
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

        // 提取位姿
        float x, y, z, r, p, yaw;
        pose.getTranslationAndEulerAngles(x, y, z, r, p, yaw);
        // 修正：SensorData 获取时间戳的正确方法是 getTimestamp()
        double curr_time = latest_data_.getTimestamp();

        // 计算速度
        if(last_pose_valid_) {
            double dt = curr_time - last_time_;
            if(dt > 1e-3) { 
                // 线速度
                float dx = x - last_x_;
                float dy = y - last_y_;
                float dist = std::sqrt(dx*dx + dy*dy);
                twist_.linear_x = dist / dt;

                // 角速度（处理角度环绕）
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

    // 获取速度
    OdomTwist get_odom_twist() {
        // 修正：读锁的正确语法
        std::shared_lock<std::shared_mutex> lock(data_rw_mutex_);
        return twist_;
    }

    // 获取RGB图像（返回numpy数组）
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
        py::array_t<uint8_t> rgb_array({rgb_mat.rows, rgb_mat.cols, 3}, 
                                      {rgb_mat.cols * 3, 3, 1}, 
                                      rgb_mat.data);
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
        py::array_t<float> depth_array({depth_float.rows, depth_float.cols}, 
                                       {depth_float.cols * sizeof(float), sizeof(float)}, 
                                       depth_float.data);
        return depth_array;
    }

    // 获取最新帧时间戳（秒）
    double get_latest_timestamp() {
        std::shared_lock<std::shared_mutex> lock(data_rw_mutex_);
        if(!latest_data_.isValid()) {
            return 0.0;
        }
        return latest_data_.getTimestamp();
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

    // 设置里程计参数
    void set_odom_param(const std::string& key, const std::string& value) {
        std::unique_lock<std::shared_mutex> lock(data_rw_mutex_);
        
        if(odom_) {
            odom_->getParameters().insert({key, value});
            if(DEBUG_MODE) {
                std::cout<<"[DEBUG] 设置参数 " << key << " = " << value << std::endl;
            }
        } else {
            throw std::runtime_error("里程计未初始化，无法设置参数");
        }
    }

    // 析构函数
    ~RealsenseOdom() {
        std::unique_lock<std::shared_mutex> lock(data_rw_mutex_);
        
        if(odom_) {
            delete odom_;
            odom_ = nullptr;
        }
        // 修正：移除 camera_.close()（私有方法，RTAB-Map 自动释放）
        camera_initialized_ = false;
        if(DEBUG_MODE) {
            std::cout << "[DEBUG] 资源已释放" << std::endl;
        }
    }

private:
    // 核心成员变量
    rtabmap::CameraRealSense2 camera_;
    rtabmap::Odometry* odom_ = nullptr;
    rtabmap::SensorData latest_data_;
    CameraIntrinsics intrinsics_;
    std::shared_mutex data_rw_mutex_;
    std::mutex init_mutex_;
    bool camera_initialized_ = false;

    // 速度计算相关变量
    OdomTwist twist_;
    bool last_pose_valid_;
    float last_x_, last_y_, last_yaw_;
    double last_time_;

    // 采集最新传感器数据（私有方法，提前声明）
    void capture_latest_data() {
        latest_data_ = camera_.takeImage();
        
        if(!latest_data_.isValid()) {
            if(DEBUG_MODE) {
                std::cout << "[DEBUG] 采集数据无效" << std::endl;
            }
        }
    }

    // 初始化相机内参（私有方法，提前声明）
    void init_camera_intrinsics() {
        const std::vector<rtabmap::CameraModel>& camera_models = camera_.getCameraModels();
        if(camera_models.empty()) {
            throw std::runtime_error("无法获取相机内参：相机模型为空");
        }

        const rtabmap::CameraModel& model = camera_models[0];
        intrinsics_.fx = model.fx();
        intrinsics_.fy = model.fy();
        intrinsics_.cx = model.cx();
        intrinsics_.cy = model.cy();
        intrinsics_.width = model.width();
        intrinsics_.height = model.height();
        intrinsics_.scale = 0.001f; // 16位深度值（mm）转米

        if(DEBUG_MODE) {
            std::cout << "[DEBUG] 深度缩放因子: " << intrinsics_.scale << std::endl;
        }
    }
};

// pybind11 绑定
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
        .def("get_pose", &RealsenseOdom::get_pose, "获取6D位姿 [x,y,z,roll,pitch,yaw]")
        .def("get_pose_with_twist", &RealsenseOdom::get_pose_with_twist, "获取位姿并计算速度")
        .def("get_odom_twist", &RealsenseOdom::get_odom_twist, "获取线速度/角速度")
        .def("get_rgb_image", &RealsenseOdom::get_rgb_image, "获取RGB图像（numpy数组，HWC）")
        .def("get_depth_image", &RealsenseOdom::get_depth_image, "获取深度图像（numpy数组，单位米）")
        .def("get_latest_timestamp", &RealsenseOdom::get_latest_timestamp, "获取硬件时间戳（秒）")
        .def("get_camera_intrinsics", &RealsenseOdom::get_camera_intrinsics, "获取相机内参（字典）")
        .def("set_odom_param", &RealsenseOdom::set_odom_param, "设置里程计参数");
}