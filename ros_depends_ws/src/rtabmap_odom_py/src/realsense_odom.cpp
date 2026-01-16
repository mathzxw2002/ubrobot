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
// 新增：用于角度计算和时间戳
#include <cmath>
#include <chrono>

namespace py = pybind11;

// 调试模式：0 关闭日志，1 开启
#define DEBUG_MODE 0

// 相机内参结构体（方便Python调用）
struct CameraIntrinsics {
    float fx;       // 焦距x
    float fy;       // 焦距y
    float cx;       // 主点x
    float cy;       // 主点y
    float scale;    // 深度缩放因子
    int width;      // 图像宽度
    int height;     // 图像高度
};

// 新增：速度数据结构体（对应 ROS 的 twist）
struct OdomTwist {
    float linear_x;  // 线速度（前进方向，m/s）
    float angular_z; // 角速度（绕z轴，rad/s）
};

class RealsenseOdom {
public:
    RealsenseOdom(const std::string& cameraSerial = "") {
        // 1. 初始化相机（原有逻辑不变）
        std::string calibFolder = ".";
        std::string cameraName = cameraSerial.empty() ? "419522070679" : cameraSerial;
        
        std::lock_guard<std::mutex> lock(init_mutex_);
        if(camera_initialized_) {
            throw std::runtime_error("相机已初始化，禁止重复创建实例");
        }
        
        if(!camera_.init(calibFolder, cameraName)) {
            std::string errMsg = "RealSense D435i 初始化失败: " + camera_.getLastError();
            std::cerr << errMsg << std::endl;
            throw std::runtime_error(errMsg);
        }
        camera_initialized_ = true;

        // 2. 配置里程计（原有逻辑不变）
        rtabmap::ParametersMap params;
        params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kOdomStrategy(), "0"));
        params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kVisMaxFeatures(), "1000"));
        
        odom_ = rtabmap::Odometry::create(params);

        // 3. 初始化相机内参（原有逻辑不变）
        init_camera_intrinsics();

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

    // 获取单帧位姿 + 同步计算速度（核心扩展）
    std::vector<float> get_pose_with_twist() {
        std::unique_lock<std::shared_mutex> lock(data_rw_mutex_, std::unique_lock::write);
        
        capture_latest_data();
        
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

        // 提取当前帧位姿（x,y,z,roll,pitch,yaw）
        float x, y, z, r, p, yaw;
        pose.getTranslationAndEulerAngles(x, y, z, r, p, yaw);
        // 提取当前帧时间戳（秒）
        double curr_time = latest_data_.timestamp();

        // 计算速度（核心：类似 odom_twist）
        if(last_pose_valid_) {
            // 时间差
            double dt = curr_time - last_time_;
            if(dt > 1e-3) { // 避免除以0
                // 1. 计算线速度（linear.x）：平面位移差 / 时间差
                float dx = x - last_x_;
                float dy = y - last_y_;
                float dist = std::sqrt(dx*dx + dy*dy);
                twist_.linear_x = dist / dt;

                // 2. 计算角速度（angular.z）：偏航角差 / 时间差
                float dyaw = yaw - last_yaw_;
                // 处理角度环绕（归一化到 [-π, π]）
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

    // 获取速度（对应 ROS 的 twist）
    OdomTwist get_odom_twist() {
        std::shared_lock<std::shared_mutex> lock(data_rw_mutex_, std::shared_lock::read);
        return twist_;
    }

    // 原有方法：仅获取位姿（保留，兼容旧代码）
    std::vector<float> get_pose() {
        std::unique_lock<std::shared_mutex> lock(data_rw_mutex_, std::unique_lock::write);
        
        capture_latest_data();
        
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

    // 其他原有方法（get_rgb_image/get_depth_image/get_latest_timestamp 等）不变
    // ... 此处省略原有方法的代码，保持和之前一致 ...

    // 析构函数（原有逻辑不变）
    ~RealsenseOdom() {
        std::unique_lock<std::shared_mutex> lock(data_rw_mutex_, std::unique_lock::write);
        
        if(odom_) {
            delete odom_;
            odom_ = nullptr;
        }
        camera_.close();
        camera_initialized_ = false;
        if(DEBUG_MODE) {
            std::cout << "[DEBUG] 资源已释放" << std::endl;
        }
    }

private:
    // 原有成员变量（不变）
    rtabmap::CameraRealSense2 camera_;
    rtabmap::Odometry* odom_ = nullptr;
    rtabmap::SensorData latest_data_;
    CameraIntrinsics intrinsics_;
    std::shared_mutex data_rw_mutex_;
    std::mutex init_mutex_;
    bool camera_initialized_ = false;

    // 新增：速度计算相关变量
    OdomTwist twist_;               // 存储计算后的速度
    bool last_pose_valid_;          // 上一帧位姿是否有效
    float last_x_, last_y_, last_yaw_; // 上一帧的x/y/yaw
    double last_time_;              // 上一帧的时间戳（秒）

    // 原有私有方法（capture_latest_data/init_camera_intrinsics 不变）
    // ... 此处省略原有私有方法的代码 ...
};

// 绑定到 Python（新增 OdomTwist 结构体和 get_odom_twist 方法）
PYBIND11_MODULE(rs_odom_module, m) {
    m.doc() = "RealSense D435i 视觉里程计模块（带速度计算）";

    // 绑定相机内参结构体（原有）
    py::class_<CameraIntrinsics>(m, "CameraIntrinsics")
        .def_readwrite("fx", &CameraIntrinsics::fx)
        .def_readwrite("fy", &CameraIntrinsics::fy)
        .def_readwrite("cx", &CameraIntrinsics::cx)
        .def_readwrite("cy", &CameraIntrinsics::cy)
        .def_readwrite("scale", &CameraIntrinsics::scale)
        .def_readwrite("width", &CameraIntrinsics::width)
        .def_readwrite("height", &CameraIntrinsics::height);

    // 绑定速度结构体（对应 ROS twist）
    py::class_<OdomTwist>(m, "OdomTwist")
        .def_readwrite("linear_x", &OdomTwist::linear_x)  // 线速度（m/s）
        .def_readwrite("angular_z", &OdomTwist::angular_z); // 角速度（rad/s）

    // 绑定核心类（扩展方法）
    py::class_<RealsenseOdom>(m, "RealsenseOdom")
        .def(py::init<const std::string&>(), py::arg("camera_serial") = "")
        .def("get_pose", &RealsenseOdom::get_pose, "获取6D位姿")
        .def("get_pose_with_twist", &RealsenseOdom::get_pose_with_twist, "获取位姿并同步计算速度")
        .def("get_odom_twist", &RealsenseOdom::get_odom_twist, "获取线速度和角速度")
        .def("get_rgb_image", &RealsenseOdom::get_rgb_image, "获取RGB图像")
        .def("get_depth_image", &RealsenseOdom::get_depth_image, "获取深度图像")
        .def("get_latest_timestamp", &RealsenseOdom::get_latest_timestamp, "获取硬件时间戳")
        .def("get_camera_intrinsics", &RealsenseOdom::get_camera_intrinsics, "获取相机内参")
        .def("set_odom_param", &RealsenseOdom::set_odom_param, "设置里程计参数");
}