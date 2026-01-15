#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <rtabmap/core/Odometry.h>
#include <rtabmap/core/CameraRealsense2.h>
#include <rtabmap/core/SensorData.h>
#include <rtabmap/core/Parameters.h>

namespace py = pybind11;

class RealsenseOdom {
public:
    RealsenseOdom() {
        // 1. Initialize Camera (Internal to C++)
        // This handles alignment and RealSense initialization
        if(!camera_.init()) {
            throw std::runtime_error("RealSense D435i could not be initialized.");
        }

        // 2. Configure Odometry Parameters
        rtabmap::ParametersMap params;
        // F2M (Frame-to-Map) is more robust for lower framerates
        params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kOdomStrategy(), "0"));
        // Optional: Increase feature count to help with lower frequencies
        params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kVisMaxFeatures(), "1000"));
        
        odom_ = rtabmap::Odometry::create(params);
    }

    // This method is called from Python whenever you want a new pose
    std::vector<float> get_pose() {
        // Trigger a single hardware frame capture
        rtabmap::SensorData data = camera_.takeImage();
        
        if(data.isEmpty()) {
            return {}; // Return empty list if frame capture failed
        }

        rtabmap::OdometryInfo info;
        rtabmap::Transform pose = odom_->process(data, &info);

        if(pose.isNull()) {
            return {}; // Return empty if tracking is lost
        }

        // Return [x, y, z, roll, pitch, yaw]
        float x, y, z, r, p, yaw;
        pose.getTranslationAndEulerAngles(x, y, z, r, p, yaw);
        return {x, y, z, r, p, yaw};
    }

    virtual ~RealsenseOdom() {
        delete odom_;
    }

private:
    rtabmap::CameraRealsense2 camera_;
    rtabmap::Odometry * odom_;
};

PYBIND11_MODULE(rs_odom_module, m) {
    py::class_<RealsenseOdom>(m, "RealsenseOdom")
        .def(py::init<>())
        .def("get_pose", &RealsenseOdom::get_pose);
}
