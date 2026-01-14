# Installing Intel RealSense SDK on Raspberry Pi with Ubuntu 24.04

To install the Intel RealSense SDK (Librealsense) on a Raspberry Pi running Ubuntu 24.04, you need to adapt to the ARM64 architecture and resolve USB compatibility issues. Below are the optimized installation steps for Ubuntu 24.04:

### I. Preparations (Mandatory)

#### 1. Hardware and System Requirements

- Hardware: Raspberry Pi 4B/5 (requires USB 3.0 port, 4GB+ RAM recommended), Intel RealSense D400 series camera (D435/D455)

- System: Ubuntu 24.04 LTS (ARM64 version, updated to the latest)

- Power Supply: 5V/3A or higher adapter is recommended to avoid insufficient power for the camera

#### 2. System Update and Dependency Installation

```Plain Text

# Update system
sudo apt update && sudo apt upgrade -y
sudo apt autoremove -y && sudo apt clean

# Install basic dependencies (compilation tools, USB drivers, graphics libraries)
sudo apt install -y build-essential cmake git libgtk-3-dev libusb-1.0-0-dev
sudo apt install -y libglu1-mesa-dev freeglut3-dev mesa-utils udev libudev-dev
sudo apt install -y python3-dev python3-pip  # Dependencies for Python interface
```

#### 3. Upgrade CMake (Optional for Ubuntu 24.04 as default version is sufficient)

```Plain Text

# If CMake version < 3.11.4, execute the following to upgrade
wget https://cmake.org/files/v3.27/cmake-3.27.0.tar.gz
tar -zxvf cmake-3.27.0.tar.gz && cd cmake-3.27.0
./configure --prefix=/usr/local && make -j4 && sudo make install
cmake --version  # Verify version ≥ 3.11.4
```

### II. Compile and Install Librealsense SDK (libuvc Backend Recommended)

Using the `libuvc` backend avoids kernel patching and offers better compatibility:

#### 1. Clone Source Code (Specify Stable Version)

```Plain Text

git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense && git checkout v2.55.1  # Latest stable version, compatible with Ubuntu 24.04
```

#### 2. Configure Compilation Options (Key Step)

```Plain Text

mkdir build && cd build

# CMake configuration (enable Python interface, force libuvc backend, optimize for ARM performance)
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_PYTHON_BINDINGS=bool:true \
         -DFORCE_LIBUVC=true \
         -DBUILD_EXAMPLES=true \
         -DCMAKE_CXX_FLAGS="-march=armv8-a+crc -mtune=cortex-a76"  # Optimized for Raspberry Pi 5 (change to cortex-a72 for Pi 4B)
```

for Jeston Orin NX/OX
```Plain Text

mkdir build && cd build

# CMake configuration (enable Python interface, force libuvc backend, optimize for ARM performance)
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_PYTHON_BINDINGS=bool:true \
         -DFORCE_LIBUVC=true \
         -DBUILD_EXAMPLES=true
```

cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=bool:true -DFORCE_LIBUVC=true -DFORCE_RSUSB_BACKEND=TRUE 


#### 3. Compile and Install

```Plain Text

# Single-threaded compilation (avoids Raspberry Pi memory overflow, takes ~40 minutes)
make -j1

# Install to system
sudo make install
```

### III. Environment Configuration and Permission Setting

#### 1. Configure Library Path and Python Path

```Plain Text

# Permanently add library path
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib' >> ~/.bashrc
# Add Python interface path (adjust according to actual Python version, usually 3.12 for Ubuntu 24.04)
echo 'export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.12/dist-packages' >> ~/.bashrc
source ~/.bashrc  # Apply changes immediately
```

#### 2. Configure USB Device Permissions (Avoid Using sudo Every Time)

```Plain Text

# Copy device rules file
sudo cp ../config/99-realsense-libusb.rules /etc/udev/rules.d/
# Reload rules and apply
sudo udevadm control --reload-rules && sudo udevadm trigger
```

cp pyrealsense2.cpython-310-aarch64-linux-gnu.so* /home/unitree/py310_env/lib/python3.10/site-packages/
cp pyrsutils.cpython-310-aarch64-linux-gnu.so* /home/unitree/py310_env/lib/python3.10/site-packages/

check by 
python3 -c "import pyrealsense2 as rs; print(rs.pipeline)"    

### IV. Installation Verification

#### 1. Tool Verification (Quick Check)

```Plain Text

# List device information (execute after connecting the camera)
rs-enumerate-devices
# Launch visualization tool (requires desktop environment)
realsense-viewer
```

If the camera model (e.g., `Intel RealSense D435`) and image streams are displayed, the installation is successful.

#### 2. Python Interface Verification

```Plain Text

import pyrealsense2 as rs

# Test camera connection
pipeline = rs.pipeline()
config = rs.config()
pipeline.start(config)
print("Camera connected successfully!")
pipeline.stop()
```

If no errors occur during execution, the Python interface is functional.

### V. ROS2 Install 

```bash
sudo apt-get install ros-${ROS_DISTRO}-realsense2-camera
sudo apt-get install ros-${ROS_DISTRO}-realsense2-description
```

```bash
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true  

```

```bash
ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true \
  enable_depth:=true \
  enable_gyro:=true \
  enable_accel:=true \
  unite_imu_method:=linear_interpolation  # 合并 IMU 数据为 /imu/data 话题
```

```bash
# 查看里程计话题
ros2 topic list | grep odom

# 查看里程计数据（位置、姿态、速度等）
ros2 topic echo /odom
```


#### Config RTABMAP to get ODOM 
https://github.com/introlab/rtabmap_ros


# Installation 

### Binaries
```bash
sudo apt install ros-$ROS_DISTRO-rtabmap-ros
```

### From Source
* Make sure to uninstall any rtabmap binaries:
    ```
    sudo apt remove ros-$ROS_DISTRO-rtabmap*
    ```
* RTAB-Map ROS2 package:
    ```bash
    cd ~/ros2_ws
    git clone https://github.com/introlab/rtabmap.git src/rtabmap
    git clone --branch ros2 https://github.com/introlab/rtabmap_ros.git src/rtabmap_ros
    rosdep update && rosdep install --from-paths src --ignore-src -r -y
    export MAKEFLAGS="-j6" # Can be ignored if you have a lot of RAM (>16GB)
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
    ```

* To build with `rgbd_cameras>1` support and/or `subscribe_user_data` support:
    ```bash
    colcon build --symlink-install --cmake-args -DRTABMAP_SYNC_MULTI_RGBD=ON -DRTABMAP_SYNC_USER_DATA=ON -DCMAKE_BUILD_TYPE=Release
    ```
note
<img width="1310" height="609" alt="image" src="https://github.com/user-attachments/assets/51479e0f-5acc-45d9-aea7-a0a89d99da7f" />

```bash
    colcon build --symlink-install --cmake-args -DRTABMAP_SYNC_MULTI_RGBD=ON -DRTABMAP_SYNC_USER_DATA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXE_LINKER_FLAGS="-lcurl"
    ```

# Usage

* For sensor integration examples (stereo and RGB-D cameras, 3D LiDAR), see [rtabmap_examples](https://github.com/introlab/rtabmap_ros/tree/ros2/rtabmap_examples/launch) sub-folder.

* For robot integration examples (turtlebot3 and turtlebot4, nav2 integration), see [rtabmap_demos](https://github.com/introlab/rtabmap_ros/tree/ros2/rtabmap_demos) sub-folder.



# Requirements:
#   A realsense D435i
#   Install realsense2 ros2 package (ros-$ROS_DISTRO-realsense2-camera)
# Example:
#   $ ros2 launch rtabmap_examples realsense_d435i_color.launch.py
