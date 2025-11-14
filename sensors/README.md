

## 1, ros2 install

https://docs.ros.org/en/rolling/Installation/Ubuntu-Install-Debs.html

```
# Set locale
locale  # check for UTF-8
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
locale  # verify settings

# Enable required repositories
sudo apt install software-properties-common
sudo add-apt-repository universe

sudo apt update && sudo apt install curl -y
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb" # If using Ubuntu derivates use $UBUNTU_CODENAME
sudo apt install /tmp/ros2-apt-source.deb

# Install development tools (optional)
sudo apt update && sudo apt install ros-dev-tools

# Install ROS 2
sudo apt update
sudo apt upgrade
sudo apt install ros-jazzy-desktop

# Setup environment
source /opt/ros/jazzy/setup.bash

```


## 2, Depth Cameras

### yahboom Nuwa-HP60C Depth Camera

https://www.yahboom.com/build.html?id=11785&cid=681 password: ac06

Install dependencies for yahboom Nuwa-HP60C Depth Camera:
```
$ROS_DISTRO = jazzy
sudo apt install libgflags-dev nlohmann-json3-dev libgoogle-glog-dev ros-jazzy-image-transport ros-jazzy-image-publisher


sudo apt update
sudo apt install python3-colcon-common-extensions


```



```



echo "source ~/ascam_ros2_ws/install/setup.bash" >> ~/.bashrc

```
