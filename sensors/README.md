

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

sudo apt install -y libzmq3-dev libturbojpeg0-dev libusb-1.0-0-dev libjpeg-dev

sudo apt update
sudo apt install python3-colcon-common-extensions


```



```



echo "source ~/ascam_ros2_ws/install/setup.bash" >> ~/.bashrc

```


1, Go to the ascam_ros2_ws directory, run sh build.sh. If success, we can get the following result.
<img width="549" height="363" alt="image" src="https://github.com/user-attachments/assets/c7fa3ef5-5a7c-4e93-8b77-cb617bc401bb" />


2, Add env

echo "source ~/ascam_ros2_ws/install/setup.bash" >> ~/.bashrc

3, Install udev rules
cd ~/ascam_ros2_ws/src/ascamera/scripts
sudo bash create_udev_rules.sh

<img width="749" height="96" alt="image" src="https://github.com/user-attachments/assets/92714af5-2266-4a32-bc07-ca6746950a09" />

4, revise configuration 

cd ~/ascam_ros2_ws/src/ascamera/configurationfiles
pwd

<img width="649" height="85" alt="image" src="https://github.com/user-attachments/assets/d53b822d-c6b7-4597-93c2-4f5c9d16cbcc" />

cd ~/ascam_ros2_ws/src/ascamera/launch
gedit hp60c.launch.py


<img width="844" height="722" alt="image" src="https://github.com/user-attachments/assets/bf9acd1a-22b7-4caf-8053-44a7e12ffcbf" />

recompile
```bash
cd ~/ascam_ros2_ws
./build.sh
```

5, run camera and visualize the image and depth data

ros2 launch ascamera hp60c.launch.py


ros2 topic list

<img width="677" height="170" alt="image" src="https://github.com/user-attachments/assets/ba35c074-64d7-4f36-82bb-bd369a106db7" />


ros2 run rqt_image_view rqt_image_view

<img width="1125" height="546" alt="image" src="https://github.com/user-attachments/assets/2a83c34e-40c1-46b3-a148-b772b648d24b" />



<img width="1123" height="554" alt="image" src="https://github.com/user-attachments/assets/fb3de589-d038-4706-8b79-78ec771d8041" />



