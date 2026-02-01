# Prepare a temporary directory for automatic building and installation.
build_dir="$HOME/tmp_build_dir"

if [ -e "$build_dir" ]; then
    rm -rf "$build_dir"
fi

mkdir $build_dir && cd $build_dir

# install apt
sudo apt update 
sudo apt install -y liborocos-kdl-dev libeigen3-dev libboost-all-dev

# install ubrobot
cd "$HOME/ubrobot"
#pip install -e .

# pytracik
#sudo apt install libboost-all-dev libeigen3-dev liborocos-kdl-dev libnlopt-dev libnlopt-cxx-dev
#cd $build_dir
#git clone https://github.com/chenhaox/pytracik.git
#cd pytracik
#python setup_linux.py install

# cyclonedds and unitree pysdk
cd ~
#rm -rf cyclonedds
#git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install
export CYCLONEDDS_HOME="~/cyclonedds/install"

cd ~
#git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip3 install -e .

# https://github.com/realsenseai/realsense-ros
#
