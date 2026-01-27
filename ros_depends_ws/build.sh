sudo apt update && sudo apt install -y python3-empy
pip install empy==3.3.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
catkin_make_isolated -j4 -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DWITH_REALSENSE2=ON
