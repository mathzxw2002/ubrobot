# ROS 
source /opt/ros/noetic/setup.bash
source ./ros_depends_ws/devel_isolated/setup.bash

#export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1

export LD_PRELOAD=/home/unitree/py310_env/lib/python3.10/site-packages/scikit_learn.libs/libgomp-947d5fa1.so.1.0.0:$LD_PRELOAD

#export PYTHONPATH=$PYTHONPATH:/usr/lib/python3/dist-packages/

#export TLS_CACHE_SIZE=4194304


# Create a symbolic link inside your venv's site-packages
# Replace 'PyKDL.so' with the actual filename found above
ln -s /usr/lib/python3/dist-packages/PyKDL.cpython-38-aarch64-linux-gnu.so /home/unitree/py310_env/lib/python3.10/site-packages/PyKDL.so

ln -s  /usr/lib/python3/dist-packages/sip.cpython-38-aarch64-linux-gnu.so ~/py310_env/lib/python3.10/site-packages/sip.so

bash ros_depends_ws/src/piper_ros/can_activate.sh

python3 src/ubrobot/robots/arm_action.py 
#python examples/piper_to_piper/teleoperate.py --teleop_type=gamepad
