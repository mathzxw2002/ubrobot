
export LD_PRELOAD=/home/unitree/py310_env/lib/python3.10/site-packages/scikit_learn.libs/libgomp-947d5fa1.so.1.0.0:$LD_PRELOAD

sudo chmod 666 /dev/ttyACM0

#bash ros_depends_ws/src/piper_ros/can_activate.sh

#lerobot-teleoperate \
#    --robot.type=piper \
#    --robot.id=piper_arm \
#    --robot.can_interface=can0 \
#    --robot.bitrate=1000000 \
#    --robot.include_gripper=true \
#    --robot.use_degrees=false \
#    --robot.cameras='{}' \
#    --teleop.type=so101_leader \
#    --teleop.port=/dev/ttyACM0 \
#    --teleop.use_degrees=false \
#    --teleop.id=leader_arm_so101

python examples/piper/teleoperate.py
