#!/usr/bin/env python3
"""Direct base movement: bypass Cortex/Kompass, publish raw_cmd_vel + lease.

Moves the LeKiwi base forward by a given distance at a given speed.
Requires the lekiwi-base-driver running with enable_motor_torque:=true.

Usage (run inside a ROS-enabled container, e.g. emos-nav-readonly):
    python3 /tmp/direct_move.py [speed_m/s] [duration_s]

    python3 /tmp/direct_move.py 0.1 2.0   # 0.1 m/s for 2s = 20cm
    python3 /tmp/direct_move.py 0.05 1.0  # 0.05 m/s for 1s = 5cm

Safety: the cmd_vel_guard zeros velocity within 250ms of the lease
expiring, so the robot stops automatically when the script exits.
"""
import rclpy
import sys
import time
from geometry_msgs.msg import Twist
from std_msgs.msg import String

speed = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
duration = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

rclpy.init()
node = rclpy.node.Node("direct_move")
lease_pub = node.create_publisher(String, "/navigation/command_lease", 10)
cmd_pub = node.create_publisher(Twist, "/navigation/raw_cmd_vel", 10)
time.sleep(0.5)  # let publishers match subscribers

start = time.time()
while time.time() - start < duration:
    lease_pub.publish(String(data="direct-move"))
    t = Twist()
    t.linear.x = speed
    cmd_pub.publish(t)
    time.sleep(0.05)  # 20 Hz (>4 Hz lease/command freshness threshold)

# Stop: hold lease briefly while publishing zero so the guard's last
# forwarded cmd is zero, not the previous forward velocity.
for _ in range(10):
    lease_pub.publish(String(data="direct-move"))
    cmd_pub.publish(Twist())
    time.sleep(0.05)

node.destroy_node()
rclpy.shutdown()
print(f"Done: {speed} m/s x {duration}s = {speed*duration*100:.1f}cm")
