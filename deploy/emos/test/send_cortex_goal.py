#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from automatika_embodied_agents.action import VisionLanguageAction

rclpy.init()
node = Node("verify_cortex_goal")
client = ActionClient(node, VisionLanguageAction, "/cortex_input_command")
if not client.wait_for_server(timeout_sec=5.0):
    print("cortex action server unavailable")
    rclpy.shutdown()
    raise SystemExit(2)

goal = VisionLanguageAction.Goal()
goal.task = "请走到椅子旁边"
print("sending goal task=", goal.task)

send_future = client.send_goal_async(goal, feedback_callback=lambda fb: print("feedback:", fb.feedback.feedback))
rclpy.spin_until_future_complete(node, send_future, timeout_sec=10)
if not send_future.done():
    print("goal send timed out")
    rclpy.shutdown()
    raise SystemExit(3)
goal_handle = send_future.result()
if not goal_handle.accepted:
    print("goal REJECTED")
    rclpy.shutdown()
    raise SystemExit(4)
print("goal ACCEPTED, waiting result...")
result_future = goal_handle.get_result_async()
rclpy.spin_until_future_complete(node, result_future, timeout_sec=90)
if result_future.done():
    result = result_future.result()
    print("RESULT status=", result.status)
    print("RESULT reply=", getattr(result.result, "reply", None))
else:
    print("result timed out (long navigation); cancelling")
    goal_handle.cancel_goal_async()
rclpy.shutdown()
