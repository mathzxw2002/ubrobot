"""Robot Edge hardware health readers (M6, read-only).

These modules map read-only sources (ROS topics / injected system probes)
onto the shared telemetry and capability contracts. No module here imports
pyrealsense2, piper_sdk, unitree_sdk2py, or any hardware SDK, and none
constructs motion/session clients.
"""
