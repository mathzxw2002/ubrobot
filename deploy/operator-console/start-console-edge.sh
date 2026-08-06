#!/bin/bash
# Supervisor: run robot-edge (:8780) + operator-console (:7863) in one container.
# If either process exits, kill both so Docker restarts the container cleanly.
source /opt/ros/jazzy/setup.bash
source /opt/emos_overlay/setup.bash 2>/dev/null
export LD_LIBRARY_PATH="/opt/ros/jazzy/lib/aarch64-linux-gnu:/opt/ros/jazzy/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cd /app

# 1. robot-edge (venv Python: has rclpy + edge deps)
/opt/edge-venv/bin/python -m robot_edge.app &
EDGE_PID=$!

# Give edge a 3-second head start so the console can connect.
sleep 3

# 2. operator-console (system Python: has gradio + fastapi)
PYTHONPATH=/app python3 -m chat_ui.app &
CONSOLE_PID=$!

# Either process dying -> kill both -> container exits -> Docker restarts.
wait -n "$EDGE_PID" "$CONSOLE_PID"
kill "$EDGE_PID" "$CONSOLE_PID" 2>/dev/null
wait
exit 1
