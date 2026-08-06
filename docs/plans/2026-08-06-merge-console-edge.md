# Plan: Merge operator-console + robot-edge into one container

## Approach
Base on the robot-edge image (has rclpy + edge code), add console deps + code,
run both via a supervisor script. Two processes, one container.

## Files to create

### 1. deploy/operator-console/Dockerfile.merged
```dockerfile
FROM ubrobot/robot-edge:20260806-fix
WORKDIR /app
COPY requirements-operator-console.txt .
RUN sed -i '/^psutil/d; /^pydantic-core/d' requirements-operator-console.txt && \
    python3 -m pip install --no-cache-dir --break-system-packages \
      -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
      -r requirements-operator-console.txt
COPY src/chat_ui /app/chat_ui
COPY src/ubrobot_contracts /app/ubrobot_contracts
COPY assets/icon /app/assets/icon
COPY deploy/operator-console/start-console-edge.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/start-console-edge.sh
ENV PYTHONPATH=/app
CMD ["/usr/local/bin/start-console-edge.sh"]
```

### 2. deploy/operator-console/start-console-edge.sh
```bash
#!/bin/bash
source /opt/ros/jazzy/setup.bash
source /opt/emos_overlay/setup.bash 2>/dev/null
export LD_LIBRARY_PATH="/opt/ros/jazzy/lib/aarch64-linux-gnu:/opt/ros/jazzy/lib/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# 1. robot-edge (venv Python, has rclpy)
/opt/edge-venv/bin/python -m robot_edge.app &
EDGE_PID=$!

sleep 3  # let edge start first

# 2. console (system Python, has gradio)
PYTHONPATH=/app python3 -m chat_ui.app &
CONSOLE_PID=$!

# either dies -> kill both -> container restarts
wait -n $EDGE_PID $CONSOLE_PID
kill $EDGE_PID $CONSOLE_PID 2>/dev/null
wait
exit 1
```

### 3. Deploy
- Build image on Pi
- Stop operator-console + ubrobot-robot-edge-fixture
- Start merged container with: --read-only --tmpfs /tmp --cap-drop ALL
  --network host --device gpiochip0-4 --restart unless-stopped
  all env vars + token mount
- Browser: http://192.168.18.233:7863 (console)
- API: http://192.168.18.233:8780 (edge, still accessible)
