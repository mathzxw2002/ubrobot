# Operator Console - Gradio/FastAPI web UI for the robot.
#
# Uses the EMOS image as base (already on the Pi, Docker Hub unreachable
# from CN). Python 3.12 + pip are available; ROS is present but unused
# by the console (it talks to robot-edge via HTTP, not rclpy).
FROM ubrobot/emos:jazzy-20260806-fix

WORKDIR /app

COPY requirements-operator-console.txt .
# Drop version pins for packages installed by Debian (psutil, pydantic-core)
# to avoid "Cannot uninstall" errors; the system versions work fine.
RUN sed -i '/^psutil/d; /^pydantic-core/d' requirements-operator-console.txt && \
    python3 -m pip install --no-cache-dir --break-system-packages \
      -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
      -r requirements-operator-console.txt

COPY src/chat_ui /app/chat_ui
COPY src/ubrobot_contracts /app/ubrobot_contracts
COPY assets/icon /app/assets/icon
RUN mkdir -p /app/workspaces

ENV PYTHONPATH=/app
ENV UBROBOT_CHAT_HOST=0.0.0.0
ENV UBROBOT_CHAT_PORT=7863

# Override the ROS entrypoint; the console is pure Python (no ROS needed).
ENTRYPOINT ["python3"]
EXPOSE 7863
CMD ["-m", "chat_ui.app"]
