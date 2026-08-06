# Plan: Move Gradio Console to Pi

## Goal
Eliminate the local operator console service. The browser accesses the Pi directly at http://192.168.18.233:7863.

## Changes

### 1. Create `deploy/operator-console/Dockerfile`
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements-operator-console.txt .
RUN pip install --no-cache-dir -r requirements-operator-console.txt
COPY src/chat_ui /app/chat_ui
COPY src/ubrobot_contracts /app/ubrobot_contracts
COPY assets/icon /app/assets/icon
RUN mkdir -p /app/workspaces
ENV PYTHONPATH=/app
EXPOSE 7863
CMD ["python", "-m", "chat_ui.app"]
```

### 2. Build on Pi
```bash
docker build -f deploy/operator-console/Dockerfile -t ubrobot/operator-console:20260806 .
```
Context: the build dir (~/ubrobot-builds/20260806-fix-branch). ARM64 wheels exist for all deps (pure Python + pydantic-core/psutil have aarch64 wheels).

### 3. Run container
```bash
docker run -d --name operator-console --network host \
  -e UBROBOT_CHAT_BACKEND=robot-edge \
  -e UBROBOT_EDGE_URL=http://127.0.0.1:8780 \
  -e UBROBOT_EDGE_TOKEN_FILE=/app/config/tokens.json \
  -e UBROBOT_CHAT_MEDIA=off \
  -e UBROBOT_CHAT_TLS=off \
  -e UBROBOT_EDGE_HARDWARE_AUTHORITY=true \
  -e UBROBOT_EDGE_ESTOP_EXEMPTED=true \
  -v <build>/deploy/robot-edge/config/tokens.json:/app/config/tokens.json:ro \
  --restart unless-stopped \
  ubrobot/operator-console:20260806
```

### 4. Stop local console
- `scripts/start_console_hardware.ps1 stop` on the Windows machine.

### 5. Verify
- Browser: http://192.168.18.233:7863
- Console talks to robot-edge at localhost:8780 (same Pi, no network hop)
- Token loaded from mounted tokens.json
- Chat + timeline + camera panel work

## Key decisions
- **No TLS**: LAN access, same as current local setup
- **Host network**: console accesses robot-edge on localhost:8780 (no DDS, just HTTP)
- **Token via volume mount**: reads the same tokens.json as robot-edge
- **No ROS**: pure Python (gradio + fastapi + httpx), no rclpy needed
- **restart: unless-stopped**: survives Pi reboot

## Risks
- Pi CPU: Gradio is a Python web framework, adds some CPU. But it's lightweight compared to the EMOS containers. Monitor load.
- ARM64 wheel availability: all deps have aarch64 wheels (verified: pure Python or psutil/pydantic-core have binary wheels)
- Build time: ~3-5 min on Pi (pip download + install)
