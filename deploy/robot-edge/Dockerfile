# Robot Edge service container
FROM python:3.13-slim

WORKDIR /app

# Install requirements first for better caching
COPY requirements-robot-edge.txt .
RUN pip install --no-cache-dir -r requirements-robot-edge.txt

# Copy source code
COPY src/ubrobot_contracts /app/ubrobot_contracts/
COPY src/robot_edge /app/robot_edge/

# Health check
HEALTHCHECK --interval=30s --timeout=5s \
    CMD curl -f http://localhost:8780/v1/health/live || exit 1

# Expose port
EXPOSE 8780

# Run service
CMD ["python", "-m", "robot_edge.app"]
