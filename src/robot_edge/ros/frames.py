"""Latest RealSense frame cache for the Operator Console camera view (M8).

Subscribes the robot edge's own rclpy node to the color image topic and
keeps the newest frame as a small JPEG. The Robot Edge HTTP endpoint
``GET /v1/camera/frame`` serves it to the workstation frontend so operators
can "see what the robot sees". Raw frames never cross the boundary; only the
compressed JPEG leaves the service.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

from ubrobot_contracts.telemetry import (
    TelemetryChannel,
    TelemetrySnapshot,
    TelemetryState,
    TimestampedSample,
)

_logger = logging.getLogger("ubrobot.robot_edge.frames")

COLOR_IMAGE_TOPIC = "/camera/camera/color/image_raw"
_MAX_ENCODE_INTERVAL = 0.5  # seconds between JPEG re-encodes


class RosFrameCache:
    """Caches the latest color frame as a JPEG (robot-side read-only).

    Uses its own rclpy node and a dedicated executor thread so callbacks are
    processed independently of the telemetry graph spins.
    """

    def __init__(self, *, topic: str = COLOR_IMAGE_TOPIC) -> None:
        import rclpy  # noqa: PLC0415 - ROS-only
        from rclpy.executors import SingleThreadedExecutor  # noqa: PLC0415
        from rclpy.node import Node  # noqa: PLC0415

        if not rclpy.ok():
            rclpy.init(args=[])
        self._node = Node("robot_edge_frames")
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, daemon=True, name="robot-edge-frames"
        )
        self._topic = topic
        self._jpeg: Optional[bytes] = None
        self._last_stamp = 0.0
        self._last_encode = 0.0
        self._lock = threading.Lock()
        self._sub = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        from rclpy.qos import qos_profile_sensor_data  # noqa: PLC0415
        from sensor_msgs.msg import Image  # noqa: PLC0415 - ROS-only

        self._sub = self._node.create_subscription(
            Image,
            self._topic,
            self._on_image,
            qos_profile_sensor_data,
        )
        self._spin_thread.start()
        self._started = True
        _logger.info(
            "subscribed to %s with sensor_data QoS (spin thread started)",
            self._topic,
        )

    def _on_image(self, msg: Any) -> None:
        now = time.monotonic()
        if now - self._last_encode < _MAX_ENCODE_INTERVAL:
            return  # rate-limit JPEG encode
        self._last_encode = now
        try:
            import cv2  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415

            arr = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(
                (msg.height, msg.width, 3)
            )
            ok, jpg = cv2.imencode(
                ".jpg",
                cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 70],
            )
            if ok:
                with self._lock:
                    self._jpeg = jpg.tobytes()
                first_frame = self._last_stamp == 0.0
                self._last_stamp = now
                if first_frame:
                    _logger.info(
                        "first frame cached topic=%s size=%dx%d jpeg_bytes=%d",
                        self._topic,
                        msg.width,
                        msg.height,
                        len(self._jpeg),
                    )
        except Exception:
            pass  # best-effort caching; no frame -> endpoint 404s

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    def telemetry(self, *, now: Optional[Any] = None) -> TelemetrySnapshot:
        """Optional camera-channel snapshot (metadata only, no frame)."""
        jpeg = self.latest_jpeg()
        state = (
            TelemetryState.AVAILABLE
            if jpeg is not None
            else TelemetryState.DISCONNECTED
        )
        return TelemetrySnapshot(
            channel=TelemetryChannel.CAMERA,
            latest=TimestampedSample(
                timestamp=now or datetime.now(timezone.utc),
                state=state,
                value={
                    "source": "robot-edge:ros",
                    "topic": self._topic,
                    "jpeg_bytes": len(jpeg) if jpeg is not None else 0,
                },
            ),
            sequence=1,
        )

    def stop(self) -> None:
        if self._sub is not None:
            try:
                self._sub.destroy()
            except Exception:
                pass
            self._sub = None
        if self._started:
            try:
                self._executor.shutdown(timeout_sec=2.0)
            except Exception:
                pass
            self._spin_thread.join(timeout=3.0)
            self._started = False
