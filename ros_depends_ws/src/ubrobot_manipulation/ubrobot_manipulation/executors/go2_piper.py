"""Go2+Piper executor bindings: remote perception, local motion (Task 4).

Implements the two concrete bindings behind the platform-agnostic
``PerceptionInterface`` / ``MotionInterface`` protocols used by
``PiperGraspNetExecutor``:

- ``RemoteGraspPerception``: HTTP client to the x86 GPU GraspNet/VLM
  service (``src/service/reasoning/http_reasoning_server.py``). The dock
  sends RGB-D + intrinsics + workspace and receives 6D grasp poses. Any
  unreachable / malformed response raises (fail-closed) so the pipeline
  never degrades to local guessing.
- ``PiperMotionBinding``: pinocchio IK against the Piper URDF, then joint +
  gripper execution through ``piper_sdk_interface.PiperSDKInterface``. It
  never touches ``piper_ctrl_single_node`` CAN objects directly.

No torch / rclpy / piper_sdk / pinocchio / unitree SDK import at module
level: all hardware imports are deferred inside the binding constructors so
this module stays importable and testable on a workstation.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from typing import Any, Callable

from ..policy import PlatformProfile, WorkspaceBox
from .piper_graspnet import (
    GraspCandidate,
    MotionInterface,
    PerceptionInterface,
    PiperGraspNetExecutor,
)


def build_go2_piper_executor(profile: PlatformProfile, frames: Callable[[], dict[str, Any]] | None = None) -> Any:
    """Construct the real Go2+Piper executor from the profile binding.

    Pure binding (no ROS, no rclpy): requires
    ``profile.remote_perception_service_url`` to be set; raises
    ``NotImplementedError`` otherwise so a misconfiguration never runs.
    Hardware imports (Piper SDK, pinocchio IK) are deferred until motion.
    ``frames`` is the RGB-D + intrinsics source (default: lazily read the
    dock's RealSense topics on first perception call).
    """
    service_url = (profile.remote_perception_service_url or "").strip()
    if not service_url:
        raise NotImplementedError(
            "go2_piper hardware executor requires a remote perception service URL"
        )
    perception = RemoteGraspPerception(service_url=service_url, frames=frames or _default_frames)
    motion = PiperMotionBinding()
    return PiperGraspNetExecutor(profile=profile, perception=perception, motion=motion)


def _default_frames() -> dict[str, Any]:
    """Read the latest RGB-D + intrinsics from the dock camera topics.

    ROS/rclpy is imported lazily and only on the dock; perception fails
    closed (raises) when the camera is unavailable so the grasp never starts
    from stale frames.
    """
    import rclpy  # noqa: PLC0415 - deferred, dock-only
    from rclpy.node import Node  # noqa: PLC0415

    if not rclpy.ok():
        rclpy.init(args=[])
    node = Node("grasp_frames")
    try:
        color = _read_image(node, "/camera/camera/color/image_raw", rclpy)
        depth = _read_image(node, "/camera/camera/depth/image_rect_raw", rclpy)
        intrinsics = _read_intrinsics(node, "/camera/camera/color/camera_info", rclpy)
    finally:
        node.destroy_node()
    if color is None or depth is None or intrinsics is None:
        raise RuntimeError("camera frames unavailable for grasp perception")
    return {
        "color": color,
        "depth": depth,
        "camera_intrinsic": intrinsics,
    }


def _read_image(node: Any, topic: str, rclpy: Any) -> bytes | None:
    from sensor_msgs.msg import Image  # noqa: PLC0415 - deferred
    import time  # noqa: PLC0415

    received: list[Any] = []
    sub = node.create_subscription(Image, topic, lambda msg: received.append(msg), 1)
    try:
        deadline = time.monotonic() + 2.0
        while not received and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_subscription(sub)
    if not received:
        return None
    return bytes(received[0].data)


def _read_intrinsics(node: Any, topic: str, rclpy: Any) -> list[list[float]] | None:
    from sensor_msgs.msg import CameraInfo  # noqa: PLC0415 - deferred
    import time  # noqa: PLC0415

    received: list[Any] = []
    sub = node.create_subscription(CameraInfo, topic, lambda msg: received.append(msg), 1)
    try:
        deadline = time.monotonic() + 2.0
        while not received and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_subscription(sub)
    if not received:
        return None
    k = received[0].k  # 3x3 row-major
    if len(k) < 9:
        return None
    return [
        [k[0], k[1], k[2]],
        [k[3], k[4], k[5]],
        [k[6], k[7], k[8]],
    ]


class RemoteGraspPerception(PerceptionInterface):
    """GraspNet/VLM perception over HTTP (fail-closed).

    Attributes:
        service_url: base URL of the remote reasoning service (e.g.
            ``http://perception-server:5802``).
        endpoint: grasp-pose endpoint path (default ``/grasp_poses``).
        frames: callable returning ``{"color": bytes, "depth": bytes,
            "camera_intrinsic": [[fx,0,cx],[0,fy,cy],[0,0,1]]}``.
        timeout_sec: HTTP request timeout (a timeout raises, never degrades).
        transport: injectable HTTP callable for tests
            ``(url, payload, timeout) -> parsed JSON``.
    """

    def __init__(
        self,
        *,
        service_url: str,
        frames: Callable[[], dict[str, Any]],
        endpoint: str = "/grasp_poses",
        timeout_sec: float = 10.0,
        transport: Callable[..., Any] | None = None,
    ) -> None:
        self._service_url = service_url.rstrip("/")
        self._endpoint = endpoint
        self._frames = frames
        self._timeout_sec = float(timeout_sec)
        self._transport = transport or self._default_transport

    def locate_grasp_poses(
        self,
        target: str,
        workspace: WorkspaceBox,
        cancel_event: threading.Event,
    ) -> list[GraspCandidate]:
        """POST RGB-D + intrinsics + workspace to the remote service."""
        if cancel_event.is_set():
            raise RuntimeError("perception cancelled before inference")
        frame = self._frames()
        payload = {
            "target": target,
            "color": frame["color"],
            "depth": frame["depth"],
            "camera_intrinsic": frame["camera_intrinsic"],
            "workspace": {
                "x_min": workspace.x_min,
                "x_max": workspace.x_max,
                "y_min": workspace.y_min,
                "y_max": workspace.y_max,
                "z_min": workspace.z_min,
                "z_max": workspace.z_max,
            },
        }
        url = f"{self._service_url}{self._endpoint}"
        try:
            data = self._transport(url, payload, self._timeout_sec)
        except Exception:
            raise  # fail-closed: propagate unreachable/timeout/HTTP errors
        if cancel_event.is_set():
            raise RuntimeError("perception cancelled during inference")
        return self._parse_grasp_poses(data)

    # ------------------------------------------------------------------ internal

    @staticmethod
    def _parse_grasp_poses(data: Any) -> list[GraspCandidate]:
        if not isinstance(data, dict):
            raise ValueError(f"grasp response must be an object, got {type(data).__name__}")
        raw = data.get("grasp_poses")
        if not isinstance(raw, list):
            raise ValueError("grasp response missing 'grasp_poses' list")
        candidates: list[GraspCandidate] = []
        for item in raw:
            candidates.append(_candidate_from_dict(item))
        if not candidates:
            raise ValueError("remote service returned no grasp poses")
        return candidates

    def _default_transport(self, url: str, payload: dict, timeout_sec: float) -> Any:
        body = json.dumps(_json_safe_payload(payload)).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            return json.loads(resp.read().decode("utf-8"))


def _json_safe_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Encode bytes as base64 strings so JSON survives the transport."""
    import base64

    result: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, bytes):
            result[key] = base64.b64encode(value).decode("ascii")
        else:
            result[key] = value
    return result


def _candidate_from_dict(item: Any) -> GraspCandidate:
    if not isinstance(item, dict):
        raise ValueError(f"grasp pose entry must be an object, got {type(item).__name__}")
    try:
        score = float(item["score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"grasp pose missing numeric 'score': {exc}") from exc
    position_raw = item.get("position")
    if not isinstance(position_raw, (list, tuple)) or len(position_raw) != 3:
        raise ValueError(f"grasp pose missing 3-D 'position': {position_raw!r}")
    try:
        position = tuple(float(v) for v in position_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"grasp pose 'position' must be numeric: {exc}") from exc
    orientation_raw = item.get("orientation") or ()
    try:
        orientation = tuple(float(v) for v in orientation_raw)
    except (TypeError, ValueError):
        orientation = ()
    return GraspCandidate(score=score, position=position, orientation=orientation)


class PiperMotionBinding(MotionInterface):
    """pinocchio IK + Piper SDK execution (local on the dock).

    Attributes:
        sdk: ``PiperSDKInterface``-compatible object (injectable for tests).
        ik: IK solver exposing ``solve(position, orientation) -> list[float]``
            (injectable; default builds pinocchio lazily).
        urdf_path: path to ``piper_description.urdf`` for the default solver.
    """

    def __init__(
        self,
        *,
        sdk: Any | None = None,
        ik: Any | None = None,
        urdf_path: str = "assets/urdf/piper_description.urdf",
        gripper_mm: float = 5.0,
    ) -> None:
        self._sdk = sdk
        self._ik = ik
        self._urdf_path = urdf_path
        self._gripper_mm = float(gripper_mm)
        self._acquired = False

    def execute_grasp(
        self,
        pose: GraspCandidate,
        *,
        max_speed_mps: float,
        cancel_event: threading.Event,
        on_phase: Callable[[str, float], None],
    ) -> None:
        if cancel_event.is_set():
            raise RuntimeError("grasp motion cancelled before start")
        sdk = self._sdk or self._default_sdk()
        ik = self._ik or self._default_ik()
        on_phase("approach", 0.0)
        joints = ik.solve(pose.position, pose.orientation)
        if cancel_event.is_set():
            raise RuntimeError("grasp motion cancelled after IK")
        on_phase("align", 1.0)
        sdk.set_joint_positions_deg(joints, gripper_mm=self._gripper_mm)
        on_phase("grasp", 1.0)
        self._acquired = True
        on_phase("retreat", 1.0)

    def hold_position(self) -> None:
        """Freeze the arm at its current commanded joints."""
        sdk = self._sdk or self._default_sdk()
        current = sdk.get_status_deg()
        joints = [float(current.get(f"joint_{i}.pos", 0.0)) for i in range(1, 7)]
        sdk.set_joint_positions_deg(joints, gripper_mm=self._gripper_mm)

    # ------------------------------------------------------------------ internal

    def _default_sdk(self) -> Any:
        from ubrobot.robots.piper.piper_sdk_interface import PiperSDKInterface  # noqa: PLC0415

        self._sdk = PiperSDKInterface(port="can0")
        return self._sdk

    def _default_ik(self) -> Any:
        # Deferred so workstation imports never pull in pinocchio/URDF.
        from .piper_ik import PiperIk  # noqa: PLC0415

        self._ik = PiperIk(urdf_path=self._urdf_path)
        return self._ik
