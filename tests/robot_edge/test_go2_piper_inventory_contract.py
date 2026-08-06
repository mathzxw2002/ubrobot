"""Contract test for the Go2+Piper integration inventory (Task 1).

Guards two version-controlled artifacts:
  - deploy/robot-edge/config/go2-piper.example.env
  - docs/hardware/go2-piper-integration-inventory.md

It checks that the dock-environment inventory is *complete* (every required
key is defined) and *sanitized* (no real IPs, tokens, or serials land in the
repo). It does NOT verify that dock-specific values are true on hardware --
that verification status is tracked in the inventory doc as
``verified on hardware`` / ``fixture only`` / ``unknown``.

Intentionally self-contained: only stdlib, never imports rclpy, piper_sdk,
unitree_sdk2py, or any robot_edge module, so it runs on a plain workstation
(including the Windows dev host) before any ROS environment is available.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / "deploy" / "robot-edge" / "config" / "go2-piper.example.env"
INVENTORY_PATH = REPO_ROOT / "docs" / "hardware" / "go2-piper-integration-inventory.md"

# Required inventory keys. Dock-specific values may be placeholders
# (``TODO-verify-on-dock`` / ``pending``) until Step 3 hardware verification is
# done; the contract requires presence + sanitation, not truth.
REQUIRED_ENV_KEYS = (
    # --- platform identity ---
    "UBROBOT_PLATFORM",
    "UBROBOT_GRASP_PLATFORM",
    # --- dock environment (frozen; hard prerequisite) ---
    "DOCK_HOST_OS",
    "DOCK_JETPACK_VERSION",
    "DOCK_CUDA_VERSION",
    "DOCKER_BASE_IMAGE",
    "DOCK_DOCKER_NOBLE_VERIFIED",
    "DOCK_REALSENSE_KERNEL_OK",
    "DOCK_CAN0_OK",
    # --- Go2 ROS 2 interface (unitree_ros2 bridge, containerized on dock) ---
    "GO2_ROS2_BRIDGE_SOURCE",
    "GO2_ROS2_BRIDGE_VERSION",
    "GO2_INTERFACE_TYPE",
    "GO2_BRIDGE_RMW",
    "GO2_CMD_VEL_TOPIC",
    "GO2_ODOM_TOPIC",
    "GO2_IMU_TOPIC",
    "GO2_JOINT_STATES_TOPIC",
    "GO2_TF_ROOT_FRAME",
    "GO2_STAND_PRIMITIVE",
    "GO2_STOP_PRIMITIVE",
    # --- Piper (local on dock) ---
    "PIPER_CAN_INTERFACE",
    "PIPER_DRIVER_START",
    # --- RGB-D perception input ---
    "RGBD_COLOR_TOPIC",
    "RGBD_DEPTH_TOPIC",
    "RGBD_CAMERA_INFO_SOURCE",
    # --- remote perception service (x86 GPU server; NOT on the dock) ---
    "REMOTE_PERCEPTION_SERVICE_URL",
    "REMOTE_PERCEPTION_SERVICE_ENDPOINT",
    "REMOTE_PERCEPTION_VERIFIED",
    # --- ROS domain + calibration ---
    "ROS_DOMAIN_ID",
    "CALIBRATION_VERSION",
)

_IPV4_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
# A real token/serial is typically >= 16 contiguous hex chars.
_LONG_HEX_RE = re.compile(r"\b[0-9a-fA-F]{16,}\b")


def _parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


class TestGo2PiperInventoryEnv(unittest.TestCase):
    def test_env_file_exists(self) -> None:
        self.assertTrue(ENV_PATH.is_file(), f"missing inventory env: {ENV_PATH}")

    def test_all_required_keys_present_and_nonempty(self) -> None:
        values = _parse_env(ENV_PATH)
        missing = [k for k in REQUIRED_ENV_KEYS if not values.get(k)]
        self.assertEqual(missing, [], f"missing/empty inventory keys: {missing}")

    def test_docker_base_image_is_jazzy_noble(self) -> None:
        values = _parse_env(ENV_PATH)
        self.assertEqual(
            values.get("DOCKER_BASE_IMAGE"),
            "ros:jazzy-ros-base-noble",
            "Docker base must be the Jazzy/Noble image used by the rest of the stack",
        )

    def test_go2_interface_type_is_known_enum(self) -> None:
        values = _parse_env(ENV_PATH)
        self.assertIn(
            values.get("GO2_INTERFACE_TYPE"),
            {"ros2", "ros1"},
            "GO2_INTERFACE_TYPE must be 'ros2' or 'ros1' (ros1 => ros1_bridge needed)",
        )

    def test_remote_perception_verification_status(self) -> None:
        values = _parse_env(ENV_PATH)
        self.assertIn(
            values.get("REMOTE_PERCEPTION_VERIFIED"),
            {"yes", "no", "pending"},
        )

    def test_dock_verification_fields_are_status_or_pending(self) -> None:
        values = _parse_env(ENV_PATH)
        for key in (
            "DOCK_DOCKER_NOBLE_VERIFIED",
            "DOCK_REALSENSE_KERNEL_OK",
            "DOCK_CAN0_OK",
        ):
            self.assertIn(values.get(key), {"yes", "no", "pending"}, key)

    def test_no_real_ips_tokens_or_serials_in_env(self) -> None:
        values = _parse_env(ENV_PATH)
        for key, value in values.items():
            self.assertIsNone(
                _IPV4_RE.search(value),
                f"{key} contains a raw IPv4 address; use a hostname/placeholder",
            )
            self.assertIsNone(
                _LONG_HEX_RE.search(value),
                f"{key} looks like a raw token/serial; use a placeholder",
            )


class TestInventoryDoc(unittest.TestCase):
    def test_inventory_doc_exists(self) -> None:
        self.assertTrue(
            INVENTORY_PATH.is_file(), f"missing inventory doc: {INVENTORY_PATH}"
        )

    def test_inventory_has_status_legend(self) -> None:
        text = INVENTORY_PATH.read_text(encoding="utf-8")
        for marker in ("verified on hardware", "fixture only", "unknown"):
            self.assertIn(marker, text, f"inventory must define status '{marker}'")

    def test_inventory_marks_direct_sdk_motion_deprecated(self) -> None:
        text = INVENTORY_PATH.read_text(encoding="utf-8")
        self.assertIn("unitree_go2_robot.py", text)
        self.assertIn("SportClient", text)
        self.assertTrue("废弃" in text or "deprecated" in text.lower())

    def test_inventory_records_dock_verification_probes(self) -> None:
        text = INVENTORY_PATH.read_text(encoding="utf-8")
        # The exact Noble docker probe must be referenced so it is reproducible.
        self.assertIn("ros:jazzy-ros-base-noble", text)
        self.assertIn("can0", text)


if __name__ == "__main__":
    unittest.main()
