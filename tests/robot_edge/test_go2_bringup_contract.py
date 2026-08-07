"""Contract test for the Go2+Piper driver bring-up + Kompass go2 configuration.

Asserts configuration-level facts only (no real Go2/Piper/dock required):

- ``deploy/go2-piper-driver/compose.yaml`` runs the Go2+Piper hardware
  driver container: Go2 bridge (subscribes ``/cmd_vel``, publishes
  ``/odom``/``/imu``/``/joint_states``) + Piper driver (subscribes
  ``/piper/joint_cmd``, maps ``can0``), with RMW + ROS_DOMAIN_ID consistent
  with the rest of the dock stack.
- ``cmd_vel_guard`` still publishes ``/cmd_vel`` and gates on
  ``/navigation/command_lease`` (unchanged safety chain).
- Kompass ``DriveManager`` still outputs ``/navigation/raw_cmd_vel``.
- The ``go2_piper`` platform carries conservative base velocity limits that
  differ from the LeKiwi default (max_base_linear_mps <= 0.2 m/s,
  max_base_angular_radps <= 0.5 rad/s).

Self-contained: no rclpy / unitree_sdk2py / piper_sdk imports.
"""

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

COMPOSE_PATH = REPO_ROOT / "deploy" / "go2-piper-driver" / "compose.yaml"
NAV_POLICY_PATH = (
    REPO_ROOT
    / "ros_depends_ws"
    / "src"
    / "ubrobot_navigation"
    / "ubrobot_navigation"
    / "policy.py"
)
CMD_VEL_GUARD_PATH = (
    REPO_ROOT
    / "ros_depends_ws"
    / "src"
    / "ubrobot_navigation"
    / "ubrobot_navigation"
    / "cmd_vel_guard.py"
)
RECIPE_PATH = REPO_ROOT / "deploy" / "emos" / "recipes" / "cortex_navigation" / "recipe.py"
PLATFORMS_PATH = REPO_ROOT / "src" / "robot_edge" / "platforms.py"


class TestGo2PiperDriverCompose(unittest.TestCase):
    def test_compose_exists(self) -> None:
        self.assertTrue(COMPOSE_PATH.is_file(), f"missing {COMPOSE_PATH}")

    def test_compose_has_driver_service(self) -> None:
        text = COMPOSE_PATH.read_text(encoding="utf-8")
        self.assertIn("go2-piper-driver", text.lower())
        self.assertIn("ros2", text.lower())

    def test_compose_sets_rmw_and_domain_consistently(self) -> None:
        text = COMPOSE_PATH.read_text(encoding="utf-8")
        # RMW must be set and be CycloneDDS (Go2 DDS interop, Task 1 finding 3).
        self.assertIn("RMW_IMPLEMENTATION", text)
        self.assertRegex(text, r"RMW_IMPLEMENTATION\s*[:=]")
        self.assertIn("ROS_DOMAIN_ID", text)
        # CycloneDDS URI configured (Go2 DDS needs CycloneDDS on eth0).
        self.assertRegex(text, r"(?i)cyclonedds")

    def test_compose_mounts_cyclonedds_config(self) -> None:
        text = COMPOSE_PATH.read_text(encoding="utf-8")
        self.assertIn("CYCLONEDDS_URI", text)

    def test_compose_maps_can0_for_piper(self) -> None:
        text = COMPOSE_PATH.read_text(encoding="utf-8")
        self.assertIn("/dev/can0", text)
        self.assertIn("PIPER_CAN_INTERFACE", text)

    def test_compose_bringup_launch_includes_piper(self) -> None:
        launch = (
            REPO_ROOT
            / "deploy"
            / "go2-piper-driver"
            / "launch"
            / "go2_piper_bringup.launch.py"
        )
        source = launch.read_text(encoding="utf-8")
        self.assertIn("go2_bridge_node", source)
        self.assertIn("piper_driver_node", source)

    def test_compose_does_not_use_host_ros1(self) -> None:
        text = COMPOSE_PATH.read_text(encoding="utf-8")
        # No noetic/ros1 reference; the driver runs in a Jazzy container.
        self.assertNotRegex(text, r"(?i)noetic|ros1_bridge|/opt/ros")


class TestCommandVelocityChain(unittest.TestCase):
    def test_cmd_vel_guard_publishes_cmd_vel(self) -> None:
        text = CMD_VEL_GUARD_PATH.read_text(encoding="utf-8")
        self.assertIn('"/cmd_vel"', text)
        self.assertIn("Twist", text)

    def test_cmd_vel_guard_subscribes_raw_and_lease(self) -> None:
        text = CMD_VEL_GUARD_PATH.read_text(encoding="utf-8")
        self.assertIn('"/navigation/raw_cmd_vel"', text)
        self.assertIn('"/navigation/command_lease"', text)

    def test_kompass_drivemanager_outputs_raw_cmd_vel(self) -> None:
        text = RECIPE_PATH.read_text(encoding="utf-8")
        self.assertIn('"/navigation/raw_cmd_vel"', text)
        self.assertIn("DriveManager", text)

    def test_navigation_server_heartbeats_lease(self) -> None:
        server = next(
            REPO_ROOT.glob(
                "ros_depends_ws/src/ubrobot_navigation/ubrobot_navigation/navigate_to_object_server.py"
            )
        )
        text = server.read_text(encoding="utf-8")
        self.assertIn('"/navigation/command_lease"', text)
        self.assertIn("heartbeat", text)


class TestGo2PiperVelocityLimits(unittest.TestCase):
    def test_platforms_defines_go2_piper_limits(self) -> None:
        text = PLATFORMS_PATH.read_text(encoding="utf-8")
        self.assertIn("go2_piper", text)
        self.assertIn("max_base_linear_mps", text)
        self.assertIn("max_base_angular_radps", text)
        # Conservative: 0.2 m/s linear, 0.5 rad/s angular caps present.
        self.assertRegex(text, r"max_base_linear_mps\s*=\s*0\.2")
        self.assertRegex(text, r"max_base_angular_radps\s*=\s*0\.5")

    def test_go2_limits_differ_from_lekiwi(self) -> None:
        text = PLATFORMS_PATH.read_text(encoding="utf-8")
        # lekiwi is slower (0.05/0.20); go2_piper is 0.2/0.5. Match within
        # each PlatformDefinition block by anchoring on the key.
        lekiwi_linear = re.search(
            r'key="lekiwi".*?max_base_linear_mps\s*=\s*([\d.]+)', text, re.S
        )
        go2_linear = re.search(
            r'key="go2_piper".*?max_base_linear_mps\s*=\s*([\d.]+)', text, re.S
        )
        self.assertIsNotNone(lekiwi_linear)
        self.assertIsNotNone(go2_linear)
        self.assertNotEqual(lekiwi_linear.group(1), go2_linear.group(1))
        self.assertLess(float(go2_linear.group(1)), 0.3)

    def test_navigation_policy_has_go2_piper_limits(self) -> None:
        text = NAV_POLICY_PATH.read_text(encoding="utf-8")
        self.assertIn("go2_piper", text)
        self.assertIn("GO2_PIPER_MAX_LINEAR_SPEED", text)
        self.assertIn("GO2_PIPER_MAX_ANGULAR_SPEED", text)

    def test_sanitize_twist_accepts_profile_limits(self) -> None:
        text = NAV_POLICY_PATH.read_text(encoding="utf-8")
        # sanitize_twist must accept per-profile max limits so go2 can use
        # its own caps without loosening the LeKiwi default.
        self.assertRegex(text, r"def\s+sanitize_twist\s*\(")
        self.assertRegex(text, r"max_linear_speed")
        self.assertRegex(text, r"max_angular_speed")


if __name__ == "__main__":
    unittest.main()
