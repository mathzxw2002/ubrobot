from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENT_ROOT = REPOSITORY_ROOT / "deploy" / "lekiwi-driver"
BRINGUP_ROOT = REPOSITORY_ROOT / "ros_depends_ws" / "src" / "lekiwi_bringup"


def read_repository_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class LeKiwiDeploymentContractTest(unittest.TestCase):
    def test_dockerfile_defaults_to_official_jazzy_ros_base(self):
        dockerfile = read_repository_file(DEPLOYMENT_ROOT / "Dockerfile")
        self.assertIn("ARG ROS_BASE_IMAGE=ros:jazzy-ros-base-noble", dockerfile)

    def test_mock_compose_is_host_networked_and_has_no_device_access(self):
        compose = read_repository_file(DEPLOYMENT_ROOT / "compose.yaml")
        self.assertIn("network_mode: host", compose)
        self.assertIn("hardware_mode:=mock", compose)
        self.assertNotIn("privileged:", compose)
        self.assertNotIn("devices:", compose)
        self.assertNotIn("/dev/", compose)

    def test_hardware_override_maps_only_the_stable_lekiwi_device(self):
        compose = read_repository_file(DEPLOYMENT_ROOT / "compose.hardware.yaml")
        self.assertNotIn("privileged:", compose)
        self.assertIn("devices:", compose)
        device_lines = [line.strip() for line in compose.splitlines() if "/dev/" in line]
        self.assertEqual(
            device_lines,
            ["- /dev/lekiwi-base:/dev/lekiwi-base"],
        )

    def test_launch_defaults_to_mock_hardware(self):
        launch = read_repository_file(BRINGUP_ROOT / "launch" / "lekiwi_driver.launch.py")
        self.assertIn('DeclareLaunchArgument("hardware_mode", default_value="mock")', launch)

    def test_controller_has_safe_timeout_and_does_not_publish_odom_tf(self):
        controllers = read_repository_file(BRINGUP_ROOT / "config" / "controllers.yaml")
        self.assertIn("cmd_vel_timeout: 0.25", controllers)
        self.assertIn("enable_odom_tf: false", controllers)

    def test_udev_rule_creates_stable_lekiwi_device(self):
        rule = read_repository_file(DEPLOYMENT_ROOT / "99-lekiwi-base.rules")
        for token in ('ATTRS{idVendor}=="1a86"', 'ATTRS{idProduct}=="55d3"'):
            self.assertIn(token, rule)
        self.assertIn('ATTRS{serial}=="5A68011386"', rule)
        self.assertIn('SYMLINK+="lekiwi-base"', rule)
        self.assertIn('GROUP="dialout"', rule)
        self.assertIn('MODE="0660"', rule)


if __name__ == "__main__":
    unittest.main()
