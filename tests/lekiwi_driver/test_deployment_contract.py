from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENT_ROOT = REPOSITORY_ROOT / "deploy" / "lekiwi-driver"
EMOS_DEPLOYMENT_ROOT = REPOSITORY_ROOT / "deploy" / "emos"
FASTDDS_PROFILE = REPOSITORY_ROOT / "deploy" / "fastdds" / "udp-only.xml"
BRINGUP_ROOT = REPOSITORY_ROOT / "ros_depends_ws" / "src" / "lekiwi_bringup"
HARDWARE_ROOT = REPOSITORY_ROOT / "ros_depends_ws" / "src" / "lekiwi_hardware"


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
        self.assertIn("FASTRTPS_DEFAULT_PROFILES_FILE: /etc/fastdds/udp-only.xml", compose)
        self.assertIn("../fastdds/udp-only.xml:/etc/fastdds/udp-only.xml:ro", compose)
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
        self.assertIn('DeclareLaunchArgument("enable_real_hardware", default_value="false")', launch)

    def test_real_hardware_requires_explicit_acknowledgement(self):
        launch = read_repository_file(BRINGUP_ROOT / "launch" / "lekiwi_driver.launch.py")
        self.assertIn('hardware_mode == "real"', launch)
        self.assertIn('enable_real_hardware.lower() != "true"', launch)
        compose = read_repository_file(DEPLOYMENT_ROOT / "compose.hardware.yaml")
        self.assertIn("enable_real_hardware:=true", compose)

    def test_real_plugin_is_built_into_the_image(self):
        dockerfile = read_repository_file(DEPLOYMENT_ROOT / "Dockerfile")
        self.assertIn("COPY ros_depends_ws/src/lekiwi_hardware src/lekiwi_hardware", dockerfile)
        plugin = read_repository_file(HARDWARE_ROOT / "lekiwi_hardware.xml")
        self.assertIn("lekiwi_hardware/LeKiwiSystemHardware", plugin)

    def test_fastdds_profile_disables_shared_memory_transport(self):
        profile = read_repository_file(FASTDDS_PROFILE)
        self.assertIn("<type>UDPv4</type>", profile)
        self.assertIn("<useBuiltinTransports>false</useBuiltinTransports>", profile)

    def test_emos_compose_persists_the_shared_fastdds_profile(self):
        compose = read_repository_file(EMOS_DEPLOYMENT_ROOT / "compose.yaml")
        for token in (
            "container_name: emos",
            "network_mode: host",
            "restart: always",
            "RMW_IMPLEMENTATION: rmw_fastrtps_cpp",
            "FASTRTPS_DEFAULT_PROFILES_FILE: /etc/fastdds/udp-only.xml",
            "FASTDDS_DEFAULT_PROFILES_FILE: /etc/fastdds/udp-only.xml",
            "${EMOS_DATA_DIR:-/home/china/emos}:/emos",
            "../fastdds/udp-only.xml:/etc/fastdds/udp-only.xml:ro",
        ):
            self.assertIn(token, compose)

    def test_real_hardware_uses_verified_motor_contract(self):
        control_xacro = read_repository_file(
            REPOSITORY_ROOT
            / "ros_depends_ws"
            / "src"
            / "lekiwi_description"
            / "ros2_control"
            / "lekiwi_base.ros2_control.xacro"
        )
        for token in (
            "lekiwi_hardware/LeKiwiSystemHardware",
            "<param name=\"back_motor_id\">8</param>",
            "<param name=\"right_motor_id\">9</param>",
            "<param name=\"left_motor_id\">7</param>",
            "<param name=\"baud_rate\">1000000</param>",
        ):
            self.assertIn(token, control_xacro)

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
