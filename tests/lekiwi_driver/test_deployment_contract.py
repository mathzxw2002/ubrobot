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
        self.assertIn("stop_signal: SIGINT", compose)
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
        self.assertIn('DeclareLaunchArgument("enable_motor_torque", default_value="false")', launch)

    def test_real_hardware_requires_explicit_acknowledgement(self):
        launch = read_repository_file(BRINGUP_ROOT / "launch" / "lekiwi_driver.launch.py")
        self.assertIn('hardware_mode == "real"', launch)
        self.assertIn('enable_real_hardware != "true"', launch)
        compose = read_repository_file(DEPLOYMENT_ROOT / "compose.hardware.yaml")
        self.assertIn("enable_real_hardware:=true", compose)
        self.assertNotIn("enable_motor_torque:=true", compose)

    def test_motor_torque_requires_a_separate_non_restarting_override(self):
        launch = read_repository_file(BRINGUP_ROOT / "launch" / "lekiwi_driver.launch.py")
        self.assertIn('enable_motor_torque == "true"', launch)
        self.assertIn('hardware_mode != "real"', launch)
        self.assertIn('enable_real_hardware != "true"', launch)

        compose = read_repository_file(DEPLOYMENT_ROOT / "compose.hardware-torque-test.yaml")
        self.assertIn('restart: "no"', compose)
        self.assertIn("enable_motor_torque:=true", compose)

    def test_real_hardware_defaults_to_torque_disabled_and_first_test_speed_limit(self):
        control_xacro = read_repository_file(
            REPOSITORY_ROOT
            / "ros_depends_ws"
            / "src"
            / "lekiwi_description"
            / "ros2_control"
            / "lekiwi_base.ros2_control.xacro"
        )
        top_level_xacro = read_repository_file(
            REPOSITORY_ROOT
            / "ros_depends_ws"
            / "src"
            / "lekiwi_description"
            / "urdf"
            / "lekiwi_base.urdf.xacro"
        )
        hardware_header = read_repository_file(
            HARDWARE_ROOT / "include" / "lekiwi_hardware" / "lekiwi_system_hardware.hpp"
        )
        hardware_source = read_repository_file(
            HARDWARE_ROOT / "src" / "lekiwi_system_hardware.cpp"
        )
        self.assertIn('<xacro:arg name="enable_motor_torque" default="false"/>', top_level_xacro)
        self.assertIn(
            '<param name="enable_motor_torque">'
            "${'true' if enable_motor_torque else 'false'}"
            "</param>",
            control_xacro,
        )
        self.assertIn('<param name="max_raw_velocity">300</param>', control_xacro)
        self.assertIn("bool enable_motor_torque_{false};", hardware_header)
        self.assertIn('boolean_parameter(parameters, "enable_motor_torque")', hardware_source)
        self.assertIn("if (enable_motor_torque_)", hardware_source)
        preflight_write_branch = hardware_source.split(
            "if (!enable_motor_torque_) {", 1
        )[1].split("FeetechBus::RawVelocities", 1)[0]
        self.assertNotIn("write_velocities", preflight_write_branch)
        self.assertIn("return hardware_interface::return_type::OK", preflight_write_branch)

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
            "build:",
            "dockerfile: deploy/emos/Dockerfile",
            "container_name: emos",
            "network_mode: host",
            "restart: always",
            "RMW_IMPLEMENTATION: rmw_fastrtps_cpp",
            "FASTRTPS_DEFAULT_PROFILES_FILE: /etc/fastdds/udp-only.xml",
            "FASTDDS_DEFAULT_PROFILES_FILE: /etc/fastdds/udp-only.xml",
            "${EMOS_DATA_DIR:-/home/china/emos}:/emos",
            "${EMOS_DATA_DIR:-/home/china/emos}:/home/china/emos",
            "../fastdds/udp-only.xml:/etc/fastdds/udp-only.xml:ro",
        ):
            self.assertIn(token, compose)

    def test_emos_image_contains_recipe_runtime_dependencies(self):
        dockerfile = read_repository_file(EMOS_DEPLOYMENT_ROOT / "Dockerfile")
        for token in (
            "ARG EMOS_BASE_IMAGE=ghcr.io/automatika-robotics/emos@sha256:8ee294cffd187328ac3c2776e3389d8d93ad0bc7479e0dac284ae3d095e90f41",
            "libompl16t64",
            "libboost-system1.83.0",
            "libfcl0.7",
            "liboctomap1.9t64",
            "ros-jazzy-depthimage-to-laserscan",
            "ros-jazzy-realsense2-camera",
            "ros-jazzy-realsense2-camera-msgs",
            "ros-jazzy-rtabmap-odom",
            "kompass-core==${KOMPASS_CORE_VERSION}",
        ):
            self.assertIn(token, dockerfile)

    def test_wheel_joints_publish_finite_position_state(self):
        control_xacro = read_repository_file(
            REPOSITORY_ROOT
            / "ros_depends_ws"
            / "src"
            / "lekiwi_description"
            / "ros2_control"
            / "lekiwi_base.ros2_control.xacro"
        )
        hardware_header = read_repository_file(
            HARDWARE_ROOT / "include" / "lekiwi_hardware" / "lekiwi_system_hardware.hpp"
        )
        hardware_source = read_repository_file(
            HARDWARE_ROOT / "src" / "lekiwi_system_hardware.cpp"
        )
        self.assertEqual(control_xacro.count('<state_interface name="position"/>'), 3)
        self.assertIn("position_states_{}", hardware_header)
        self.assertIn("hardware_interface::HW_IF_POSITION", hardware_source)
        self.assertIn("position_states_[index] += velocity_states_[index] * period.seconds()", hardware_source)

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

    def test_cmd_vel_adapter_accepts_best_effort_guard_output(self):
        adapter = read_repository_file(
            BRINGUP_ROOT / "lekiwi_bringup" / "cmd_vel_adapter.py"
        )
        for token in (
            "QoSProfile(",
            "ReliabilityPolicy.BEST_EFFORT",
            "DurabilityPolicy.VOLATILE",
            "HistoryPolicy.KEEP_LAST",
        ):
            self.assertIn(token, adapter)
        subscription = adapter.split("self._subscription =", 1)[1].split(")", 1)[0]
        self.assertIn("velocity_qos", subscription)

    def test_udev_rule_creates_stable_lekiwi_device(self):
        rule = read_repository_file(DEPLOYMENT_ROOT / "99-lekiwi-base.rules")
        for token in ('ATTRS{idVendor}=="1a86"', 'ATTRS{idProduct}=="55d3"'):
            self.assertIn(token, rule)
        self.assertIn('ATTRS{serial}=="5A68011386"', rule)
        self.assertIn('SYMLINK+="lekiwi-base"', rule)
        self.assertIn('GROUP="dialout"', rule)
        self.assertIn('MODE="0660"', rule)

    def test_torque_enable_requires_verified_zero_goal_and_stationary_wheels(self):
        bus_source = read_repository_file(
            HARDWARE_ROOT / "src" / "feetech_bus.cpp"
        )
        # Fire-and-forget sync writes cannot prove the goal registers cleared;
        # torque must only follow acknowledged writes plus read-back, and
        # activation must fail safe if a wheel is already moving.
        self.assertIn("zero_goal_registers_verified()", bus_source)
        self.assertIn("assert_wheels_stationary(150)", bus_source)
        self.assertIn(
            'write_register(id, ft::kGoalVelocityAddress, {0, 0});', bus_source
        )
        self.assertIn(
            'read_register(id, ft::kGoalVelocityAddress, 2)', bus_source
        )

    def test_emos_image_persists_full_stack_supervisor(self):
        dockerfile = read_repository_file(EMOS_DEPLOYMENT_ROOT / "Dockerfile")
        compose = read_repository_file(EMOS_DEPLOYMENT_ROOT / "compose.yaml")
        for token in (
            "COPY ros_depends_ws/src/emos_bringup src/emos_bringup",
            "--install-base /opt/emos_overlay",
            "COPY deploy/emos/start-stack.sh /usr/local/bin/emos-stack.sh",
            "/opt/ubrobot/recipes/vision_depth_follower/recipe.py",
            "_vision_follower.py",
        ):
            self.assertIn(token, dockerfile)
        self.assertIn("/usr/local/bin/emos-stack.sh", compose)

    def test_emos_bringup_launch_covers_the_validated_sensor_chain(self):
        launch = read_repository_file(
            REPOSITORY_ROOT
            / "ros_depends_ws"
            / "src"
            / "emos_bringup"
            / "launch"
            / "vision_depth_bringup.launch.py"
        )
        for token in (
            "realsense2_camera",
            "rgbd_odometry",
            "depthimage_to_laserscan",
            "fix_detection_header",
            "/vision_detections_raw",
            "/vision_detections",
            "camera_depth_frame",
            "camera_depth_link",
            "base_link",
        ):
            self.assertIn(token, launch)
        # camera_depth_link must have exactly one TF parent: the identity
        # alias from camera_depth_frame, not a second static publisher.
        self.assertNotIn("base_to_camera_depth_link_tf", launch)


if __name__ == "__main__":
    unittest.main()
