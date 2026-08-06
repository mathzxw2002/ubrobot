from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
LAUNCH = (
    ROOT
    / "ros_depends_ws"
    / "src"
    / "emos_bringup"
    / "launch"
    / "cortex_navigation_bringup.launch.py"
)
RECIPE = ROOT / "deploy/emos/recipes/vision_depth_follower/recipe.py"


class NavigationBringupContractTest(unittest.TestCase):
    def test_launch_composes_sensors_capability_server_and_guard(self):
        source = LAUNCH.read_text(encoding="utf-8")
        self.assertIn("vision_depth_bringup.launch.py", source)
        # The guarded /cmd_vel chain lives in the launch: grasp server +
        # cmd_vel_guard. The navigate_to_object_server was moved to the
        # recipe container so its /track_vision_target call is in-process.
        self.assertIn('executable="grasp_object_server"', source)
        self.assertIn('executable="cmd_vel_guard"', source)
        self.assertIn("UBROBOT_GRASP_PLATFORM", source)
        self.assertIn("UBROBOT_GRASP_EXECUTOR", source)

    def test_guard_defaults_are_explicit_and_bounded(self):
        source = LAUNCH.read_text(encoding="utf-8")
        for name, default in (
            ("lease_timeout_sec", "0.25"),
            ("raw_command_timeout_sec", "0.25"),
            ("guard_period_sec", "0.05"),
        ):
            self.assertIn(f'"{name}"', source)
            self.assertIn(f'default_value="{default}"', source)

    def test_only_drive_manager_output_targets_the_guard_input(self):
        source = RECIPE.read_text(encoding="utf-8")
        self.assertIn("driver.outputs(", source)
        self.assertIn("robot_command=Topic(", source)
        self.assertIn('name="/navigation/raw_cmd_vel"', source)
        self.assertNotIn("driver.launch_cmd_args", source)

    def test_bringup_declares_navigation_runtime_dependency(self):
        package = (
            ROOT / "ros_depends_ws/src/emos_bringup/package.xml"
        ).read_text(encoding="utf-8")
        self.assertIn("<exec_depend>ubrobot_navigation</exec_depend>", package)


if __name__ == "__main__":
    unittest.main()
