from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
ACTION_PATH = (
    ROOT
    / "ros_depends_ws"
    / "src"
    / "ubrobot_interfaces"
    / "action"
    / "NavigateToObject.action"
)

EXPECTED_ACTION = """string target
float32 timeout_sec
---
uint8 SUCCEEDED=0
uint8 CANCELLED=1
uint8 TIMED_OUT=2
uint8 REJECTED=3
uint8 FAILED=4
uint8 status
string message
---
string phase
float32 distance_error
float32 orientation_error
"""


class NavigationInterfaceContractTest(unittest.TestCase):
    def test_navigate_to_object_action_is_stable(self):
        self.assertEqual(ACTION_PATH.read_text(encoding="utf-8"), EXPECTED_ACTION)

    def test_interface_package_is_built_in_emos_overlay(self):
        dockerfile = (ROOT / "deploy/emos/Dockerfile").read_text(encoding="utf-8")
        copy = "COPY ros_depends_ws/src/ubrobot_interfaces src/ubrobot_interfaces"
        build = "colcon build --merge-install --install-base /opt/emos_overlay"
        self.assertIn(copy, dockerfile)
        self.assertLess(dockerfile.index(copy), dockerfile.index(build))


if __name__ == "__main__":
    unittest.main()
