"""Contract tests for the multi-platform grasp capability layer (W4)."""

from __future__ import annotations

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
INTERFACES = ROOT / "ros_depends_ws/src/ubrobot_interfaces"
MANIPULATION = ROOT / "ros_depends_ws/src/ubrobot_manipulation"
DESIGN_DOC = ROOT / "docs/plans/2026-07-31-cortex-grasp-capability.md"


class GraspActionInterfaceTest(unittest.TestCase):
    def test_action_fields_and_status_constants(self):
        text = (INTERFACES / "action/GraspObject.action").read_text(
            encoding="utf-8"
        )
        sections = text.strip().split("---")
        self.assertEqual(len(sections), 3)
        goal_fields = sections[0].strip().splitlines()
        self.assertEqual(
            goal_fields, ["string target", "float32 timeout_sec"]
        )
        result = sections[1]
        for constant in (
            "uint8 SUCCEEDED=0",
            "uint8 CANCELLED=1",
            "uint8 TIMED_OUT=2",
            "uint8 REJECTED=3",
            "uint8 FAILED=4",
        ):
            self.assertIn(constant, result)
        self.assertIn("uint8 status", result)
        self.assertIn("string message", result)
        feedback = sections[2].strip().splitlines()
        self.assertEqual(
            feedback,
            [
                "string phase",
                "float32 target_distance_m",
                "float32 progress",
            ],
        )

    def test_interfaces_build_registers_both_actions(self):
        cmake = (INTERFACES / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn('"action/NavigateToObject.action"', cmake)
        self.assertIn('"action/GraspObject.action"', cmake)

    def test_image_builds_manipulation_overlay(self):
        dockerfile = (ROOT / "deploy/emos/Dockerfile").read_text(encoding="utf-8")
        self.assertIn(
            "COPY ros_depends_ws/src/ubrobot_manipulation src/ubrobot_manipulation",
            dockerfile,
        )


class GraspPackageStructureTest(unittest.TestCase):
    def test_package_files_exist(self):
        for relative in (
            "package.xml",
            "setup.py",
            "setup.cfg",
            "resource/ubrobot_manipulation",
            "ubrobot_manipulation/__init__.py",
            "ubrobot_manipulation/policy.py",
            "ubrobot_manipulation/lifecycle.py",
            "test/test_policy.py",
            "test/test_lifecycle.py",
        ):
            self.assertTrue(
                (MANIPULATION / relative).exists(), f"missing {relative}"
            )

    def test_recipe_does_not_expose_grasp_yet(self):
        # Grasp exposure to Cortex is a separate, later task; the planner
        # must not discover it before the capability is deployed.
        recipe = (
            ROOT / "deploy/emos/recipes/cortex_navigation/recipe.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("GraspObject", recipe)
        self.assertNotIn("grasp", recipe.lower())


class GraspDesignDocTest(unittest.TestCase):
    def test_design_doc_covers_platforms_and_exclusion(self):
        doc = DESIGN_DOC.read_text(encoding="utf-8")
        for token in (
            "piper_station",
            "go2_piper",
            "SO101",
            "Mutual exclusion is explicit",
            "navigation_lease_active",
            "MotionAuthorityAdapter",
            "GraspExecutorAdapter",
        ):
            self.assertIn(token, doc)


if __name__ == "__main__":
    unittest.main()
