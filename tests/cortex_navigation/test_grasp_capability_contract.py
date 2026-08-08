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
            "ubrobot_manipulation/authority.py",
            "ubrobot_manipulation/grasp_object_server.py",
            "ubrobot_manipulation/executors/__init__.py",
            "ubrobot_manipulation/executors/fixture.py",
            "ubrobot_manipulation/executors/piper_graspnet.py",
            "test/test_policy.py",
            "test/test_lifecycle.py",
            "test/test_authority.py",
            "test/test_piper_graspnet.py",
        ):
            self.assertTrue(
                (MANIPULATION / relative).exists(), f"missing {relative}"
            )

    def test_recipe_grasp_exposure_is_gated_off_by_default(self):
        # The grasp capability server ships separately; the planner must not
        # discover the tool until CORTEX_ENABLE_GRASP is explicitly enabled.
        recipe = (
            ROOT / "deploy/emos/recipes/cortex_navigation/recipe.py"
        ).read_text(encoding="utf-8")
        self.assertIn("CORTEX_ENABLE_GRASP", recipe)
        self.assertIn('env.get(GRASP_ENABLE_ENV, "false")', recipe)
        # Gate is double-conditioned: env flag AND the GraspObject action
        # type must exist (pre-grasp ubrobot_interfaces builds import it
        # optionally as None, keeping the tool hidden).
        self.assertIn(
            "if grasp_exposure_enabled(os.environ) and GraspObject is not None:",
            recipe,
        )


class GraspServerSkeletonTest(unittest.TestCase):
    def test_server_wires_action_authority_and_platform(self):
        server = (MANIPULATION / "ubrobot_manipulation/grasp_object_server.py").read_text(
            encoding="utf-8"
        )
        for token in (
            '"/ubrobot/manipulation/grasp_object"',
            '"/navigation/command_lease"',
            '"/cmd_vel"',
            "UBROBOT_GRASP_PLATFORM",
            "GraspLifecycleCoordinator",
            "AuthorityTracker",
            "CancelResponse.ACCEPT",
            "GoalResponse.REJECT",
            "build_executor",
        ):
            self.assertIn(token, server)

    def test_server_fails_fast_without_platform_or_executor(self):
        server = (MANIPULATION / "ubrobot_manipulation/grasp_object_server.py").read_text(
            encoding="utf-8"
        )
        resolver = (
            MANIPULATION / "ubrobot_manipulation/executors/__init__.py"
        ).read_text(encoding="utf-8")
        # Missing/unknown platform aborts startup (RuntimeError in the server);
        # missing executor binding fails the goal fast (NotImplementedError in
        # the pure resolver) instead of hanging.
        self.assertIn("raise RuntimeError", server)
        self.assertIn("raise NotImplementedError", resolver)
        # The server delegates the platform/env decision to the pure resolver.
        self.assertIn("resolve_executor_binding", server)

    def test_authority_tracker_is_fail_closed(self):
        # The implementation moved to ubrobot_contracts (refactor Task 1); the
        # ROS package re-exports it. Verify the single source of truth.
        contracts = (
            ROOT / "src/ubrobot_contracts/motion_authority.py"
        ).read_text(encoding="utf-8")
        authority = (MANIPULATION / "ubrobot_manipulation/authority.py").read_text(
            encoding="utf-8"
        )
        # No ROS imports keep it unit-testable; no evidence means no grasp.
        self.assertNotIn("import rclpy", authority)
        self.assertNotIn("import rclpy", contracts)
        self.assertIn("if not self._cmd_vel_samples:", contracts)
        self.assertIn("return False", contracts)
        # The ROS module must keep the historical import path working.
        self.assertIn("from ubrobot_contracts.motion_authority import", authority)

    def test_console_script_registered(self):
        setup_py = (MANIPULATION / "setup.py").read_text(encoding="utf-8")
        self.assertIn(
            "grasp_object_server = ubrobot_manipulation.grasp_object_server:main",
            setup_py,
        )


class PiperExecutorDraftTest(unittest.TestCase):
    def test_executor_draft_is_ros_and_torch_free(self):
        source = (
            MANIPULATION / "ubrobot_manipulation/executors/piper_graspnet.py"
        ).read_text(encoding="utf-8")
        # Module must stay importable on workstations without ROS/torch/SDK
        # (docstring path references to the SDK interface are fine).
        for forbidden in (
            "import rclpy",
            "import torch",
            "import piper_sdk",
            "from piper_sdk",
        ):
            self.assertNotIn(forbidden, source)
        for token in (
            "select_grasp_pose",
            "phase_progress",
            "PerceptionInterface",
            "MotionInterface",
            "hold_position",
            "max_speed_mps=self._profile.max_approach_speed_mps",
        ):
            self.assertIn(token, source)

    def test_executor_enforces_workspace_and_speed_from_profile(self):
        # The adapter re-checks perception output against the profile
        # workspace and caps motion speed from the same profile.
        source = (
            MANIPULATION / "ubrobot_manipulation/executors/piper_graspnet.py"
        ).read_text(encoding="utf-8")
        self.assertIn("workspace.contains(candidate.position)", source)
        self.assertIn("no reachable grasp pose", source)


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
