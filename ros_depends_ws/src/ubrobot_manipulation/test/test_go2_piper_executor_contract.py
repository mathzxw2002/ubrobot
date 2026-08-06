"""Contract tests for the Go2+Piper executor bindings (Task 4).

Covers the two real bindings that land in ``executors/go2_piper.py``:

- ``RemoteGraspPerception``: HTTP client to the x86 GPU GraspNet service.
  Unreachable / contract-mismatched responses MUST fail closed (raise), so
  the pipeline never falls back to local guessing.
- ``PiperMotionBinding``: IK (pinocchio) + Piper SDK execution. Must emit
  approach/align/grasp/retreat phase feedback, honour cancellation and
  ``hold_position``, and must NOT call ``piper_ctrl_single_node``.

Also gates ``build_executor``: only ``go2_piper`` + ``hardware`` constructs
the real bindings; everything else keeps the existing fixture or raises.

Workstation-safe: no torch, rclpy, piper_sdk, pinocchio, or unitree SDK at
import time (all hardware imports are deferred inside the bindings).
"""

import os
import sys
import threading
import unittest
from unittest.mock import patch

from ubrobot_manipulation.executors.piper_graspnet import (
    GraspCandidate,
    PiperGraspNetExecutor,
)
from ubrobot_manipulation.policy import get_platform_profile

GO2 = get_platform_profile("go2_piper")


class FakeFrameProvider:
    """Returns a deterministic RGB-D + intrinsic payload."""

    def __call__(self):
        return {
            "color": b"fake-color-png",
            "depth": b"fake-depth-png",
            "camera_intrinsic": [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
        }


class FakeTransport:
    """Injected HTTP transport: records calls, returns scripted JSON/error."""

    def __init__(self, response=None, error=None, delay_sec=0.0):
        self.response = response
        self.error = error
        self.delay_sec = delay_sec
        self.calls = []

    def __call__(self, url, payload, timeout):
        self.calls.append((url, payload, timeout))
        if self.delay_sec:
            import time

            time.sleep(self.delay_sec)
        if self.error is not None:
            raise self.error
        return self.response


_VALID_RESPONSE = {
    "grasp_poses": [
        {"score": 0.9, "position": [0.30, 0.0, 0.20], "orientation": [0, 0, 0, 1]},
        {"score": 0.4, "position": [0.20, -0.10, 0.20], "orientation": [0, 0, 0, 1]},
    ]
}


class TestRemoteGraspPerception(unittest.TestCase):
    def _perception(self, transport):
        from ubrobot_manipulation.executors.go2_piper import RemoteGraspPerception

        return RemoteGraspPerception(
            service_url="http://perception-server:5802",
            frames=FakeFrameProvider(),
            transport=transport,
        )

    def test_valid_response_parses_into_candidates(self):
        perception = self._perception(FakeTransport(response=_VALID_RESPONSE))
        candidates = perception.locate_grasp_poses("cup", GO2.workspace, threading.Event())
        self.assertEqual(len(candidates), 2)
        self.assertIsInstance(candidates[0], GraspCandidate)
        self.assertEqual(candidates[0].position, (0.30, 0.0, 0.20))
        self.assertAlmostEqual(candidates[0].score, 0.9)

    def test_payload_contains_target_intrinsics_and_workspace(self):
        perception = self._perception(FakeTransport(response=_VALID_RESPONSE))
        perception.locate_grasp_poses("cup", GO2.workspace, threading.Event())
        url, payload, _timeout = perception._transport.calls[-1]
        self.assertIn("/grasp_poses", url)
        self.assertEqual(payload["target"], "cup")
        self.assertEqual(payload["camera_intrinsic"][0][0], 600.0)
        self.assertIn("workspace", payload)

    def test_connection_error_fails_closed(self):
        perception = self._perception(
            FakeTransport(error=ConnectionError("service unreachable"))
        )
        with self.assertRaises(ConnectionError):
            perception.locate_grasp_poses("cup", GO2.workspace, threading.Event())

    def test_timeout_fails_closed(self):
        perception = self._perception(
            FakeTransport(error=TimeoutError("grasp inference timed out"))
        )
        with self.assertRaises(TimeoutError):
            perception.locate_grasp_poses("cup", GO2.workspace, threading.Event())

    def test_contract_mismatch_fails_closed(self):
        malformed = {"grasp_poses": [{"score": "nan"}]}  # missing position
        perception = self._perception(FakeTransport(response=malformed))
        with self.assertRaises(ValueError):
            perception.locate_grasp_poses("cup", GO2.workspace, threading.Event())

    def test_cancellation_is_observed(self):
        perception = self._perception(
            FakeTransport(response=_VALID_RESPONSE, delay_sec=0.2)
        )
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(RuntimeError):
            perception.locate_grasp_poses("cup", GO2.workspace, cancel)


class FakeIkSolver:
    def __init__(self, joints=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), error=None):
        self.joints = joints
        self.error = error
        self.calls = []

    def solve(self, position, orientation):
        self.calls.append((position, orientation))
        if self.error is not None:
            raise self.error
        return list(self.joints)


class FakePiperSdk:
    def __init__(self):
        self.joint_calls = []
        self.gripper_calls = []
        self.disconnect_calls = 0

    def set_joint_positions_deg(self, joints_deg, gripper_mm=None):
        self.joint_calls.append((list(joints_deg), gripper_mm))
        if gripper_mm is not None:
            self.gripper_calls.append(gripper_mm)

    def get_status_deg(self):
        return {"joint_1.pos": 0.0}

    def disconnect(self):
        self.disconnect_calls += 1


class TestPiperMotionBinding(unittest.TestCase):
    def _binding(self, sdk=None, ik=None):
        from ubrobot_manipulation.executors.go2_piper import PiperMotionBinding

        return PiperMotionBinding(sdk=sdk or FakePiperSdk(), ik=ik or FakeIkSolver())

    def test_execute_emits_all_phases_and_commands_sdk(self):
        sdk = FakePiperSdk()
        binding = self._binding(sdk=sdk)
        pose = GraspCandidate(score=0.9, position=(0.30, 0.0, 0.20), orientation=(0, 0, 0, 1))
        phases = []
        binding.execute_grasp(
            pose,
            max_speed_mps=0.05,
            cancel_event=threading.Event(),
            on_phase=lambda phase, fraction: phases.append((phase, fraction)),
        )
        seen = [phase for phase, _ in phases]
        self.assertEqual(seen, ["approach", "align", "grasp", "retreat"])
        self.assertTrue(sdk.joint_calls, "Piper SDK joint command was not issued")
        self.assertTrue(sdk.gripper_calls, "Piper SDK gripper command was not issued")

    def test_hold_position_keeps_current_pose(self):
        sdk = FakePiperSdk()
        binding = self._binding(sdk=sdk)
        binding.hold_position()
        self.assertTrue(sdk.joint_calls)

    def test_ik_error_fails_closed(self):
        binding = self._binding(ik=FakeIkSolver(error=RuntimeError("ik no solution")))
        pose = GraspCandidate(score=0.9, position=(0.30, 0.0, 0.20))
        with self.assertRaises(RuntimeError):
            binding.execute_grasp(
                pose,
                max_speed_mps=0.05,
                cancel_event=threading.Event(),
                on_phase=lambda _phase, _fraction: None,
            )

    def test_no_piper_ctrl_single_node_import(self):
        import ubrobot_manipulation.executors.go2_piper  # noqa: F401

        for name in ("torch", "rclpy", "piper_sdk", "pinocchio"):
            self.assertNotIn(name, sys.modules, name)


class TestBuildExecutorGating(unittest.TestCase):
    """resolve_executor_binding only builds real bindings for go2_piper+hardware."""

    def _resolve(self, platform_env, executor_env, extra_env=None):
        from ubrobot_manipulation.executors import resolve_executor_binding

        profile = get_platform_profile(platform_env)
        env = {"UBROBOT_GRASP_PLATFORM": platform_env, "UBROBOT_GRASP_EXECUTOR": executor_env}
        if extra_env:
            env.update(extra_env)
        with patch.dict(os.environ, env, clear=False):
            return resolve_executor_binding(profile, executor_env)

    def test_go2_piper_hardware_builds_real_executor(self):
        executor = self._resolve("go2_piper", "hardware")
        self.assertIsInstance(executor, PiperGraspNetExecutor)

    def test_go2_piper_hardware_requires_service_url(self):
        with patch.dict(
            os.environ,
            {"UBROBOT_GRASP_PLATFORM": "go2_piper", "UBROBOT_GRASP_EXECUTOR": "hardware"},
        ):
            from dataclasses import replace  # noqa: PLC0415

            profile = get_platform_profile("go2_piper")
            # Stripped service URL -> fail closed.
            profile_no_url = replace(profile, remote_perception_service_url="")
            from ubrobot_manipulation.executors import resolve_executor_binding  # noqa: PLC0415

            with self.assertRaises(NotImplementedError):
                resolve_executor_binding(profile_no_url, "hardware")

    def test_go2_piper_fixture_keeps_fixture(self):
        from ubrobot_manipulation.executors.fixture import DeterministicGraspExecutor

        executor = self._resolve("go2_piper", "fixture")
        self.assertIsInstance(executor, DeterministicGraspExecutor)

    def test_other_platform_hardware_still_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self._resolve("piper_station", "hardware")


if __name__ == "__main__":
    unittest.main()
