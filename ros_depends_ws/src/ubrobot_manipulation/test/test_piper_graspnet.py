import threading
import time
import unittest

from ubrobot_manipulation.executors.piper_graspnet import (
    GRASP_PHASES,
    GraspCandidate,
    PiperGraspNetExecutor,
    phase_progress,
    select_grasp_pose,
)
from ubrobot_manipulation.policy import get_platform_profile


WORKSPACE = get_platform_profile("piper_station").workspace
IN_WORKSPACE = (0.30, 0.0, 0.20)
OUT_WORKSPACE = (5.0, 0.0, 0.20)


class SelectGraspPoseTest(unittest.TestCase):
    def test_highest_score_reachable_wins(self):
        candidates = [
            GraspCandidate(score=0.5, position=IN_WORKSPACE),
            GraspCandidate(score=0.9, position=IN_WORKSPACE),
            GraspCandidate(score=0.7, position=IN_WORKSPACE),
        ]
        self.assertIs(
            select_grasp_pose(candidates, WORKSPACE), candidates[1]
        )

    def test_out_of_workspace_candidates_are_ignored(self):
        candidates = [
            GraspCandidate(score=0.99, position=OUT_WORKSPACE),
            GraspCandidate(score=0.1, position=IN_WORKSPACE),
        ]
        self.assertIs(
            select_grasp_pose(candidates, WORKSPACE), candidates[1]
        )

    def test_non_finite_scores_are_ignored(self):
        candidates = [
            GraspCandidate(score=float("nan"), position=IN_WORKSPACE),
            GraspCandidate(score=0.2, position=IN_WORKSPACE),
        ]
        self.assertIs(
            select_grasp_pose(candidates, WORKSPACE), candidates[1]
        )

    def test_none_reachable_returns_none(self):
        candidates = [GraspCandidate(score=0.9, position=OUT_WORKSPACE)]
        self.assertIsNone(select_grasp_pose(candidates, WORKSPACE))
        self.assertIsNone(select_grasp_pose([], WORKSPACE))


class PhaseProgressTest(unittest.TestCase):
    def test_windows_cover_zero_to_one_in_order(self):
        self.assertEqual(phase_progress("approach", 0.0), 0.0)
        self.assertEqual(phase_progress("retreat", 1.0), 1.0)
        previous = -1.0
        for phase in GRASP_PHASES:
            for fraction in (0.0, 1.0):
                value = phase_progress(phase, fraction)
                self.assertGreaterEqual(value, previous)
                previous = value

    def test_fraction_is_clamped(self):
        self.assertEqual(phase_progress("approach", 2.0), 0.4)
        self.assertEqual(phase_progress("align", -1.0), 0.4)

    def test_unknown_phase_raises(self):
        with self.assertRaises(ValueError):
            phase_progress("fly", 0.5)


class FakePerception:
    def __init__(self, candidates, delay_sec=0.0, error=None):
        self.candidates = candidates
        self.delay_sec = delay_sec
        self.error = error
        self.calls = []

    def locate_grasp_poses(self, target, workspace, cancel_event):
        self.calls.append(target)
        if self.delay_sec:
            time.sleep(self.delay_sec)
        if cancel_event.is_set():
            raise RuntimeError("perception should observe cancellation")
        if self.error is not None:
            raise self.error
        return self.candidates


class FakeMotion:
    def __init__(self, phases=("align", "grasp", "retreat"), block_sec=0.0):
        self.phases = phases
        self.block_sec = block_sec
        self.executed = []
        self.hold_calls = 0
        self.max_speeds = []

    def execute_grasp(self, pose, *, max_speed_mps, cancel_event, on_phase):
        self.executed.append(pose)
        self.max_speeds.append(max_speed_mps)
        for index, phase in enumerate(self.phases):
            if cancel_event.is_set():
                return
            on_phase(phase, 1.0)
        if self.block_sec:
            deadline = time.monotonic() + self.block_sec
            while time.monotonic() < deadline and not cancel_event.is_set():
                time.sleep(0.01)

    def hold_position(self):
        self.hold_calls += 1


def make_executor(perception, motion):
    return PiperGraspNetExecutor(
        profile=get_platform_profile("piper_station"),
        perception=perception,
        motion=motion,
    )


def run_until_done(executor, timeout_sec=5.0):
    deadline = time.monotonic() + timeout_sec
    while not executor.is_done() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not executor.is_done():
        raise TimeoutError("executor never finished")
    return executor.result()


class PiperGraspNetExecutorTest(unittest.TestCase):
    def test_success_path_emits_phases_in_order(self):
        perception = FakePerception(
            [GraspCandidate(score=0.9, position=IN_WORKSPACE)]
        )
        motion = FakeMotion()
        executor = make_executor(perception, motion)
        feedback = []

        started = executor.start("cup", 30.0, feedback.append)
        self.assertTrue(started)
        result = run_until_done(executor)

        self.assertTrue(result.success)
        self.assertIn("cup", result.message)
        self.assertEqual(perception.calls, ["cup"])
        self.assertEqual(len(motion.executed), 1)
        self.assertEqual(
            motion.max_speeds,
            [get_platform_profile("piper_station").max_approach_speed_mps],
        )
        phases = [item.phase for item in feedback]
        self.assertEqual(phases[0], "approach")
        self.assertEqual(phases[-1], "retreat")
        progress = [item.progress for item in feedback]
        self.assertEqual(progress, sorted(progress))
        self.assertEqual(progress[-1], 1.0)

    def test_no_reachable_pose_fails_before_motion(self):
        perception = FakePerception(
            [GraspCandidate(score=0.9, position=OUT_WORKSPACE)]
        )
        motion = FakeMotion()
        executor = make_executor(perception, motion)

        executor.start("cup", 30.0, lambda _feedback: None)
        result = run_until_done(executor)

        self.assertFalse(result.success)
        self.assertIn("no reachable grasp pose", result.message)
        self.assertEqual(motion.executed, [])

    def test_perception_error_fails_closed(self):
        perception = FakePerception([], error=RuntimeError("camera timeout"))
        motion = FakeMotion()
        executor = make_executor(perception, motion)

        executor.start("cup", 30.0, lambda _feedback: None)
        result = run_until_done(executor)

        self.assertFalse(result.success)
        self.assertIn("camera timeout", result.message)
        self.assertEqual(motion.executed, [])

    def test_cancel_stops_motion_and_holds(self):
        perception = FakePerception(
            [GraspCandidate(score=0.9, position=IN_WORKSPACE)]
        )
        motion = FakeMotion(block_sec=30.0)
        executor = make_executor(perception, motion)

        executor.start("cup", 60.0, lambda _feedback: None)
        time.sleep(0.2)
        acknowledged = executor.cancel(timeout_sec=2.0)

        self.assertTrue(acknowledged)
        self.assertEqual(motion.hold_calls, 1)
        result = run_until_done(executor)
        self.assertFalse(result.success)
        self.assertIn("cancelled", result.message)

    def test_second_start_is_refused_while_running(self):
        perception = FakePerception(
            [GraspCandidate(score=0.9, position=IN_WORKSPACE)]
        )
        motion = FakeMotion(block_sec=5.0)
        executor = make_executor(perception, motion)

        self.assertTrue(executor.start("cup", 60.0, lambda _feedback: None))
        self.assertFalse(executor.start("bottle", 60.0, lambda _feedback: None))
        executor.cancel(timeout_sec=2.0)

    def test_result_before_completion_raises(self):
        executor = make_executor(FakePerception([]), FakeMotion())
        with self.assertRaises(RuntimeError):
            executor.result()


if __name__ == "__main__":
    unittest.main()
