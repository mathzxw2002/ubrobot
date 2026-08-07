"""GraspExecutorAdapter binding for Piper arms using GraspNet (offline draft).

The pipeline, state machine, and pose-selection logic are complete and
unit-tested. Perception and motion are injected interfaces; their concrete
bindings land with the on-machine executor milestone:

- perception: arm RGB-D capture + target segmentation + GraspNet inference
  (existing `src/service/reasoning/grasp_plan.py::RobotArmMotionPlan`,
  which produces 6D poses but has no execution path — `optimize_traj` is a
  TODO upstream);
- motion: pinocchio IK against `assets/urdf/piper_description.urdf` (on the
  emos side), then joint/gripper execution via a `PiperCommandTransport`
  that publishes `/piper/joint_cmd` to the go2-piper-driver hardware
  container (see `executors/go2_piper.py`). The emos side never touches the
  Piper SDK or CAN directly.

No torch, ROS, or SDK imports happen at module level so the draft stays
importable (and testable) on development workstations.
"""

from dataclasses import dataclass
import math
import threading
import time
from typing import Callable, Optional, Protocol

from ..lifecycle import ExecutorResult, GraspFeedback
from ..policy import PlatformProfile


# Canonical executor phases with their normalized progress windows.
GRASP_PHASES = ("approach", "align", "grasp", "retreat")
_PHASE_WINDOWS = {
    "approach": (0.0, 0.4),
    "align": (0.4, 0.6),
    "grasp": (0.6, 0.8),
    "retreat": (0.8, 1.0),
}


def phase_progress(phase: str, fraction: float) -> float:
    """Map (phase, intra-phase fraction) to normalized progress in [0, 1]."""
    try:
        low, high = _PHASE_WINDOWS[phase]
    except KeyError as exc:
        raise ValueError(f"unknown grasp phase: {phase!r}") from exc
    fraction = min(1.0, max(0.0, float(fraction)))
    return low + (high - low) * fraction


@dataclass(frozen=True)
class GraspCandidate:
    """One 6D grasp candidate in the platform grasp frame."""

    score: float
    position: tuple[float, float, float]
    # Orientation is carried opaque in the draft (quaternion or rotation
    # matrix decided by the motion binding).
    orientation: tuple = ()


def select_grasp_pose(
    candidates, workspace
) -> Optional[GraspCandidate]:
    """Pick the best reachable candidate; None when none are reachable.

    This is the adapter-side workspace enforcement — poses are checked here
    even though perception was already given the workspace, because planner
    or perception output is never trusted blindly.
    """
    reachable = [
        candidate
        for candidate in candidates
        if math.isfinite(candidate.score)
        and workspace.contains(candidate.position)
    ]
    if not reachable:
        return None
    return max(reachable, key=lambda candidate: candidate.score)


class PerceptionInterface(Protocol):
    """Target localization + grasp pose generation (GraspNet binding)."""

    def locate_grasp_poses(
        self,
        target: str,
        workspace,
        cancel_event: threading.Event,
    ) -> list[GraspCandidate]: ...


class MotionInterface(Protocol):
    """IK + arm/gripper execution binding (pinocchio + PiperSDK)."""

    def execute_grasp(
        self,
        pose: GraspCandidate,
        *,
        max_speed_mps: float,
        cancel_event: threading.Event,
        on_phase: Callable[[str, float], None],
    ) -> None: ...

    def hold_position(self) -> None:
        """Stop motion immediately and hold the current pose."""


class PiperGraspNetExecutor:
    """Owns one grasp execution at a time behind the lifecycle adapter."""

    def __init__(
        self,
        *,
        profile: PlatformProfile,
        perception: PerceptionInterface,
        motion: MotionInterface,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._profile = profile
        self._perception = perception
        self._motion = motion
        self._clock = clock
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._cancel_event: Optional[threading.Event] = None
        self._result: Optional[ExecutorResult] = None

    # ---------------------------------------------- GraspExecutorAdapter

    def start(self, target: str, timeout_sec: float, feedback_callback) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._result = None
            self._cancel_event = threading.Event()
            self._thread = threading.Thread(
                target=self._run,
                args=(target, timeout_sec, feedback_callback, self._cancel_event),
                name="piper-graspnet-executor",
                daemon=True,
            )
            self._thread.start()
            return True

    def is_done(self) -> bool:
        with self._lock:
            return self._result is not None

    def result(self) -> ExecutorResult:
        with self._lock:
            if self._result is None:
                raise RuntimeError("executor result requested before completion")
            return self._result

    def cancel(self, timeout_sec: float) -> bool:
        with self._lock:
            cancel_event = self._cancel_event
            thread = self._thread
        if cancel_event is None or thread is None:
            return True
        cancel_event.set()
        thread.join(timeout=max(0.0, timeout_sec))
        try:
            self._motion.hold_position()
        except Exception:
            pass
        return not thread.is_alive()

    # ---------------------------------------------------------- internals

    def _finish(self, result: ExecutorResult) -> None:
        with self._lock:
            self._result = result

    def _run(self, target, timeout_sec, feedback_callback, cancel_event) -> None:
        deadline = self._clock() + timeout_sec

        def emit(phase: str, fraction: float, distance_m: float) -> None:
            feedback_callback(
                GraspFeedback(
                    phase=phase,
                    target_distance_m=distance_m,
                    progress=phase_progress(phase, fraction),
                )
            )

        try:
            emit("approach", 0.0, math.nan)
            candidates = self._perception.locate_grasp_poses(
                target, self._profile.workspace, cancel_event
            )
            if cancel_event.is_set():
                self._finish(ExecutorResult(False, "grasp cancelled"))
                return

            pose = select_grasp_pose(candidates, self._profile.workspace)
            if pose is None:
                self._finish(
                    ExecutorResult(
                        False,
                        f"no reachable grasp pose for '{target}' within the "
                        f"{self._profile.name} workspace",
                    )
                )
                return

            distance = math.sqrt(sum(axis * axis for axis in pose.position))
            if self._clock() >= deadline:
                self._finish(
                    ExecutorResult(False, "grasp timed out before motion")
                )
                return

            def on_phase(phase: str, fraction: float) -> None:
                emit(phase, fraction, distance)

            self._motion.execute_grasp(
                pose,
                max_speed_mps=self._profile.max_approach_speed_mps,
                cancel_event=cancel_event,
                on_phase=on_phase,
            )
            if cancel_event.is_set():
                self._finish(ExecutorResult(False, "grasp cancelled"))
                return
            emit("retreat", 1.0, 0.0)
            self._finish(ExecutorResult(True, f"grasped '{target}'"))
        except Exception as exc:
            self._finish(
                ExecutorResult(False, f"{type(exc).__name__}: {exc}")
            )
