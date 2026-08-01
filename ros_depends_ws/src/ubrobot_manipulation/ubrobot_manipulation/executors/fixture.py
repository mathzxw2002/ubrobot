"""Deterministic, hardware-free grasp executor for local EMOS validation."""

from __future__ import annotations

import threading
import time

from ..lifecycle import ExecutorResult, GraspFeedback
from ..policy import PlatformProfile
from .piper_graspnet import GraspCandidate, phase_progress


class DeterministicGraspExecutor:
    """Adapter-compatible fixture with realistic phases and cancellation."""

    def __init__(self, *, profile: PlatformProfile, phase_delay_sec: float = 0.05):
        if phase_delay_sec <= 0:
            raise ValueError("phase_delay_sec must be positive")
        self._profile = profile
        self._phase_delay_sec = float(phase_delay_sec)
        self._lock = threading.Lock()
        self._thread = None
        self._cancel = None
        self._result = None

    def start(self, target: str, timeout_sec: float, feedback_callback) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._cancel = threading.Event()
            self._result = None
            self._thread = threading.Thread(
                target=self._run,
                args=(target, timeout_sec, feedback_callback, self._cancel),
                name="deterministic-grasp-fixture",
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
                raise RuntimeError("fixture result requested before completion")
            return self._result

    def cancel(self, timeout_sec: float) -> bool:
        with self._lock:
            cancel = self._cancel
            thread = self._thread
        if cancel is None or thread is None:
            return True
        cancel.set()
        thread.join(timeout=max(0.0, float(timeout_sec)))
        return not thread.is_alive()

    def _run(self, target: str, timeout_sec: float, feedback_callback, cancel):
        deadline = time.monotonic() + float(timeout_sec)
        candidate = GraspCandidate(score=0.95, position=(0.30, 0.0, 0.20))
        try:
            for phase in ("approach", "align", "grasp", "retreat"):
                for fraction in (0.0, 0.5, 1.0):
                    if cancel.is_set():
                        self._finish(ExecutorResult(False, "grasp cancelled"))
                        return
                    if time.monotonic() >= deadline:
                        self._finish(ExecutorResult(False, "grasp timed out"))
                        return
                    feedback_callback(
                        GraspFeedback(
                            phase=phase,
                            target_distance_m=0.36 if phase != "retreat" else 0.0,
                            progress=phase_progress(phase, fraction),
                        )
                    )
                    time.sleep(self._phase_delay_sec)
            if not self._profile.workspace.contains(candidate.position):
                self._finish(ExecutorResult(False, "fixture pose outside workspace"))
                return
            self._finish(ExecutorResult(True, f"grasped '{target}'"))
        except Exception as exc:
            self._finish(ExecutorResult(False, f"{type(exc).__name__}: {exc}"))

    def _finish(self, result: ExecutorResult) -> None:
        with self._lock:
            self._result = result
