"""Pure lifecycle coordination for the semantic grasp Action.

Mirrors the navigation downstream-goal coordinator: one goal at a time,
deterministic cancellation/timeout semantics, and fail-closed behavior
around shared motion authority. ROS goal handles and platform executors
sit behind small adapter protocols so every failure path is testable
without ROS or hardware.
"""

from dataclasses import dataclass
from enum import IntEnum
import threading
import time
from typing import Callable, Protocol

from .policy import (
    PlatformProfile,
    ValidatedGraspGoal,
    grasp_may_start,
    validate_goal,
)


class GraspStatus(IntEnum):
    SUCCEEDED = 0
    CANCELLED = 1
    TIMED_OUT = 2
    REJECTED = 3
    FAILED = 4


@dataclass(frozen=True)
class GraspFeedback:
    phase: str
    target_distance_m: float
    progress: float


@dataclass(frozen=True)
class ExecutorResult:
    success: bool
    message: str


@dataclass(frozen=True)
class GraspOutcome:
    status: GraspStatus
    message: str


@dataclass(frozen=True)
class GoalReservation:
    goal: ValidatedGraspGoal
    token: object


class GoalBusyError(RuntimeError):
    pass


class OuterGoalAdapter(Protocol):
    def is_cancel_requested(self) -> bool: ...

    def publish_feedback(self, feedback: GraspFeedback) -> None: ...


class GraspExecutorAdapter(Protocol):
    """Platform binding: Piper workstation, Go2+Piper, future SO101."""

    def start(
        self,
        target: str,
        timeout_sec: float,
        feedback_callback: Callable[[GraspFeedback], None],
    ) -> bool: ...

    def is_done(self) -> bool: ...

    def result(self) -> ExecutorResult: ...

    def cancel(self, timeout_sec: float) -> bool: ...


class MotionAuthorityAdapter(Protocol):
    """Observes shared motion authority for grasp/navigation exclusion."""

    def navigation_lease_active(self) -> bool: ...

    def base_is_stationary(self) -> bool: ...


class GraspLifecycleCoordinator:
    """Own exactly one grasp goal and fail closed around motion authority."""

    def __init__(
        self,
        *,
        profile: PlatformProfile,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        poll_period_sec: float = 0.05,
        cancellation_timeout_sec: float = 2.0,
    ):
        self._profile = profile
        self._clock = clock
        self._sleep = sleep
        self._poll_period_sec = poll_period_sec
        self._cancellation_timeout_sec = cancellation_timeout_sec
        self._reservation_lock = threading.Lock()
        self._active_token = None

    def reserve(self, *, target: str, timeout_sec: float) -> GoalReservation:
        """Validate and atomically reserve the single outer-goal slot."""
        goal = validate_goal(target, timeout_sec)
        token = object()
        with self._reservation_lock:
            if self._active_token is not None:
                raise GoalBusyError("another grasp goal is already active")
            self._active_token = token
        return GoalReservation(goal=goal, token=token)

    def abandon(self, reservation: GoalReservation) -> None:
        """Release a reservation that cannot reach its execute callback."""
        self._release(reservation.token)

    def run(
        self,
        *,
        target: str,
        timeout_sec: float,
        outer: OuterGoalAdapter,
        executor: GraspExecutorAdapter,
        authority: MotionAuthorityAdapter,
    ) -> GraspOutcome:
        """Convenience entry point used by deterministic tests and adapters."""
        try:
            reservation = self.reserve(target=target, timeout_sec=timeout_sec)
        except (ValueError, GoalBusyError) as exc:
            return GraspOutcome(GraspStatus.REJECTED, str(exc))
        return self.execute(
            reservation=reservation,
            outer=outer,
            executor=executor,
            authority=authority,
        )

    def execute(
        self,
        *,
        reservation: GoalReservation,
        outer: OuterGoalAdapter,
        executor: GraspExecutorAdapter,
        authority: MotionAuthorityAdapter,
    ) -> GraspOutcome:
        """Execute one reserved goal, always releasing the slot on return."""
        if not self._owns(reservation.token):
            return GraspOutcome(
                GraspStatus.REJECTED,
                "grasp goal reservation is no longer active",
            )

        executor_started = False
        started_at = self._clock()
        try:
            if not grasp_may_start(
                navigation_lease_active=authority.navigation_lease_active(),
                base_stationary=authority.base_is_stationary(),
                profile=self._profile,
            ):
                return GraspOutcome(
                    GraspStatus.REJECTED,
                    "motion authority conflict: navigation lease active or "
                    "base not stationary",
                )

            executor_started = executor.start(
                reservation.goal.target,
                reservation.goal.timeout_sec,
                outer.publish_feedback,
            )
            if not executor_started:
                return GraspOutcome(
                    GraspStatus.FAILED,
                    "grasp executor rejected the goal",
                )

            while True:
                if authority.navigation_lease_active():
                    # Fail safe: navigation acquired motion authority
                    # mid-grasp; stop the arm rather than share it.
                    executor.cancel(self._cancellation_timeout_sec)
                    return GraspOutcome(
                        GraspStatus.FAILED,
                        "navigation lease appeared during grasp; arm stopped",
                    )

                if outer.is_cancel_requested():
                    acknowledged = executor.cancel(
                        self._cancellation_timeout_sec
                    )
                    message = (
                        "grasp cancelled"
                        if acknowledged
                        else "grasp cancelled; executor acknowledgement timed out"
                    )
                    return GraspOutcome(GraspStatus.CANCELLED, message)

                if self._clock() - started_at >= reservation.goal.timeout_sec:
                    acknowledged = executor.cancel(
                        self._cancellation_timeout_sec
                    )
                    message = (
                        "grasp timed out"
                        if acknowledged
                        else "grasp timed out; executor acknowledgement timed out"
                    )
                    return GraspOutcome(GraspStatus.TIMED_OUT, message)

                if executor.is_done():
                    result = executor.result()
                    status = (
                        GraspStatus.SUCCEEDED
                        if result.success
                        else GraspStatus.FAILED
                    )
                    return GraspOutcome(status, result.message)

                self._sleep(self._poll_period_sec)
        except Exception as exc:
            if executor_started:
                try:
                    executor.cancel(self._cancellation_timeout_sec)
                except Exception:
                    pass
            return GraspOutcome(
                GraspStatus.FAILED,
                f"grasp execution failed: {exc}",
            )
        finally:
            self._release(reservation.token)

    def _owns(self, token: object) -> bool:
        with self._reservation_lock:
            return self._active_token is token

    def _release(self, token: object) -> None:
        with self._reservation_lock:
            if self._active_token is token:
                self._active_token = None
