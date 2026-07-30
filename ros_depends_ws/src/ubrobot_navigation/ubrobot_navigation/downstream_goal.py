"""Pure lifecycle coordination for the semantic navigation Action."""

from dataclasses import dataclass
from enum import IntEnum
import threading
import time
from typing import Callable, Protocol

from .policy import ValidatedGoal, validate_goal


class NavigationStatus(IntEnum):
    SUCCEEDED = 0
    CANCELLED = 1
    TIMED_OUT = 2
    REJECTED = 3
    FAILED = 4


@dataclass(frozen=True)
class NavigationFeedback:
    phase: str
    distance_error: float
    orientation_error: float


@dataclass(frozen=True)
class DownstreamFeedback:
    distance_error: float
    orientation_error: float


@dataclass(frozen=True)
class DownstreamResult:
    success: bool
    message: str


@dataclass(frozen=True)
class NavigationOutcome:
    status: NavigationStatus
    message: str


@dataclass(frozen=True)
class GoalReservation:
    goal: ValidatedGoal
    token: object


class GoalBusyError(RuntimeError):
    pass


class OuterGoalAdapter(Protocol):
    def is_cancel_requested(self) -> bool: ...

    def publish_feedback(self, feedback: NavigationFeedback) -> None: ...


class DownstreamGoalAdapter(Protocol):
    def start(
        self,
        target: str,
        timeout_sec: float,
        feedback_callback: Callable[[DownstreamFeedback], None],
    ) -> bool: ...

    def is_done(self) -> bool: ...

    def result(self) -> DownstreamResult: ...

    def cancel(self, timeout_sec: float) -> bool: ...


class LeaseAdapter(Protocol):
    def acquire(self) -> str: ...

    def revoke(self) -> None: ...


class NavigationLifecycleCoordinator:
    """Own exactly one semantic goal and fail closed around command authority."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        poll_period_sec: float = 0.05,
        cancellation_timeout_sec: float = 2.0,
    ):
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
                raise GoalBusyError("another navigation goal is already active")
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
        downstream: DownstreamGoalAdapter,
        lease: LeaseAdapter,
    ) -> NavigationOutcome:
        """Convenience entry point used by deterministic tests and simple adapters."""
        try:
            reservation = self.reserve(target=target, timeout_sec=timeout_sec)
        except (ValueError, GoalBusyError) as exc:
            return NavigationOutcome(NavigationStatus.REJECTED, str(exc))
        return self.execute(
            reservation=reservation,
            outer=outer,
            downstream=downstream,
            lease=lease,
        )

    def execute(
        self,
        *,
        reservation: GoalReservation,
        outer: OuterGoalAdapter,
        downstream: DownstreamGoalAdapter,
        lease: LeaseAdapter,
    ) -> NavigationOutcome:
        """Execute one reserved goal, always revoking authority before return."""
        if not self._owns(reservation.token):
            return NavigationOutcome(
                NavigationStatus.REJECTED,
                "navigation goal reservation is no longer active",
            )

        lease_acquired = False
        downstream_started = False
        started_at = self._clock()
        try:
            lease.acquire()
            lease_acquired = True
            downstream_started = downstream.start(
                reservation.goal.target,
                reservation.goal.timeout_sec,
                lambda feedback: outer.publish_feedback(
                    NavigationFeedback(
                        phase="tracking",
                        distance_error=feedback.distance_error,
                        orientation_error=feedback.orientation_error,
                    )
                ),
            )
            if not downstream_started:
                return NavigationOutcome(
                    NavigationStatus.FAILED,
                    "downstream navigation goal was rejected",
                )

            while True:
                if outer.is_cancel_requested():
                    acknowledged = downstream.cancel(
                        self._cancellation_timeout_sec
                    )
                    message = (
                        "navigation cancelled"
                        if acknowledged
                        else "navigation cancelled; downstream acknowledgement timed out"
                    )
                    return NavigationOutcome(NavigationStatus.CANCELLED, message)

                if self._clock() - started_at >= reservation.goal.timeout_sec:
                    acknowledged = downstream.cancel(
                        self._cancellation_timeout_sec
                    )
                    message = (
                        "navigation timed out"
                        if acknowledged
                        else "navigation timed out; downstream acknowledgement timed out"
                    )
                    return NavigationOutcome(NavigationStatus.TIMED_OUT, message)

                if downstream.is_done():
                    result = downstream.result()
                    status = (
                        NavigationStatus.SUCCEEDED
                        if result.success
                        else NavigationStatus.FAILED
                    )
                    return NavigationOutcome(status, result.message)

                self._sleep(self._poll_period_sec)
        except Exception as exc:
            if downstream_started:
                try:
                    downstream.cancel(self._cancellation_timeout_sec)
                except Exception:
                    pass
            return NavigationOutcome(
                NavigationStatus.FAILED,
                f"navigation execution failed: {exc}",
            )
        finally:
            if lease_acquired:
                lease.revoke()
            self._release(reservation.token)

    def _owns(self, token: object) -> bool:
        with self._reservation_lock:
            return self._active_token is token

    def _release(self, token: object) -> None:
        with self._reservation_lock:
            if self._active_token is token:
                self._active_token = None
