"""Deterministic safety policy shared by navigation ROS adapters."""

from dataclasses import dataclass
import math


MAX_TARGET_LENGTH = 128
MIN_TIMEOUT_SEC = 1.0
MAX_TIMEOUT_SEC = 300.0
# LeKiwi default limits (conservative; the wheeled base is slow).
MAX_LINEAR_SPEED = 0.05
MAX_ANGULAR_SPEED = 0.20
# go2_piper base velocity limits (Task 3): the quadruped is a new /cmd_vel
# consumer with its own conservative caps, wired through the same
# sanitize_twist gate as LeKiwi.
GO2_PIPER_MAX_LINEAR_SPEED = 0.20
GO2_PIPER_MAX_ANGULAR_SPEED = 0.50
COMMAND_FRESHNESS_SEC = 0.25
ZERO_TWIST = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class ValidatedGoal:
    target: str
    timeout_sec: float


def validate_goal(target: str, timeout_sec: float) -> ValidatedGoal:
    """Normalize a navigation goal or reject it before authority is acquired."""
    if not isinstance(target, str):
        raise ValueError("target must be a string")

    normalized_target = target.strip()
    if not normalized_target:
        raise ValueError("target must not be empty")
    if len(normalized_target) > MAX_TARGET_LENGTH:
        raise ValueError(f"target must be at most {MAX_TARGET_LENGTH} characters")

    try:
        normalized_timeout = float(timeout_sec)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout_sec must be numeric") from exc
    if not math.isfinite(normalized_timeout):
        raise ValueError("timeout_sec must be finite")
    if not MIN_TIMEOUT_SEC <= normalized_timeout <= MAX_TIMEOUT_SEC:
        raise ValueError(
            f"timeout_sec must be within [{MIN_TIMEOUT_SEC}, {MAX_TIMEOUT_SEC}]"
        )

    return ValidatedGoal(normalized_target, normalized_timeout)


def lease_is_fresh(
    *,
    active: bool,
    heartbeat_age_sec: float,
    max_age_sec: float = COMMAND_FRESHNESS_SEC,
) -> bool:
    """Return whether command authority is both active and recently renewed."""
    return bool(active) and _age_is_fresh(heartbeat_age_sec, max_age_sec)


def command_is_fresh(
    command_age_sec: float,
    max_age_sec: float = COMMAND_FRESHNESS_SEC,
) -> bool:
    """Return whether the most recent raw velocity command is recent enough."""
    return _age_is_fresh(command_age_sec, max_age_sec)


def sanitize_twist(
    *,
    linear_x: float,
    linear_y: float,
    angular_z: float,
    lease_fresh: bool,
    command_fresh: bool,
    max_linear_speed: float = MAX_LINEAR_SPEED,
    max_angular_speed: float = MAX_ANGULAR_SPEED,
) -> tuple[float, float, float]:
    """Gate and clamp planar velocity, failing closed on invalid input.

    ``max_linear_speed`` / ``max_angular_speed`` default to the LeKiwi caps;
    platform adapters may pass tighter or (for the Go2 base) the configured
    ``go2_piper`` caps. Callers must pass them explicitly per platform.
    """
    if not lease_fresh or not command_fresh:
        return ZERO_TWIST

    values = (linear_x, linear_y, angular_z)
    try:
        finite = all(math.isfinite(value) for value in values)
    except TypeError:
        finite = False
    if not finite:
        return ZERO_TWIST

    return (
        _clamp(float(linear_x), max_linear_speed),
        _clamp(float(linear_y), max_linear_speed),
        _clamp(float(angular_z), max_angular_speed),
    )


def _age_is_fresh(age_sec: float, max_age_sec: float) -> bool:
    try:
        age = float(age_sec)
        maximum_age = float(max_age_sec)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(age)
        and math.isfinite(maximum_age)
        and maximum_age > 0.0
        and 0.0 <= age <= maximum_age
    )


def _clamp(value: float, magnitude: float) -> float:
    return max(-magnitude, min(magnitude, value))
