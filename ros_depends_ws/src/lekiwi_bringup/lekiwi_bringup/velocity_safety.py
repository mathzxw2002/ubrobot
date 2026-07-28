"""Pure velocity validation helpers shared by the ROS adapter and tests."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class VelocityLimits:
    linear_x: float = 0.05
    linear_y: float = 0.05
    angular_z: float = 0.20

    def __post_init__(self) -> None:
        values = (self.linear_x, self.linear_y, self.angular_z)
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("velocity limits must be finite and greater than zero")


def _clip(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def sanitize_velocity(
    linear_x: float,
    linear_y: float,
    angular_z: float,
    limits: VelocityLimits,
) -> tuple[float, float, float, bool]:
    """Validate and clamp a body velocity command.

    Invalid commands always become an explicit zero command and return ``False``.
    """

    values = (linear_x, linear_y, angular_z)
    if not all(math.isfinite(value) for value in values):
        return 0.0, 0.0, 0.0, False

    return (
        _clip(linear_x, limits.linear_x),
        _clip(linear_y, limits.linear_y),
        _clip(angular_z, limits.angular_z),
        True,
    )
