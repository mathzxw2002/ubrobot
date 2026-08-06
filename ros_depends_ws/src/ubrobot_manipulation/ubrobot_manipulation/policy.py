"""Deterministic safety policy shared by grasp ROS adapters.

Platform bindings (Piper workstation, Go2+Piper mobile manipulator, future
SO101) select a :class:`PlatformProfile` at deployment time; the semantic
``GraspObject`` Action and this policy stay platform-agnostic.
"""

from dataclasses import dataclass
import math


MAX_TARGET_LENGTH = 128
MIN_TIMEOUT_SEC = 1.0
MAX_TIMEOUT_SEC = 300.0


@dataclass(frozen=True)
class ValidatedGraspGoal:
    target: str
    timeout_sec: float


def validate_goal(target: str, timeout_sec: float) -> ValidatedGraspGoal:
    """Normalize a grasp goal or reject it before authority is acquired."""
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

    return ValidatedGraspGoal(normalized_target, normalized_timeout)


@dataclass(frozen=True)
class WorkspaceBox:
    """Axis-aligned reachable workspace in the platform grasp frame (meters)."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def __post_init__(self):
        for axis in ("x", "y", "z"):
            low = getattr(self, f"{axis}_min")
            high = getattr(self, f"{axis}_max")
            if not (
                math.isfinite(low) and math.isfinite(high) and low < high
            ):
                raise ValueError(f"invalid workspace {axis} bounds")

    def contains(self, point) -> bool:
        try:
            x, y, z = (float(value) for value in point)
        except (TypeError, ValueError):
            return False
        if not all(math.isfinite(value) for value in (x, y, z)):
            return False
        return (
            self.x_min <= x <= self.x_max
            and self.y_min <= y <= self.y_max
            and self.z_min <= z <= self.z_max
        )


@dataclass(frozen=True)
class PlatformProfile:
    """Deployment-time binding between the semantic action and a platform."""

    name: str
    executor_kind: str
    workspace: WorkspaceBox
    requires_stationary_base: bool
    max_approach_speed_mps: float
    # Remote perception service base URL (x86 GPU server; empty = local/None).
    remote_perception_service_url: str = ""
    # Conservative base velocity caps used by the shared navigation policy.
    max_base_linear_mps: float = 0.05
    max_base_angular_radps: float = 0.20


PLATFORM_PROFILES = {
    # Standalone Piper workstation: arm base frame, table-height workspace.
    "piper_station": PlatformProfile(
        name="piper_station",
        executor_kind="piper_graspnet",
        workspace=WorkspaceBox(0.10, 0.60, -0.35, 0.35, 0.00, 0.50),
        requires_stationary_base=True,
        max_approach_speed_mps=0.10,
    ),
    # Unitree Go2 carrying a Piper arm: arm base frame on the quadruped
    # back; the base MUST hold still during any grasp. Perception is a
    # remote (x86 GPU) GraspNet service; base caps are the conservative
    # go2_piper limits (Task 3: linear <= 0.2 m/s, angular <= 0.5 rad/s).
    "go2_piper": PlatformProfile(
        name="go2_piper",
        executor_kind="piper_graspnet",
        workspace=WorkspaceBox(0.10, 0.55, -0.30, 0.30, 0.05, 0.55),
        requires_stationary_base=True,
        max_approach_speed_mps=0.05,
        remote_perception_service_url="http://perception-server.local:5802",
        max_base_linear_mps=0.2,
        max_base_angular_radps=0.5,
    ),
}


def get_platform_profile(name: str) -> PlatformProfile:
    normalized = (name or "").strip().lower()
    try:
        return PLATFORM_PROFILES[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown grasp platform profile: {name!r}") from exc


def target_pose_is_reachable(point, workspace: WorkspaceBox) -> bool:
    return workspace.contains(point)


def grasp_may_start(
    *,
    navigation_lease_active: bool,
    base_stationary: bool,
    profile: PlatformProfile,
) -> bool:
    """Mutual exclusion: never grasp while navigation holds motion authority.

    On mobile platforms (Go2+Piper today, LeKiwi/SO101 variants later) base
    motion during a grasp is a tip-over and collision hazard, so the grasp
    capability refuses to start while a navigation command lease is active
    and requires a stationary base whenever the profile demands one.
    """
    if navigation_lease_active:
        return False
    if profile.requires_stationary_base and not base_stationary:
        return False
    return True
