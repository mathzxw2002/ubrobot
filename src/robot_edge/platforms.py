"""Robot Edge platform assembly registry (single source of truth).

``go2_piper`` is the only platform that couples a Go2 mobile base with a
Piper arm and remote (x86 GPU) perception. ``platforms.py`` is the single
enumeration source for Robot Edge: adapters, health readers and capability
gates must reference ``get_platform(...)`` rather than re-deriving the
assembly from environment variables.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlatformDefinition:
    """Read-only description of one robot platform.

    Attributes:
        key: canonical platform key (e.g. ``"go2_piper"``).
        base: mobile base subsystem id (``"go2"`` | ``"lekiwi"``).
        arm: manipulation subsystem id or None (``"piper"``).
        perception: perception mode (``"remote-service"`` | ``"local"``).
        requires_stationary_base: grasp requires a fully stationary base.
        max_base_linear_mps: conservative linear speed cap for the base.
        max_base_angular_radps: conservative angular speed cap for the base.
    """

    key: str
    base: str
    arm: str | None
    perception: str
    requires_stationary_base: bool
    max_base_linear_mps: float
    max_base_angular_radps: float


# Conservative limits for the Go2 base (Task 1 inventory: linear <= 0.2 m/s,
# angular <= 0.5 rad/s for the first bring-up). These are limits, never
# targets; the navigation stack's own policy clamps harder if needed.
_GO2_PIPER = PlatformDefinition(
    key="go2_piper",
    base="go2",
    arm="piper",
    perception="remote-service",
    requires_stationary_base=True,
    max_base_linear_mps=0.2,
    max_base_angular_radps=0.5,
)

# LeKiwi base alone: no arm, local perception, no stationary-base
# requirement (navigation-only profile used by the Pi).
_LEKIWI = PlatformDefinition(
    key="lekiwi",
    base="lekiwi",
    arm=None,
    perception="local",
    requires_stationary_base=False,
    max_base_linear_mps=0.05,
    max_base_angular_radps=0.2,
)

_PLATFORMS: dict[str, PlatformDefinition] = {
    _GO2_PIPER.key: _GO2_PIPER,
    _LEKIWI.key: _LEKIWI,
}


def supported_platforms() -> tuple[str, ...]:
    """All platform keys known to Robot Edge, in canonical order."""
    return tuple(_PLATFORMS)


def get_platform(key: str) -> PlatformDefinition:
    """Return the platform definition for ``key`` or raise ValueError."""
    try:
        return _PLATFORMS[key]
    except KeyError:
        raise ValueError(
            f"unsupported platform {key!r}; supported: {', '.join(supported_platforms())}"
        ) from None
