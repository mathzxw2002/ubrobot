"""Grasp executor binding selection (pure, no ROS/rclpy imports).

``resolve_executor_binding`` is the single decision point used by
``grasp_object_server.build_executor`` and by workstation tests. It keeps
the "which executor for this platform/env" logic free of rclpy so it can be
validated without a ROS installation.
"""

from __future__ import annotations

from typing import Any

from ..policy import PlatformProfile

FIXTURE_KIND = "fixture"
HARDWARE_KIND = "hardware"


def resolve_executor_binding(
    profile: PlatformProfile,
    executor_kind: str,
    *,
    fixture_phase_delay_sec: float = 0.05,
) -> Any:
    """Return a fresh executor for ``profile`` + ``executor_kind``.

    - ``fixture``: deterministic offline fixture.
    - ``go2_piper`` + ``hardware``: real remote-perception / local-motion
      executor (see ``build_go2_piper_executor``).
    - anything else: ``NotImplementedError`` (an accidental hardware
      connection must never happen).
    """
    kind = (executor_kind or "").strip().lower()
    if kind == FIXTURE_KIND:
        from .fixture import DeterministicGraspExecutor

        return DeterministicGraspExecutor(
            profile=profile,
            phase_delay_sec=fixture_phase_delay_sec,
        )
    if profile.name == "go2_piper" and kind == HARDWARE_KIND:
        from .go2_piper import build_go2_piper_executor

        return build_go2_piper_executor(profile)
    raise NotImplementedError(
        f"no grasp executor binding implemented for profile "
        f"'{profile.name}' (executor kind '{kind}')"
    )
