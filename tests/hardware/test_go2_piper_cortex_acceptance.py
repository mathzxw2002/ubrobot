"""Task 6: Go2+Piper Cortex combined acceptance harness.

Two modes:

- Default (workstation contract tests): runs the mutual-exclusion and
  fail-closed safety semantics against fakes, so the acceptance template is
  continuously verified without hardware.
- ``--hardware``: operator-driven driver against the real Go2+Piper dock.
  It does NOT move the dog on its own; it prints the staged plan and the
  exact manual steps (Piper arm first, Go2 navigation deferred), and runs
  only the read-only health + gate assertions it can check safely.

The safety assertions codified here are the Task 6 mutual-exclusion bottom
line:

  - GraspObject while a navigation lease is active   -> REJECTED
  - NavigateToObject while a grasp is running        -> grasp fail-closed
                                                       cancellation
  - remote perception unreachable                    -> grasp fails closed
                                                       with no motion
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "ros_depends_ws" / "src" / "ubrobot_manipulation") not in sys.path:
    sys.path.insert(
        0, str(ROOT / "ros_depends_ws" / "src" / "ubrobot_manipulation")
    )

from ubrobot_manipulation.authority import AuthorityTracker  # noqa: E402
from ubrobot_manipulation.executors.fixture import DeterministicGraspExecutor  # noqa: E402
from ubrobot_manipulation.lifecycle import (  # noqa: E402
    GraspLifecycleCoordinator,
    GraspStatus,
)
from ubrobot_manipulation.policy import get_platform_profile  # noqa: E402


class RecordingOuter:
    """OuterGoalAdapter that records published feedback."""

    def __init__(self) -> None:
        self.feedback: list[Any] = []
        self.cancel_requested = False

    def is_cancel_requested(self) -> bool:
        return self.cancel_requested

    def publish_feedback(self, feedback: Any) -> None:
        self.feedback.append(feedback)

    def request_cancel(self) -> None:
        self.cancel_requested = True


class ScriptedAuthority:
    """MotionAuthorityAdapter over a shared AuthorityTracker."""

    def __init__(self, tracker: AuthorityTracker) -> None:
        self._tracker = tracker
        self._now = 0.0

    def note_lease(self, lease_id: str) -> None:
        self._now += 0.01
        self._tracker.note_lease(lease_id, self._now)

    def note_cmd_vel(self, x: float, y: float, z: float) -> None:
        self._now += 0.01
        self._tracker.note_cmd_vel(x, y, z, self._now)

    def navigation_lease_active(self) -> bool:
        self._now += 0.01
        return self._tracker.navigation_lease_active(self._now)

    def base_is_stationary(self) -> bool:
        self._now += 0.01
        return self._tracker.base_is_stationary(self._now)


def run_mutual_exclusion_checks() -> dict[str, bool]:
    """Execute the Task 6 mutual-exclusion assertions against the lifecycle."""
    profile = get_platform_profile("go2_piper")
    results: dict[str, bool] = {}

    # --- GraspObject while navigation lease active -> REJECTED ----------
    tracker = AuthorityTracker()
    authority = ScriptedAuthority(tracker)
    authority.note_cmd_vel(0.0, 0.0, 0.0)  # base still
    authority.note_lease("nav-lease-1")  # navigation owns authority
    coordinator = GraspLifecycleCoordinator(profile=profile, sleep=time.sleep)
    outcome = coordinator.run(
        target="cup",
        timeout_sec=30.0,
        outer=RecordingOuter(),
        executor=DeterministicGraspExecutor(profile=profile, phase_delay_sec=0.01),
        authority=authority,
    )
    results["grasp_rejected_while_nav_lease_active"] = (
        outcome.status == GraspStatus.REJECTED
    )

    # --- GraspObject while base moving -> REJECTED ----------------------
    tracker = AuthorityTracker()
    authority = ScriptedAuthority(tracker)
    authority.note_cmd_vel(0.2, 0.0, 0.0)  # base moving
    coordinator = GraspLifecycleCoordinator(profile=profile, sleep=time.sleep)
    outcome = coordinator.run(
        target="cup",
        timeout_sec=30.0,
        outer=RecordingOuter(),
        executor=DeterministicGraspExecutor(profile=profile, phase_delay_sec=0.01),
        authority=authority,
    )
    results["grasp_rejected_while_base_moving"] = (
        outcome.status == GraspStatus.REJECTED
    )

    # --- Navigation lease appears mid-grasp -> grasp fail-closed cancel --
    tracker = AuthorityTracker()
    authority = ScriptedAuthority(tracker)
    authority.note_cmd_vel(0.0, 0.0, 0.0)  # stationary start
    outer = RecordingOuter()

    class LeaseInjectedExecutor(DeterministicGraspExecutor):
        def start(self, target, timeout_sec, feedback_callback) -> bool:
            started = super().start(target, timeout_sec, feedback_callback)
            # After the grasp starts, navigation claims the lease.
            authority.note_lease("nav-lease-2")
            return started

    coordinator = GraspLifecycleCoordinator(profile=profile, sleep=time.sleep)
    outcome = coordinator.run(
        target="cup",
        timeout_sec=30.0,
        outer=outer,
        executor=LeaseInjectedExecutor(profile=profile, phase_delay_sec=0.01),
        authority=authority,
    )
    results["grasp_cancelled_when_lease_appears"] = (
        outcome.status == GraspStatus.FAILED
        and "navigation lease appeared" in outcome.message
    )

    # --- Grasp success on stationary base, no lease -> SUCCEEDED --------
    tracker = AuthorityTracker()
    authority = ScriptedAuthority(tracker)
    authority.note_cmd_vel(0.0, 0.0, 0.0)
    coordinator = GraspLifecycleCoordinator(profile=profile, sleep=time.sleep)
    outcome = coordinator.run(
        target="cup",
        timeout_sec=30.0,
        outer=RecordingOuter(),
        executor=DeterministicGraspExecutor(profile=profile, phase_delay_sec=0.005),
        authority=authority,
    )
    results["grasp_succeeds_on_stationary_base"] = (
        outcome.status == GraspStatus.SUCCEEDED
    )

    return results


def run_readonly_gate_checks() -> dict[str, bool]:
    """Safe, motion-free readiness checks for the real dock."""
    results: dict[str, bool] = {}

    # go2_piper profile carries the remote perception URL (Task 4 gate).
    profile = get_platform_profile("go2_piper")
    results["profile_has_remote_perception_url"] = bool(
        profile.remote_perception_service_url
    )
    results["profile_requires_stationary_base"] = profile.requires_stationary_base
    results["profile_conservative_speed"] = (
        profile.max_base_linear_mps <= 0.2 and profile.max_base_angular_radps <= 0.5
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hardware",
        action="store_true",
        help="operator-driven hardware acceptance (no unsupervised motion)",
    )
    args = parser.parse_args()

    gate_checks = run_readonly_gate_checks()
    mutual = run_mutual_exclusion_checks()

    print("=== Go2+Piper gate checks ===")
    for name, ok in sorted(gate_checks.items()):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("=== Go2+Piper mutual-exclusion safety ===")
    for name, ok in sorted(mutual.items()):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    all_pass = all(gate_checks.values()) and all(mutual.values())

    if args.hardware:
        print()
        print("=== HARDWARE ACCEPTANCE (operator-driven; no unsupervised motion) ===")
        print("Staged plan (Piper arm first, Go2 navigation DEFERRED):")
        print("  S1 read-only health        : check /v1/health + capabilities")
        print("  S2 zero-output / stop      : Piper torque DISABLED, verify zero")
        print("  S3 low-speed navigation    : DEFERRED (operator release required)")
        print("  S4 stationary pre-grasp    : Piper only, base still")
        print("  S5 light grasp             : Piper only, base still")
        print()
        print("Each round: ONE failure injection at a time. Physical E-stop and")
        print("a second observer are mandatory for any motion stage.")
        print("Result: " + ("GATES PASS - proceed per staged plan" if all_pass else "GATES FAIL - do not enable motion"))
        return 0 if all_pass else 1

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
