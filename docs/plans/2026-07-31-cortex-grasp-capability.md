# Cortex Grasp Capability Layer — Design (W4)

> **Status: interface + logic layer only.** ROS node, executor bindings, and
> on-robot validation wait for the Raspberry Pi / arm machines to return.
> Written 2026-07-31 during the Windows-only development window.

**Goal:** give Cortex a second semantic capability — `GraspObject` — that
works across multiple arm platforms without the planner ever seeing
joints, torque, CAN buses, or base velocity, and that can never run while
navigation holds motion authority.

## Platform matrix

| Profile | Platform | Executor kind | Base coupling |
|---|---|---|---|
| `piper_station` | Standalone Piper workstation | `piper_graspnet` (existing `src/service/reasoning/grasp_plan.py` + GraspNet + piper_ros) | No mobile base; arm base frame is world-fixed |
| `go2_piper` | Unitree Go2 quadruped carrying a Piper arm | `piper_graspnet` (same arm stack, different TF root) | Base MUST be stationary during grasp; conservative limits |
| future: `so101` | SO101 arm (LeKiwi-style) | TBD | Same mutual-exclusion rules |

Adding a platform = one `PlatformProfile` entry plus one
`GraspExecutorAdapter` implementation. The Action interface, lifecycle,
and Cortex tool surface do not change.

## Architecture (mirrors the navigation capability)

```text
Chat UI / Cortex planner
  -> /ubrobot/manipulation/grasp_object (semantic Action, capability server)
       - GraspLifecycleCoordinator: single goal, cancel/timeout/exception paths
       - MotionAuthorityAdapter: observes /navigation/command_lease + base state
       - policy: target/timeout validation, workspace bounds, exclusion rules
  -> GraspExecutorAdapter (platform binding, owns arm stack)
  -> arm hardware
```

Deliberate differences from navigation:

- **No lease/guard for the arm.** The navigation guard exists because
  `/cmd_vel` is a shared, anonymous, high-rate channel. Arm execution is a
  single owned executor behind the capability server; safety comes from the
  lifecycle (single goal, bounded cancel) and the authority checks below.
- **Mutual exclusion is explicit.** `grasp_may_start` refuses a grasp while
  `navigation_lease_active`, and the coordinator re-checks every poll
  (50 ms): if a navigation lease appears mid-grasp, the executor is
  cancelled and the goal fails — base motion during a grasp is a tip-over
  and collision hazard on Go2, and would break perception assumptions on a
  stationary workstation. The reverse direction (navigation refused during
  grasp) is handled at the Cortex level by single-goal planning today and
  will get a symmetric check when the grasp server is deployed.

## Interface

`ubrobot_interfaces/action/GraspObject.action`:

```text
string target
float32 timeout_sec
---
uint8 SUCCEEDED=0
uint8 CANCELLED=1
uint8 TIMED_OUT=2
uint8 REJECTED=3
uint8 FAILED=4
uint8 status
string message
---
string phase
float32 target_distance_m
float32 progress
```

Same status constants as `NavigateToObject` so UI/Cortex result handling is
uniform. Feedback phases are executor-defined (`approach`, `align`,
`grasp`, `retreat`, …) with normalized `progress` in [0, 1].

## Safety invariants (enforced in `policy.py` / `lifecycle.py`, all tested)

1. Target validated before anything starts: non-empty, ≤128 chars,
   timeout finite within [1, 300] s.
2. Workspace bounds per profile (`WorkspaceBox` in the platform grasp
   frame); executors must reject unreachable poses — checked again in the
   adapter, never trusted from the planner.
3. `grasp_may_start(navigation_lease_active, base_stationary, profile)`:
   no grasp while navigation holds motion authority; stationary base when
   the profile requires it (both current profiles do).
4. Single goal at a time (`GoalBusyError`); every return path releases the
   slot.
5. Outer cancel and timeout both cancel the executor with a bounded
   acknowledgement (2 s); exceptions cancel best-effort and return FAILED.
6. Navigation lease appearing mid-grasp → executor cancelled immediately,
   goal FAILED (fail safe, never share authority).

## Offline deliverables (this milestone)

- `GraspObject.action` registered in `ubrobot_interfaces`.
- `ubrobot_manipulation` package: `policy.py` (validation, workspace,
  profiles, exclusion) and `lifecycle.py` (`GraspLifecycleCoordinator`
  with `GraspExecutorAdapter` / `MotionAuthorityAdapter` protocols).
- 24 package tests + repo contract tests (action shape, build wiring,
  design doc coverage, recipe does NOT expose grasp yet).

## Deferred until machines return (in order)

1. **ROS node skeleton** `grasp_object_server.py`: serve the Action, wire
   `MotionAuthorityAdapter` to `/navigation/command_lease` (empty = free)
   and base odometry/`/cmd_vel` zero-check, select profile from
   `UBROBOT_GRASP_PLATFORM` env. Overlay builds on the Pi (Dockerfile COPY
   already in place).
2. **Executor fixture** (deterministic, like the navigation
   TrackVisionTarget fixture) + mock e2e: Cortex → grasp tool → fixture,
   including navigation/grasp mutual-exclusion injection.
3. **Recipe exposure**: register the grasp Action as a second
   `NavigationCapabilityProxy`-style metadata component, tool description
   (grasps one visually detectable object label; cancellable; fails when
   perception or the arm is unavailable; never moves the base).
4. **Piper executor**: adapt `grasp_plan.py`/GraspNet + piper_ros behind
   `GraspExecutorAdapter`, starting with `piper_station` on the real
   workstation; workspace calibration per profile.
5. **Go2+Piper executor**: same adapter kind, TF rooted at the arm base on
   the quadruped; base-stationary evidence from Go2 odometry; hardware
   gates written (not executed) mirroring the LeKiwi hardware plan style.
6. **Multi-step orchestration validation**: "导航到桌子 → 抓取杯子" as one
   Cortex plan (two sequential semantic goals), first in mock, then lifted,
   then ground — each step separately authorized.
