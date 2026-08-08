# Legacy / rollback hardware-direct code

This page documents the deprecated hardware-direct code paths that remain in
the repo for research/rollback, their status, and the production alternatives.

## Why these exist

UBRobot's history has several generations of robot drivers. The current
operator stack routes motion through semantic capabilities (`NavigateToObject`
/ `GraspObject` → Cortex → executor → guarded `/cmd_vel` / driver container),
which is the only path that carries the fail-closed safety gates. The old
hardware-direct adapters are kept for experiments but must not be wired into
the production path.

## Status table

| Module | Status | Deprecated because | Production alternative |
|---|---|---|---|
| `ubrobot.robots.unitree_go2_robot.UnitreeGo2Robot` | deprecated (import warns) | Direct `SportClient` motion bypasses the Kompass `/cmd_vel` chain and its velocity/lease guard | Go2 moves via `/cmd_vel` → Kompass → `cmd_vel_guard` (see `2026-08-06-go2-piper-cortex-integration.md`) |
| `ubrobot.robots.ubrobot.Go2Manager` | rollback/research | Keyword-driven agent loop, legacy `UBROBOT_CHAT_BACKEND=legacy` path; connects hardware implicitly | `cortex` / `cortex-mock` / `robot-edge` backends in `chat_ui` |
| `ubrobot.robots.arm_action.PoseTransformer` | **archived** (`archive/arm_action.py`) | Dead experimental code referencing undefined symbols (`tf_trans`, `tf2_ros`, `Float64`, `self.piper_mp`, ...); not importable | `src/service/reasoning/http_reasoning_server.py` + go2-piper-driver for grasp |

## Migration notes (Task 2)

- `Go2Manager.__init__` no longer calls `lekiwi_base.connect()`; use the
  explicit `connect_base()` before `start_threads()` on the robot. Constructing
  `Go2Manager` on a workstation is import-safe (no serial/camera attach).
- `unitree_go2_robot` imports `unitree_sdk2py` lazily inside `__init__`, so
  importing the module does not require the SDK.
- `archive/arm_action.py` is not part of `src/` packaging; do not import it.

## Rollback policy

- Setting `UBROBOT_CHAT_BACKEND=legacy` routes the console through
  `_LegacyBackend` → `Go2Manager`. This requires connected LeKiwi base +
  camera hardware; failures surface as a readable `RuntimeError`, not a bare
  crash.
- The old keyword-based navigation recipe (`vision_depth_follower`) remains in
  the EMOS image for rollback, but the operator stack default is the Cortex
  semantic path.
