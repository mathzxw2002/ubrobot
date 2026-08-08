# `ubrobot.robots` — robot adapters and research code

Status legend: **production** (used by the operator stack) · **rollback/research**
(deprecated hardware-direct paths, kept for experiments) · **vendored/experimental**
(third-party or unfinished).

## Subdirectories

| Path | Status | Purpose |
|---|---|---|
| `lekiwi/` | rollback/research | LeKiwi three-omniwheel base (LeRobot-style); hardware connection is explicit (`connect_base`). |
| `piper/` | rollback/research | AgileX Piper arm SDK interface, LeRobot client/host, ZMQ teleop. |
| `so101_follower/` | rollback/research | SO-101 follower/client/host for networked teleoperation. |
| `logoplanner/` | vendored/experimental | Logoplanner / Pi3 / Depth-Anything model checkpoints and hosts. Not part of the operator stack. |
| `navdp/` | experimental | Navigation data-processing experiments (unfinished). |

## Top-level modules

| Module | Status | Notes |
|---|---|---|
| `ubrobot.py` | rollback/research | `Go2Manager` — legacy keyword-driven agent loop. `connect_base()` is explicit; constructing the class never attaches hardware. |
| `unitree_go2_robot.py` | **deprecated** | Direct `SportClient` motion. Go2 must move via `/cmd_vel` (Kompass). Import emits `DeprecationWarning`; SDK imported lazily. |
| `vlm.py` / `nav.py` / `asr.py` / `tts.py` | rollback/research | VLM reasoning, navigation policies, ASR/TTS service clients. |
| `controllers.py` / `pointcloud.py` / `utils.py` / `thread_utils.py` | rollback/research | MPC/PID controllers, point-cloud perception, helpers. |
| `arm_action.py` | **archived** | Moved to `archive/arm_action.py` (dead experimental code referencing undefined symbols; not importable). |

## Rules

- **Production robot motion never imports `unitree_go2_robot` or raw
  `SportClient`.** Go2 moves only through the Kompass `/cmd_vel` chain; Piper
  through the go2-piper-driver container. See
  `docs/plans/2026-08-06-go2-piper-cortex-integration.md`.
- Hardware-direct adapters here are rollback/research only: they are **not**
  wired into the Operator Console's default path (`cortex`/`cortex-mock`/
  `robot-edge` backends).
- Archived files under `archive/` are excluded from packaging; do not import
  them from `src/`.
