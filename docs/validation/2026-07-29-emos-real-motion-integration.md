# 2026-07-29 EMOS Real-Motion Integration — Results and Incident Record

**Scope:** First attempt at the open item from
[2026-07-28 lifted-wheel validation plan](../plans/2026-07-28-lekiwi-lifted-wheel-hardware-validation.md):
send `/track_vision_target` and let EMOS actually drive the LeKiwi base.

**Outcome:** Mock-mode integration passed completely. The first real-hardware
attempt ended in a **runaway-prevention incident**: a stale action goal drove
the robot as soon as the driver came up; the operator cut motor power and USB.
No damage. Three root causes identified; fixes implemented in this commit.

## Environment state at session start

- Pi containers up ~14 h: `emos` (`ubrobot/emos:jazzy-0209959`),
  `lekiwi-base-driver` mock (`0.2.0-rc1-emos-stage1`, no device mapping).
- Sensor chain alive from yesterday's manual processes: RealSense
  (640×480@15, RGBD), `/scan` 8.8 Hz, `/odom` ~3 Hz, TF chain complete,
  `/cmd_vel` exactly 1 publisher (my_driver) / 1 subscriber (adapter).
- **Broken:** `/vision_detections_raw` at 0.1 Hz. Root cause: VLM server
  (192.168.18.230) has `torch 2.12.0+cu130` but NVIDIA driver 535 (CUDA 12.2)
  → CUDA unavailable → RT-DETR R50 runs on CPU. Decision: do NOT touch the
  server today; proceed with slow detections.

## Stage 1 — mock-mode action integration: PASS

Two blocking defects found and fixed first:

1. **kompass patch lost.** The `_vision_follower.py` setup-condition patch
   (emos.md §7) was applied container-locally yesterday, then lost when the
   container was recreated from `jazzy-0209959`. Symptom: goal accepted, then
   ABORTED after 0.012 s. Re-applied at runtime; now also baked into
   `deploy/emos/Dockerfile`.
2. **`fix_detection_header` relay never ran.** The recipe outputs detections
   to `/vision_detections_raw` while `my_controller` subscribes
   `/vision_detections`; the header-fixing relay (part of the
   `emos_bringup` launch) was never started during yesterday's manual bringup.
   Started it via `docker cp` + detached process. Persisted properly in the
   image/compose (see Fixes).

After both fixes, goal `{label: "chair"}` (no person available; chair ~2.7 m
in front of camera, detected at 0.93 confidence):

- Feedback streamed continuously for the full 105 s window:
  `distance_error ≈ -2.66 m`, `orientation_error ≈ 0.013–0.020 rad`.
- `/cmd_vel`: pure forward `linear.x ≈ 0.2476 m/s` (at the recipe's 0.25 m/s
  limit), `angular.z = 0` throughout.
- Mock wheel signature `[rear 0, left -0.866, right +0.866]` — matches the
  2026-07-28 mock baseline for +x (`0/−/+`).
- Stop-on-stale-target verified: between CPU-inference frames (~10 s apart)
  commands drop to zero and resume on the next detection — DriveManager
  behaves correctly with slow detections.

## Stage 2 — first real-hardware attempt: INCIDENT

Sequence (times approximate, driver container `0.2.0-rc1-b90fa1c`):

1. Stage-1 action client was killed by its 105 s `timeout` **without
   cancelling the goal**. The server-side goal stayed active; `my_driver`
   kept publishing `linear.x = 0.2465` on `/cmd_vel`.
2. Driver switched to real + torque-test override (`restart: "no"`, device
   mapped, `motor torque ENABLED with zero command` logged).
3. ~14 s after start, `/joint_states` showed **-0.3835 rad/s on all three
   wheels** (= -250 steps/s, exactly 5× the ±0.0767 quantization unit; the
   `−/−/−` pattern corresponds to negative rotation).
4. The operator observed real motion and **cut motor power and USB** — the
   authoritative safety response.
5. `docker stop -t 3` then exited with code **143 (SIGTERM)**: the Pi's
   `deploy/lekiwi-driver/compose.yaml` lacked `stop_signal: SIGINT` (fix
   bcc058c was never synced; the Pi repo sat at June commit 53a7922), so the
   graceful deactivate/torque-release path did not run.
6. The stale goal was cleared by restarting the EMOS recipe; `/cmd_vel` went
   silent. Compose files on the Pi were then synced from repo HEAD
   (checksum-verified, `.bak` backups kept).

### Root causes

1. **Stale action goals survive client death.** kompass `my_controller`
   keeps tracking (and `my_driver` keeps publishing) after the client
   disappears. Any driver (re)start inherits the live command stream.
2. **Pi deploy drift.** The Pi repo is not the source of truth it appears to
   be; yesterday's fixes existed only in the Windows repo branch and in
   images. `stop_signal: SIGINT` was missing on the Pi.
3. **-0.383 rad/s at T+14 s is real motion, not a parse bug** — status
   packets are ID-validated and checksummed per motor, so three identical
   values mean three motors genuinely reported -250 steps/s. The exact
   provenance is still open: the stale 0.2476 m/s x-command explains motion
   *after* controller activation, but not this early rotation signature, and
   `enable_torque()`'s zeroing used a fire-and-forget sync write that cannot
   prove the goal registers actually cleared. Treated as "unexplained motion
   right after torque enable" and hardened against (Fix 1).

## Fixes implemented in this commit

1. **Verified goal-zero + stationary assertion** (`feetech_bus`):
   - New `zero_goal_registers_verified()`: per-motor acknowledged write of
     goal velocity 0 **plus register read-back**; throws on mismatch. Called
     from `configure_velocity_mode()`, `enable_torque()` (replacing the blind
     sync-write zero), and `stop_and_disable()`.
   - New `assert_wheels_stationary(150)`: after torque enable, every motor's
     present velocity must be ≤150 steps/s (idle jitter is ±50; the clamp is
     300). Violation → exception → `on_activate()` fails safe with torque
     disabled.
2. **EMOS full-stack persistence** (no more manual processes):
   - `emos_bringup` overlay (launch + `fix_detection_header`) colcon-built
     into the EMOS image at `/opt/emos_overlay`.
   - `deploy/emos/start-stack.sh` supervises sensor chain + recipe, logs to
     `/home/china/emos/logs`, exits non-zero if any stack process dies so
     `restart: always` recreates a clean stack; wired as the compose
     `command`.
   - Launch file: added the `camera_depth_frame → camera_depth_link`
     identity alias (RealSense 4.58 frame naming) and **removed** the direct
     `base_link → camera_depth_link` publisher that would have given
     `camera_depth_link` two TF parents.
   - kompass `_vision_follower.py` patch baked into the Dockerfile.
   - Reference recipe (with `detections_raw` wiring) version-controlled at
     `deploy/emos/recipes/vision_depth_follower/recipe.py`; the supervisor
     seeds it only if the data-dir recipe is missing.
3. **Repo hygiene:** `.gitattributes` keeps `*.sh` at LF for Linux
   containers; 3 new deployment-contract tests (18/18 pass); g++ syntax
   check of the modified bus code passed on ARM64.

## Updated safety procedure (mandatory for future real-mode sessions)

- **Before any driver (re)start or mode switch:** cancel all action goals
  (or restart the recipe) and verify `/cmd_vel` carries no non-zero data.
  Client-side `timeout` does NOT cancel server-side goals.
- **Before real-mode start:** after torque enable, wheel velocities must be
  ~0 — now enforced in software by `assert_wheels_stationary`.
- The physical motor-power cutoff remains the authoritative stop; software
  stops are secondary.

## Deployment outcome (same day, evening)

- Built `ubrobot/emos:jazzy-7a64982` and
  `ubrobot/lekiwi-base-driver:0.2.0-rc1-7a64982` on the Pi from a
  checksummed git archive (build dir `/home/china/ubrobot-builds/7a64982-stack`).
- First supervised boot exposed two startup bugs, fixed in `start-stack.sh`
  and re-deployed:
  1. `set -u` aborted on `COLCON_TRACE` referenced by the ROS setup scripts
     (container restart loop);
  2. `librealsense2.so.2.58` / `librtabmap_core.so.0.22` live in non-default
     paths in the EMOS base image — the supervisor now exports
     `LD_LIBRARY_PATH` (the old manual processes did this inline).
- Final supervised state verified: `/scan` 11.2 Hz, `/odom` 2.4 Hz, all
  lifecycle components active, `/vision_detections` 1 pub + 1 sub,
  TF `base_link → camera_depth_link` present via the single-parent chain,
  `/track_vision_target` available, `/cmd_vel` silent.
- `lekiwi-base-driver` recreated in mock mode with the hardened image
  (healthy, no device mapping). Motor power and USB remained unplugged after
  the incident; no real-mode run was attempted.

## Open items

- **VLM server GPU inference**: torch 2.12.0+cu130 vs driver 535 (CUDA 12.2).
  Proposed fix: install a cu121-matched torch in the `roboml` env (owner:
  sany's server; deferred by user decision today). Detection at 0.1 Hz makes
  tracking unusable beyond static proof-of-concept.
- **Motor state dump before next real test**: read goal-velocity, present
  velocity, and error registers before enabling torque, to close the
  -250 steps/s open question. The new `assert_wheels_stationary(150)` will
  additionally fail activation if the anomaly recurs.
- Branch `codex/lekiwi-hardware-0.2.0-rc1` still unpushed.

## Evidence pointers (on the Pi)

- `/tmp/stage1c_action.log` — stage-1 feedback stream
- `/tmp/stage1c_cmdvel.log` — `/cmd_vel` during tracking
- `/tmp/stage1c_joints.log` — mock wheel velocities
- `/home/china/emos/logs/` — recipe, sensor chain, relay logs
- Driver container logs via `docker logs lekiwi-base-driver` (pre-stop)
