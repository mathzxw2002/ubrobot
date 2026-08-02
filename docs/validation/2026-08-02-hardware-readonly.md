# Robot Edge Read-only Hardware Health Validation Report (M6)

- Date/time: 2026-08-02 21:05 (+08:00)
- Commit: `6c43f32` (M6 Task 11 implementation) + untracked report only
- Machine role: workstation (developer PC, Windows, AMD64, Python 3.13.2)
- Execution mode: software fixtures + fake ROS graph (workstation); the
  robot-side host (Raspberry Pi) is **not started** in this validation
- Mobile-base profile: **`lekiwi`** (owner-selected, see inventory report)
- Hardware authority: **false** everywhere
- Safety controls present in software: lease expiry fail-closed, safety
  latch, emergency-stop endpoint, authorized reset (M5 fixture, unchanged)
- Physical safety controls on robot host: **absent** (no E-stop evidence;
  M7 blocked per inventory report)
- Live hardware/ROS tests: **not executed** — see "Deferred live validation"

## Scope

Milestone M6, Task 11 acceptance — truthful read-only health mapping for
RealSense, the selected mobile base (`lekiwi`), and Piper, with command
authority remaining disabled:

1. RealSense: read camera metadata only (RGB/depth `camera_info`), validate
   dimensions, encoding, frame IDs, calibration presence, and stale behavior.
   Raw depth/color frames never cross the boundary.
2. Mobile base: read only the owner-selected `lekiwi` profile
   (`/odom/wheel`, `/joint_states`); no movement commands are emitted.
3. Piper: verify CAN/driver/arm status with torque disabled; any
   torque-enabled observation is an unhealthy stop condition.
4. Disconnect/stale/missing-topic cases map to `stale`/`disconnected`, never
   healthy.
5. Command endpoints reject with "hardware authority disabled" in read-only
   backend mode.

## Commands

```powershell
PYTHONPATH=src python -m unittest discover -s tests/robot_edge -p "test_*.py" -q
PYTHONPATH=src python -m unittest discover -s tests/cortex_navigation -p "test_*.py" -q
```

## Results

| Suite | Tests | Result |
|---|---:|---|
| `tests/robot_edge` (incl. 26 new M6 ROS-adapter + hardware-health tests) | 99 | PASS |
| `tests/cortex_navigation` (regression guard) | 165 | PASS |

## What the 26 new M6 tests assert

### Import boundaries (both new test modules)

- `robot_edge` package import never imports `rclpy`, `pyrealsense2`,
  `piper_sdk`, `unitree_sdk2py`, `lerobot.cameras.realsense`, or a
  Go2Manager; the hardware-SDK-importing factory is only reachable in
  hardware mode.
- No SDK object, ROS message, or binary frame reaches the shared DTOs or
  FastAPI payloads.

### RealSense (`RealsenseHealthReader`)

- Valid `camera_info` metadata (dimensions, encoding, expected optical frame
  IDs, calibration present) maps to `available` for the CAMERA and DEPTH
  channels.
- Uncalibrated camera metadata is **not** claimed calibrated.
- A stamp older than the 2.0 s deadline is `stale`, not `available`.
- Missing topics or no message map to `disconnected`.
- Raw frames are never serialized; the snapshot carries only validated
  metadata (source, topic, dimensions, encoding, frame IDs, calibration
  flag, stamp age).

### Mobile base (`MobileBaseHealth`, profile `lekiwi`)

- `go2` profile is rejected (`ValueError`) until the owner selects it — the
  "exactly one base profile" constraint is enforced in code, not just prose.
- `/odom/wheel` maps x/y and vx into the ODOMETRY channel; `/joint_states`
  maps names/positions/velocities and motor count into JOINT_STATES.
- Missing topics → `disconnected`; stale odometry → `stale`; fresh → `available`.
- All reads are read-only: the module constructs no motion/session clients.

### Piper (`PiperHealth`)

- No CAN/device present maps to `disconnected` (truthful — the inventory
  found no `can0` on the robot host).
- Ready driver with torque disabled maps to `available` with
  `authority=false`.
- **Torque-enabled observation maps to `unhealthy`/stop-condition** — the
  same gate the inventory report recorded for the LeKiwi driver container.
- No enable, go-zero, trajectory, gripper, or SDK motion methods are called.

### Read-only backend (`RosBackend`)

- Hardware mode with no authority: readiness reports
  `execution_mode=hardware`, `hardware_authority=false`.
- All command sequences raise "hardware authority disabled".
- All six telemetry channels are present with explicit state, timestamps are
  timezone-aware, and missing Actions/topics report `unavailable`/
  `disconnected`, never healthy.

## Hardware authority state

- `hardware_authority=false` in every module, snapshot, and test.
- Command/cancel/stop authority is structurally disabled in the read-only
  backend; only fixture mode was exercised end to end (M5 report).

## Known limitations and deferred live validation

1. This report is **software-only evidence**. All ROS data came from the
   fake `RosGraph` fixture; the mapped topics
   (`/camera/camera_info`, `/camera/depth/camera_info`, `/odom/wheel`,
   `/joint_states`, Piper CAN status) were not observed on a live ROS graph.
2. Live robot-side validation on the Raspberry Pi remains blocked by the
   inventory gate: the LeKiwi driver container is configured with
   `enable_motor_torque:=true` and must not be started until a supervised
   lifted-wheel preflight is approved and the physical E-stop is verified
   (inventory report, critical findings 1–2).
3. The RealSense serial mismatch found during inventory is unresolved on the
   committed config; the camera health reader is serial-agnostic by design,
   so this does not block Task 11 software validation, but the live D435i
   must be re-confirmed against the updated config before M7.
4. Piper is not connected to this host (no CAN); `PiperHealth` behavior on a
   live driver is untested and deferred to M8.
5. Measured live telemetry rates and age are not yet available; the 2.0 s
   stale deadline is asserted in unit tests only.

## Acceptance (Task 11)

- [x] Observation/odometry/joints mapping shows truthful
      available/stale/disconnected states in software fixtures.
- [x] Command authority remains false everywhere.
- [x] Import boundaries hold: no hardware/ROS SDK in workstation processes.
- [x] No torque enable and no motion-capable method reachable from these
      modules.
- [ ] Live ROS graph read-only validation — **deferred**, blocked by the
      torque-enabled LeKiwi container gate and absent physical E-stop.

Mock/fixture evidence is not hardware evidence. M6 Task 11 is accepted at
the software level only; the live read-only check and M7 stay blocked per
the plan's stop conditions.
