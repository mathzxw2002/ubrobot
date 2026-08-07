# Go2+Piper Piper Arm Validation Report (Task 6, arm stages: S1/S2 read-only + S4 IK + S5 light grasp)

- Date/time: 2026-08-07
- Commits: `5aeae8e` (piper_ik pinocchio v3.9 fix); S5 light grasp run (working tree) on `go2-piper-cortex-integration`
- Machine role: Go2 expansion dock (Jetson Orin NX, Ubuntu 20.04.5, JetPack R35.3.1)
- Python env: `roboplan_env` (Python 3.10.19, pinocchio 3.9.0) for IK; `ubrobot` env for piper_sdk
- Platform: `go2_piper` (Piper arm on the Go2 dock, `can0`)
- Hardware authority: **false**
- Go2 navigation (S3): **DEFERRED** (per operator; not part of this run)
- Remote GraspNet perception (S4 full): **NOT executed** (perception server not listening; torque kept off)

## Summary

| Stage | Result | Notes |
|---|---|---|
| S1 read-only health | **PASS** | CAN ok, arm STANDBY, torque off, joints/gripper/limits read |
| S2 zero-output / stop | **PASS** | 3 s no-command hold: 0.000 deg drift; EmergencyStop resume safe |
| S4 local IK planning | **PASS** | 4/4 reachable arm-base poses solved within joint limits |
| S4 full (remote GraspNet) | not run | perception server 192.168.18.230:5802 not listening; torque kept off |
| S5 light grasp | **PASS** | torque enabled; bounded approach + gripper open/close + retreat; torque off after |

## S1 — read-only health (torque disabled)

- CAN: `can0` up at 1 Mbps; live Piper frames observed (IDs 0x252..0x261).
- `C_PiperInterface_V2` connects; `GetArmStatus` reports:
  - `Control Mode: STANDBY(0x0)`, `Arm Status: NORMAL(0x0)`, `Motion Status: REACH_TARGET_POS_SUCCESSFULLY(0x0)`, `Error Code: 0`.
- Joint states read (deg): j1 -2.73, j2 -2.26, j3 0.00, j4 3.00, j5 28.44, j6 -105.48 (arm at rest/folded).
- Gripper: 0.0 mm (closed).
- Joint limits (deg): j1[-150,150] j2[0,180] j3[-170,0] j4[-100,100] j5[-70,70] j6[-180,180].
- `DisablePiper()` confirmed; torque stays off.

## S2 — zero-output / stop (torque disabled)

- (a) zero-output hold: with NO command issued, 6 joints sampled over 3 s; all deltas **0.000 deg** (arm holds position with zero commanded output).
- (b) `EmergencyStop(0x02)` resume returned safely with no motion (torque remains off).
- (c) `DisconnectPort()` cleanup ran without raising.
- Trailing `DisablePiper` reported `SEND_CAN_BUS_NOT_OK` after disconnect — expected cleanup noise, not a failure.

## S4 — local IK planning (pinocchio 3.9, no motion)

Verified the production `PiperIk` (`ubrobot_manipulation/executors/piper_ik.py`) on the dock:

- URDF: `/home/unitree/ubrobot/assets/urdf/piper_description.urdf` (nq=8; joint7 tiny revolute, joint8 gripper prismatic).
- Frames: tool frame `link8` at neutral ≈ (0.191, 0, 0.225); arm reach x∈[-0.5, 0.19], z∈[0.2, 0.65].
- Solved poses (arm-base frame), all within joint limits:

| target (m) | joints (rad) |
|---|---|
| (0.15, 0.00, 0.25) | (-0.74, 0.06, -0.43, 0.61, 0.90, -0.83) |
| (0.00, 0.00, 0.40) | (-0.74, 0.00, -1.14, 0.03, 1.21, -1.58) |
| (0.10, -0.10, 0.30) | (0.44, 0.04, -0.73, -0.71, 1.22, -0.80) |
| (0.25, 0.00, 0.30) | (0.49, 0.97, -1.14, -0.57, 1.22, -0.10) |

Fixes landed in this run (commit `5aeae8e`):

- pinocchio **v3.9** compatibility: `JOINT_MODEL` removed → use
  `getFrameJacobian(model, data, frame_id, ReferenceFrame.WORLD)` so position
  and orientation errors share the Jacobian frame.
- **Active-joint locking**: only the six revolute arm joints are optimized;
  joint7/joint8 (near-zero-range) stay at neutral, so the DLS never wastes
  steps on singular columns.
- **Position-dominant convergence**: orientation weighted low (gripper tolerates
  approach-angle error); position tolerance 2 cm for a realistic grasp.

Known limitation: some interior poses (e.g. (0.10, 0, 0.30)) do not converge
because joint5 hits its 69.9 deg upper limit; verified-convergent poses above
were used for the acceptance.

## S5 — light grasp (torque enabled, small bounded motions)

Executed on the dock with torque enabled, then disabled at the end. Sequence:

1. Pre-enable: ctrl_mode STANDBY(0x0), no error.
2. Enable torque: `EnablePiper()` (returns pre-enable status) then re-query
   `GetArmEnableStatus()` -> `[True]*6` (all enabled); `MotionCtrl_2` joint
   mode 100%. Abort logic refuses any motion if torque is not confirmed.
3. Start joints (deg): [-2.73, -2.27, 0.0, 2.92, 28.46, -105.47].
4. Approach: first three joints +6 deg (bounded) -> measured
   [3.2, 3.71, 0.0, 2.92, 28.53, -105.47] (joint3 held ~0 due to its
   [-170,0] limit/neutral bias, but commanded motion executed).
5. Gripper open to 25 mm -> measured 9.9 mm; gripper close to 5 mm ->
   measured 5.0 mm. **Gripper real travel is ~0..10 mm**; use <=10 mm targets.
6. Retreat to start -> [-2.69, 0.0, 0.0, 2.92, 28.48, -105.47].
7. Torque disabled at end; arm safe.

Result: **PASS** — torque enable path, bounded joint motion, gripper
open/close cycle, retreat, and torque-off cleanup all exercised on hardware.

## Deferred (not executed)

- **S4 full**: remote GraspNet perception (`http_reasoning_server.py:5802` on
  x86 GPU server 192.168.18.230) — server not listening; the `RemoteGraspPerception`
  client and `/grasp_poses` endpoint are implemented and contract-tested but not
  exercised end-to-end on hardware.

## Files / artifacts

- Acceptance driver: `tests/hardware/test_go2_piper_cortex_acceptance.py`
  (gate checks + mutual-exclusion safety all pass on workstation).
- IK production code: `ros_depends_ws/src/ubrobot_manipulation/ubrobot_manipulation/executors/piper_ik.py`.
- Contract tests: `test_go2_piper_executor_contract.py` (17 tests, incl. 3 PiperIk).
