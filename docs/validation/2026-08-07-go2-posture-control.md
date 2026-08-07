# Go2 Posture Control (Stand/Sit) Validation Report

- Date/time: 2026-08-07
- Commits: `9000820` (containerized Go2 posture control) on `go2-piper-cortex-integration`
- Machine role: Go2 expansion dock (Jetson Orin NX, Ubuntu 20.04.5, JetPack R35.3.1)
- Image: `ubrobot/go2-piper-driver:0.1.0` (`f7fa7bcd0f4b` → `320b1b8bbfe3` → final)
- Container: `go2-piper-driver` (systemd-managed, `rmw_cyclonedds_cpp`)
- Hardware authority: **false**; Piper torque disabled; Go2 posture only, **no movement**
- Physical effect: **confirmed by operator observation**

## Objective

Verify the Go2 control chain end to end with **low-risk posture primitives only**
(StandUp / StandDown), given the Go2 base is carrying a heavy load and must not
travel. Command path: operator → `/go2/stand` service (container) → isolated
subprocess → unitree_sdk2py `SportClient` → Unitree DDS (CycloneDDS) → Go2 body.

## Summary

| Step | Result |
|---|---|
| Container cyclonedds integration | **PASS** — host CycloneDDS 0.10.2 build + libssl1.1 staged into image |
| `SportClient` Init (container, no rclpy) | **PASS** — Init OK |
| `go2_bridge_node` stability | **PASS** — no segfault after process-isolation refactor |
| `/go2/stand` STAND UP | **PASS** — `success=True`; Go2 physically stood up |
| `/go2/stand` SIT DOWN | **PASS** — `success=True`; Go2 physically sat down |
| No movement during test | **PASS** — posture only, base stationary |

## Key problems found and fixed

### 1. Container missing the `cyclonedds` Python package
`unitree_sdk2py`'s `SportClient` imports `cyclonedds`; the image had only the RMW
C++ impl (`ros-jazzy-rmw-cyclonedds-cpp`), no Python binding.

- `pip install cyclonedds` failed: needs a CycloneDDS C library; Ubuntu Noble has
  no `libcyclonedds-dev` (apt only carries legacy focal-era `ros-rolling-cyclonedds` 0.8.0).
- **Fix**: staged the host source build of CycloneDDS **0.10.2**
  (`/home/unitree/cyclonedds_ws/install/cyclonedds`) into the build context
  (`.dock-build/cyclonedds`) and compiled `cyclonedds==0.10.2 --no-build-isolation`
  against it (`CMAKE_PREFIX_PATH`/`CYCLONEDDS_HOME`).
- **Fix**: `libddsc.so` links OpenSSL 1.1 (`libssl.so.1.1`/`libcrypto.so.1.1`),
  absent from Noble (only OpenSSL 3). Copied the host `libssl1.1` pair into the
  staged `cyclonedds/lib`.

### 2. In-process rclpy + unitree SDK segfault
Constructing a `SportClient` inside the `go2_bridge_node` process (which already
runs `rclpy`/`rmw_cyclonedds_cpp`) **segfaults** (`exit code -11`): the unitree
`cyclonedds` Python participant conflicts with the RMW CycloneDDS participant in
the same process.

- **Fix**: `/go2/stand` no longer builds a SportClient in-process. It runs the
  standalone `go2_stand_cli.py` (no rclpy, only unitree_sdk2py + cyclonedds) via
  `subprocess`. `/cmd_vel` forwarding from the ROS node is intentionally disabled
  for the same reason (guarded velocity path is separate).

### 3. Posture read-back not available in this stack
`ChannelSubscriber` on `rt/sportmodestate` fails with
`TypeError: rt/sportmodestate is not an idl type` — the SDK's IDL message
(`SportModeState_`) is a custom dataclass, not a `cyclonedds` `IdlStruct`, so the
cyclonedds-based subscriber cannot decode it. Command sending works (SDK's own DDS
wrapper); posture read-back is unavailable in this SDK/cyclonedds pairing.
Physical effect was therefore confirmed by operator observation.

## Validation evidence

- `ros2 service call /go2/stand std_srvs/srv/SetBool "{data: true}"` →
  `SetBool_Response(success=True, message='ok')`
- `ros2 service call /go2/stand ... "{data: false}"` →
  `SetBool_Response(success=True, message='ok')`
- bridge logs: `go2 STAND UP (subprocess)` / `go2 SIT DOWN (subprocess)`
- Operator observed the Go2 physically stand up and sit down.

## Architecture constraint (documented)

- **Process isolation is required**: the unitree SDK (`SportClient` +
  `cyclonedds` Python) must run in a separate process from any rclpy node, or it
  segfaults. `/go2/stand` shells out to `go2_stand_cli.py`.

## Files / artifacts

- `deploy/go2-piper-driver/go2_piper_driver/go2_piper_driver/go2_stand_cli.py`
- `deploy/go2-piper-driver/go2_piper_driver/go2_piper_driver/go2_bridge_node.py`
  (process-isolated `/go2/stand`; no in-process SportClient)
- `deploy/go2-piper-driver/Dockerfile` (staged CycloneDDS 0.10.2 + libssl1.1 +
  `pip install cyclonedds==0.10.2`)
- `deploy/go2-piper-driver/README.md` (staging + process-isolation notes)

## Next steps

- Build emos overlay (CycloneDDS) and verify RMW interop with go2-piper-driver
  (`/piper/*`, `/go2/*` topics across containers).
- End-to-end grasp (IK → `/piper/joint_cmd` → real motion), remote GraspNet
  perception still required.
