# Real Kompass Vision Navigation Integration Validation (M7 upper layer)

- Date/time: 2026-08-03 13:30-14:45 (+08:00)
- Commits: a194b35 (float32 timestamp patch), 528472a (control time step),
  d4c1409/f68b030 (recipe build fixes), 7861361 (vision mode config)
- Planner: Volcengine ARK glm-5-2-260617 (via relay :18081)
- Detection: RoboML roboml-resp on vlm_server (192.168.18.230:6379),
  rtdetr_r50vd_coco_o365
- Hardware authority: false; LeKiwi torque disabled; wheels lifted; no motion

## Stack

| Container | Role |
|---|---|
| lekiwi-base-driver (0.2.0-rc1-m7-20260803) | torque-disabled driver, BEST_EFFORT cmd_vel adapter |
| emos-nav-readonly (e2e-2c83d27) | cortex_navigation_bringup (NavigateToObject server, cmd_vel_guard, RealSense chain) |
| emos-cortex-recipe (m3-8bd4be9 + mounted patched recipe) | **full recipe**: Cortex + ARK + RoboML detection + Kompass vision stack |
| roboml-resp (vlm_server) | rtdetr object detection service |

## Full chain (real components, no fixture)

```
请走到椅子旁边 -> ARK glm-5 plan -> NavigateToObject
  -> TrackVisionTarget (REAL Kompass Vision)
  -> RoboML rtdetr detection: chair 91.7%
  -> Kompass VISION_DEPTH controller
  -> /navigation/raw_cmd_vel -> cmd_vel_guard (lease) -> /cmd_vel
  -> lekiwi adapter -> /lekiwi_base_controller/cmd_vel -> controller (no torque)
```

### Evidence

- Plan: send_goal_to__ubrobot_navigation_navigate_to_object
- Detection: labels=[chair], scores=[0.9165]
- /cmd_vel non-zero: 140 samples with REAL varying commands
  (0.05, 0.0, 0.1246) -> (0.05, 0.0, 0.1295) -> ... (angular adjusts as the
  controller tracks the detected chair)
- Adapter output: 139 synchronized samples at /lekiwi_base_controller/cmd_vel

## Root cause found and fixed: float32 timestamp collapse

- Kompass C++ FeatureBasedBboxTracker computes time_step as the difference of
  Bbox2D.timestamp (C++ float32, epoch seconds).
- At ~1.79e9 s the float32 ULP is 128 s; every 15 Hz detection frame rounds
  to the same value -> time_step <= 0 -> "Box updated with invalid time
  step, Velocity wil be reset to zero" forever.
- Verified: float32(1785736766.349) = 1785736704.0.
- Fix: kompass detection timestamps now use time.monotonic() (small numbers,
  float32 precision ~1 ms) — sufficient because the tracker only compares
  consecutive detections in one process. Patched in the container and
  permanently in deploy/emos/Dockerfile (a194b35).

## Issues fixed along the way

1. Kompass algorithm setter crashes at build time (global rclpy.ok() check
   before node init) -> set config values only (7861361).
2. ControllerMode not exported at top level -> import from
   kompass.components._modes (d4c1409).
3. m3 image lacks GraspObject -> optional import (d4c1409).
4. control_time_step 0.3 vs loop_rate 10 Hz -> aligned to 0.1 (528472a).
5. Component subprocesses need AMENT_PREFIX_PATH when entrypoint overridden.
6. float32 timestamp collapse (a194b35) — the blocking issue.


## Production image upgrade (jazzy-m7-20260803)

Both emos containers were upgraded to the rebuilt production image
(no recipe mounts, no container-layer patches):

- bringup + recipe run entirely from the image.
- Verified patches inside the image: kompass monotonic timestamps,
  recipe config-mode vision setup, cortex tool-args, grasp optional.
- New finding fixed during upgrade: the recipe container must mount the
  shared Fast DDS udp-only profile; without it DDS discovery inside the
  recipe container did not deliver /scan (CriticalZoneChecker never
  initialized). With the mount: "CriticalZoneChecker is READY!".
- Full chain re-validated on the production image: 166 non-zero varying
  commands at /cmd_vel and 166 synchronized at the driver adapter.

## Acceptance

- [x] Real Kompass (not the fixture) serves /track_vision_target.
- [x] Real object detection (RoboML rtdetr) drives navigation.
- [x] Real varying velocity commands reach /cmd_vel and the driver adapter.
- [x] No motion (torque disabled), no hardware authority.
- [x] Root cause documented and patched persistently in the Dockerfile.

## Limitations / next steps

1. emos image rebuild (with the patches) still pending on the Pi; the
   running container carries the container-layer patch (lost on ).
2. Real robot motion still requires torque enablement + owner-authorized
   cleared-area session (M7 Task 13).
3. Kompass RobotConfig geometry (radius 0.18 m) should be verified against
   the real LeKiwi chassis before motion.
