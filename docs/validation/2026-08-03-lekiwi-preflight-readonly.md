# LeKiwi Lifted-Wheel Preflight + Robot Edge Read-only Stack Validation (M7)

- Date/time: 2026-08-03 09:30–10:30 (+08:00)
- Commits: `e05583e` (odometry topic), `d5eeb86` (camera topic),
  `0bdeaf1` (rclpy slots), `28241ce` (array + covariance unwrap),
  `8542d54`/`d731959` (edge image build fixes), `4b65b4a` (safety decision)
- Machine role: robot-side host (Raspberry Pi 5, Ubuntu 24.04, Docker) +
  workstation for code/test
- Execution mode: **hardware mode, torque disabled** for LeKiwi; Robot Edge
  hardware mode with `hardware_authority=false` (read-only)
- Mobile-base profile: **`lekiwi`** (owner-selected)
- Hardware authority: **false** everywhere
- Physical E-stop: **absent by owner decision** (2026-08-03 ADR-0002);
  final cutoff is the operator pulling the power cable

## 1. Lifted-wheel preflight (torque disabled)

Per `deploy/lekiwi-driver/README.md` lifted-wheel checkpoint, with owner
confirmation: three wheels lifted and secured, power cable reachable.

Container recreated from the hard-gate configuration
(`enable_motor_torque:=true`) to **torque-disabled real mode**:

```text
hardware_mode:=real  enable_real_hardware:=true   (no enable_motor_torque)
```

| Check | Result |
|---|---|
| Container health | **healthy** |
| Controllers | `lekiwi_base_controller` + `joint_state_broadcaster` both **active** |
| Driver log | **"LeKiwi bus active in torque-disabled preflight mode"** |
| `/joint_states` | 3 joints (back/left/right), positions ~0 (≤0.24 rad), velocities 0 |
| `/lekiwi_base_controller/odom` | publishing, x/y near zero, yaw from quaternion |
| Motor IDs | 8/9/7 → model 777 (previously verified 2026-07-28); controller activation confirms enumeration |

No wheel movement observed. No torque enabled.

## 2. Findings fixed during preflight

Measured live topics differed from the design assumptions; adapters updated
and committed:

1. **Odometry topic**: `/odom/wheel` (design) → actual
   `/lekiwi_base_controller/odom` (ros2_control controller namespace).
2. **Camera topic**: `/camera/camera_info` (design) → actual
   `/camera/camera/color/camera_info` (RealSense under double
   `/camera/camera/...` namespace).
3. **rclpy message extraction**: generated messages declare slots with
   private names (`_pose`) and no `__dict__`; `_json_safe` now strips the
   underscore and reads public properties.
4. **rclpy arrays**: `float64[]` fields are `array.array`, not `list`;
   normalized to lists.
5. **PoseWithCovariance nesting**: live `Odometry` serializes
   `pose.pose.position` / `twist.twist.linear`; extractors unwrap one level.
6. **Edge container build**: `python3-pip` absent from ros:jazzy base; PEP
   668 on Noble → dedicated venv with `--system-site-packages`; rclpy logs
   redirected to tmpfs (`ROS_HOME/ROS_LOG_DIR`).

Workstation regression suite after fixes: `tests/robot_edge` 128 PASS,
`tests/cortex_navigation` 185 PASS, E2E fixture 8 PASS.

## 3. Robot Edge hardware read-only validation

Stack on the Pi (all host network, ROS domain 0, UDP-only Fast DDS):

- `lekiwi-base-driver` (torque disabled) — above
- `emos-nav-readonly` from `ubrobot/emos:e2e-2c83d27` (the image with the
  navigation stack; `jazzy-7a64982` predates it): `cortex_navigation_bringup`
  with `start_sensors:=true`, **no recipe, no /cmd_vel publisher**
- `robot-edge` from `ubrobot/robot-edge:m7-20260803`
  (`compose.ros-readonly.yaml`): `execution_mode=hardware`,
  `hardware_authority=false`

### Readiness

```json
{"status":"ready","execution_mode":"hardware","hardware_authority":false,"local_stop":{"bound":false}}
```

### Telemetry snapshot (live ROS graph, truthful values)

| Channel | State | Value |
|---|---|---|
| odometry | available | x=-0.0069, y=0.0041, yaw=-0.0322, vx=0.0 (from `/lekiwi_base_controller/odom`) |
| joint_states | available | names [back, left, right], positions [0.0, 0.238, 0.0] |
| camera | available | 640×480, distortion plumb_bob (from `/camera/camera/color/camera_info`) |
| depth | available | 640×480 |
| navigation_lease | unavailable | M6 read-only: Edge local state not tracked |
| capability_health | unavailable | M6 read-only: Edge local state not tracked |

### Capabilities (from real graph)

| Capability | Availability | Health | Note |
|---|---|---|---|
| navigation | available | healthy | `/ubrobot/navigation/navigate_to_object` action server present |
| observation | available | healthy | camera_info topics present |
| follow | available | healthy | lekiwi odom topic present |
| stop | available | healthy | |
| grasp | unavailable | unknown | no CAN / Piper not connected — truthful |

## 4. Acceptance

- [x] Preflight (torque-disabled real mode) passes: healthy, controllers
  active, joint states near zero, no motion, no torque.
- [x] Read-only Robot Edge serves truthful live telemetry from the real ROS
  graph with the corrected topic names.
- [x] Capability inventory reflects the real graph (grasp correctly
  unavailable without Piper).
- [x] Command authority remains false; all command endpoints reject.
- [x] Import boundaries hold; rclpy only inside the hardware factory.
- [ ] Motion trials (Task 13) — **blocked** by design: torque remains
  disabled; requires the owner-approved lifted-wheel/cleared-area session
  with the power cable as the final cutoff.

## 5. Known limitations

1. `emos-nav-readonly` uses the `e2e-2c83d27` image because the current
   `jazzy-7a64982` image predates the navigation stack; a current-image
   rebuild is pending.
2. RealSense serial mismatch from the inventory is unresolved (config only;
   the health reader is serial-agnostic).
3. `navigation_lease`/`capability_health` channels are unavailable by design
   in M6 read-only mode; they become live when lease tracking lands in the
   hardware backend (M7 Task 13 phase).
4. Physical E-stop is not implemented (owner decision); power-cable pull is
   the final cutoff and no latency measurement was performed.
