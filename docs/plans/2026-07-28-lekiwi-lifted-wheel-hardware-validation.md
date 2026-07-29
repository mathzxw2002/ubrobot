# LeKiwi Lifted-Wheel Hardware Validation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Safely validate the LeKiwi STS3215 base driver on a Raspberry Pi with all wheels mechanically secured off the floor, without allowing EMOS or an unexpected container restart to command motion.

**Architecture:** Use three explicit stages: candidate-image mock baseline, real serial bus with motor torque disabled, and a separate one-shot torque-enabled test override. The real hardware plugin must default to torque disabled, force zero commands while disabled, and require a second launch acknowledgement before torque can be enabled. EMOS remains stopped for the entire first hardware-validation session.

**Tech Stack:** ROS 2 Jazzy, ros2_control, Docker Compose, Fast DDS UDP-only, Feetech STS3215 serial protocol, Raspberry Pi ARM64.

---

## Current state and decision

- Candidate commit: `3005610e1bf1b2a4c2665188a34f4e7ac548ffdc`.
- Candidate image: `ubrobot/lekiwi-base-driver:0.2.0-rc1-3005610`.
- Candidate image digest: `sha256:ce4060c9f8a614bb8eee2439de29209d218e4349152d122e6887ecf24f4e5655`.
- The Raspberry Pi sees USB device `1a86:55d3`, but `/etc/udev/rules.d/99-lekiwi-base.rules` is not installed and `/dev/lekiwi-base` is missing.
- The current plugin calls `enable_torque()` from `on_activate()`. Starting current real mode therefore enables torque immediately after configuration.
- The current hardware clamp is `max_raw_velocity=3000`, which is unnecessarily high for the first lifted-wheel test.
- The running formal container remains `0.2.0` mock, healthy, with no device mapping.

### Considered approaches

1. **Recommended: add a second torque gate.** Real mode can identify motors and read feedback while torque remains disabled. A separate test-only Compose override enables torque with restart disabled and a conservative raw velocity ceiling.
2. Start `3005610` real mode directly. This is faster, but bus identification and first torque activation happen in one operation, leaving no review checkpoint.
3. Use a vendor utility for bus identification. This avoids a driver change but validates a different protocol path and can leave servo registers in an unknown state.

Proceed with approach 1. Do not start the existing candidate in real mode.

## Global stop conditions

Stop immediately and cut motor power if any of the following occurs:

- the chassis is not rigidly secured with all three wheels clear of obstacles;
- no second person is available at the independent motor-power cutoff;
- USB vendor, product, or serial differs from `1a86`, `55d3`, or `5A68011386`;
- either `/cmd_vel` or `/lekiwi_base_controller/cmd_vel` has an unexpected publisher;
- any wheel moves during the torque-disabled preflight;
- any motor is not model `777` or IDs `8`, `9`, and `7` are not all reachable;
- feedback is non-finite, stale, or inconsistent with the commanded wheel;
- a command does not return to zero within one second;
- the driver restarts automatically during a failure test;
- there is unexpected noise, vibration, heating, cable movement, or chassis instability.

Do not test USB disconnect or `SIGKILL` while a non-zero command is active. Software cannot transmit a stop command after losing the serial link.

### Task 1: Add a separate motor-torque safety gate

**Files:**

- Modify: `ros_depends_ws/src/lekiwi_bringup/launch/lekiwi_driver.launch.py`
- Modify: `ros_depends_ws/src/lekiwi_description/urdf/lekiwi_base.urdf.xacro`
- Modify: `ros_depends_ws/src/lekiwi_description/ros2_control/lekiwi_base.ros2_control.xacro`
- Modify: `ros_depends_ws/src/lekiwi_hardware/include/lekiwi_hardware/lekiwi_system_hardware.hpp`
- Modify: `ros_depends_ws/src/lekiwi_hardware/src/lekiwi_system_hardware.cpp`
- Modify: `deploy/lekiwi-driver/compose.hardware.yaml`
- Create: `deploy/lekiwi-driver/compose.hardware-torque-test.yaml`
- Modify: `tests/lekiwi_driver/test_deployment_contract.py`

**Step 1: Add failing deployment-contract assertions**

Assert all of the following:

- launch argument `enable_motor_torque` defaults to `false`;
- `hardware_mode:=real` still requires `enable_real_hardware:=true`;
- the Xacro passes `enable_motor_torque` to the plugin;
- the plugin parameter defaults to false and is parsed strictly as `true` or `false`;
- `compose.hardware.yaml` does not enable torque;
- `compose.hardware-torque-test.yaml` sets `enable_motor_torque:=true` and `restart: "no"`;
- first-test `max_raw_velocity` is `300`, approximately `0.46 rad/s` at the motor.

Run:

```bash
python tests/lekiwi_driver/test_deployment_contract.py -v
```

Expected: the new assertions fail before implementation.

**Step 2: Pass the gate through launch and Xacro**

Declare `enable_motor_torque:=false` in the launch file, pass it to the top-level URDF Xacro, then pass it to the ros2_control macro and hardware parameter. Reject `enable_motor_torque:=true` unless both `hardware_mode:=real` and `enable_real_hardware:=true` are present.

Change the first-test clamp in the ros2_control Xacro:

```xml
<param name="max_raw_velocity">300</param>
<param name="enable_motor_torque">${enable_motor_torque}</param>
```

**Step 3: Make torque disabled the plugin default**

Add `bool enable_motor_torque_{false};` to `LeKiwiSystemHardware`. Parse only literal `true` and `false` values during `on_init()`.

In `on_activate()`:

```cpp
std::fill(velocity_commands_.begin(), velocity_commands_.end(), 0.0);
if (enable_motor_torque_) {
  bus_.enable_torque();
  RCLCPP_WARN(get_logger(), "LeKiwi motor torque ENABLED with zero command");
} else {
  bus_.stop_and_disable();
  RCLCPP_INFO(get_logger(), "LeKiwi bus active in torque-disabled preflight mode");
}
```

In `write()`, ignore controller commands while torque is disabled and transmit only raw zero values. This prevents a latent non-zero goal from being stored before a later restart with torque enabled.

**Step 4: Add the test-only torque override**

`compose.hardware.yaml` maps `/dev/lekiwi-base` and selects real hardware, but does not enable torque. Create `compose.hardware-torque-test.yaml` containing only:

```yaml
services:
  lekiwi-base-driver:
    restart: "no"
    command:
      - ros2
      - launch
      - lekiwi_bringup
      - lekiwi_driver.launch.py
      - hardware_mode:=real
      - enable_real_hardware:=true
      - enable_motor_torque:=true
```

**Step 5: Run tests and commit the safety hardening**

```bash
python tests/lekiwi_driver/test_deployment_contract.py -v
docker compose -f deploy/lekiwi-driver/compose.yaml config --quiet
docker compose -f deploy/lekiwi-driver/compose.yaml \
  -f deploy/lekiwi-driver/compose.hardware.yaml config --quiet
docker compose -f deploy/lekiwi-driver/compose.yaml \
  -f deploy/lekiwi-driver/compose.hardware.yaml \
  -f deploy/lekiwi-driver/compose.hardware-torque-test.yaml config --quiet
git diff --check
```

Expected: all tests pass and the default Compose still has no device mapping.

Commit only the safety-gate files. Do not include the existing user changes to `src/ubrobot/robots/lekiwi/lekiwi_base.py` or `docs/embodied_ai_stack_analysis.html`.

### Task 2: Build a new commit-addressed candidate image

**Files:** None. Build only from the new Git commit archive.

**Step 1: Create a Git archive and verify its SHA-256 on both machines**

Use the same isolated archive workflow used for `3005610`. Do not build from `/home/china/ubrobot`, because that worktree contains uncommitted files.

**Step 2: Build with immutable provenance labels**

```bash
docker build \
  --label org.opencontainers.image.revision=<full-new-sha> \
  --label org.opencontainers.image.version=0.2.0-rc1-<short-new-sha> \
  -f deploy/lekiwi-driver/Dockerfile \
  -t ubrobot/lekiwi-base-driver:0.2.0-rc1-<short-new-sha> \
  .
```

Expected: all three ROS packages build; no build error is accepted.

**Step 3: Run isolated mock smoke validation**

Start the image with `ROS_DOMAIN_ID=99`, no devices, `Privileged=false`, and a read-only root filesystem. Run `/usr/local/bin/lekiwi-healthcheck.sh` and verify both controllers are active. Remove only the temporary smoke container afterward.

### Task 3: Establish the physical and operator safety gate

**Files:** None.

**Step 1: Physically secure the robot**

- Disconnect motor power while positioning the robot.
- Lift and rigidly secure the chassis so all wheels rotate freely.
- Keep hair, clothing, tools, and cables outside every wheel envelope.
- Assign one operator to the terminal and another to the independent motor-power cutoff.
- Keep the cutoff operator in control for every torque-enabled step.

**Step 2: Prepare abort commands in a dedicated terminal**

```bash
docker stop -t 1 lekiwi-base-driver
```

The physical motor-power cutoff is authoritative. The Docker command is secondary.

**Step 3: Stop EMOS and verify no command publishers**

```bash
docker stop emos
docker exec lekiwi-base-driver bash -lc \
  'source /opt/lekiwi_ws/setup.bash && ros2 topic info /cmd_vel -v'
docker exec lekiwi-base-driver bash -lc \
  'source /opt/lekiwi_ws/setup.bash && ros2 topic info /lekiwi_base_controller/cmd_vel -v'
```

Expected: publisher count is zero on both topics. If either count is non-zero, stop and identify the publisher.

### Task 4: Install and verify the stable serial-device rule

**Files:**

- Source: `deploy/lekiwi-driver/99-lekiwi-base.rules`
- Install: `/etc/udev/rules.d/99-lekiwi-base.rules`

**Step 1: Back up any existing rule and install the candidate rule**

```bash
sudo install -m 0644 deploy/lekiwi-driver/99-lekiwi-base.rules \
  /etc/udev/rules.d/99-lekiwi-base.rules
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=tty
```

**Step 2: Verify immutable USB identity**

```bash
ls -l /dev/lekiwi-base
udevadm info --query=property --name=/dev/lekiwi-base | \
  grep -E '^(ID_VENDOR_ID|ID_MODEL_ID|ID_SERIAL_SHORT|DEVNAME)='
stat -Lc 'mode=%a owner=%U group=%G' /dev/lekiwi-base
```

Expected:

- vendor `1a86`;
- product `55d3`;
- serial `5A68011386`;
- group `dialout`;
- mode `660`;
- the resolved target is one specific `ttyUSB*` or `ttyACM*` device.

If the serial does not match, do not weaken the udev rule; stop and inspect the physical adapter.

### Task 5: Capture a candidate-image mock baseline

**Files:** None.

**Step 1: Switch the formal container to the new candidate in mock mode**

```bash
LEKIWI_DRIVER_TAG=0.2.0-rc1-<short-new-sha> \
docker compose -f deploy/lekiwi-driver/compose.yaml up -d --no-build --force-recreate
```

Expected: healthy, `Devices=null`, and both controllers active.

**Step 2: Record mock wheel signatures**

Publish one short command for each case with at least five seconds between cases:

- `linear.x=0.01`;
- `linear.y=0.01`;
- `angular.z=0.05`.

Record `/joint_states` names and velocity signs while active, then verify all velocities return to zero within one second. These mock signatures are the comparison oracle for the real-wheel direction test; do not rely on visual memory.

### Task 6: Run the real bus preflight with torque disabled

**Files:** None.

**Step 1: Confirm motor power is available and wheels remain secured**

The cutoff operator powers the motor rail only after both operators confirm the mechanical setup.

**Step 2: Start real mode without the torque-test override**

```bash
LEKIWI_DRIVER_TAG=0.2.0-rc1-<short-new-sha> \
docker compose \
  -f deploy/lekiwi-driver/compose.yaml \
  -f deploy/lekiwi-driver/compose.hardware.yaml \
  up -d --no-build --force-recreate
```

Expected logs:

- serial port opened at `1000000` baud;
- motors `8`, `9`, and `7` respond as model `777`;
- velocity mode configured;
- explicit `torque-disabled preflight mode` message;
- no wheel movement for at least 30 seconds.

**Step 3: Validate feedback and disabled-command behavior**

- Capture ten `/joint_states` samples; all values must be finite. At rest, accept only
  zero or the observed STS3215 minimum quantized feedback of `±0.076699 rad/s`, with
  absolute velocity below `0.08 rad/s` and no visible wheel motion.
- Publish each mock-baseline command once. Wheels must not move because torque is disabled.
- Confirm commands return to zero and no hardware error appears.

Any wheel motion in this task is an unconditional failure.

### Task 7: Perform first zero-command torque activation

**Files:** None.

**Step 1: Recheck all gates immediately before activation**

- EMOS stopped;
- zero publishers on both command topics;
- wheels clear and secured;
- cutoff operator ready;
- abort terminal ready;
- torque-disabled preflight passed without warnings.

**Step 2: Start the torque-test override**

```bash
LEKIWI_DRIVER_TAG=0.2.0-rc1-<short-new-sha> \
docker compose \
  -f deploy/lekiwi-driver/compose.yaml \
  -f deploy/lekiwi-driver/compose.hardware.yaml \
  -f deploy/lekiwi-driver/compose.hardware-torque-test.yaml \
  up -d --no-build --force-recreate
```

Expected:

- `restart: no` in `docker inspect`;
- one explicit `motor torque ENABLED with zero command` log;
- no visible wheel rotation for 30 seconds;
- measured absolute wheel velocity below `0.08 rad/s` at rest;
- container remains running and controllers remain active.

If any wheel moves, cut motor power first, then stop the container.

### Task 8: Run short direction and watchdog tests

**Files:** None.

**Step 1: Test one axis at a time**

Publish a single `/cmd_vel` sample for each case, separated by a complete stop and five-second observation interval:

```text
linear.x = 0.01 m/s
linear.y = 0.01 m/s
angular.z = 0.05 rad/s
```

If a single sample produces valid feedback but the marked wheels do not move far enough
for reliable visual confirmation, repeat at the same velocity using at most 10 samples at
10 Hz (one second total). Do not increase velocity. Keep the five-second stopped observation
interval between axes and re-verify the watchdog after the final sample.

Use `/cmd_vel`, never publish directly to the controller input. Record video and `/joint_states` simultaneously.

**Step 2: Compare against mock signatures**

Pass only if:

- the same named wheels have the same velocity signs as the mock baseline;
- no uninvolved wheel shows unexpected sustained motion;
- feedback magnitude is plausible and finite;
- there is no oscillation or acceleration after the command ends.

Do not fix a direction mismatch live. Stop, power off, record the required `*_direction` change, update code, rebuild a new commit-addressed image, and repeat from the torque-disabled preflight.

**Step 3: Verify the 250 ms watchdog**

After each single command, verify all wheel velocities return below `0.08 rad/s` within one second. Failure requires immediate power cutoff and ends the test session.

### Task 9: Test safe shutdown paths at zero velocity

**Files:** None.

**Step 1: Graceful container stop**

With all measured velocities at zero:

```bash
docker stop -t 3 lekiwi-base-driver
```

Expected: wheels remain stopped and torque is disabled before serial disconnect.

**Step 2: Serial disconnect only at zero**

Restart torque-enabled mode, issue no motion command, verify zero, then unplug USB serial. Expected: controller reports a read/write error and the container does not automatically restart because the test override uses `restart: no`. Cut motor power immediately if any wheel moves.

Do not perform a moving serial-disconnect test until a servo-side communication timeout has been independently configured and verified.

**Step 3: Motor-power interruption at zero**

With the container stopped and velocity zero, remove motor power. Reconnect USB and motor power only after confirming the driver has not restarted.

### Task 10: Restore mock mode and record the result

**Files:**

- Create after execution: `docs/validation/2026-07-28-lekiwi-lifted-wheel-results.md`

**Step 1: Restore the candidate image in mock mode**

```bash
LEKIWI_DRIVER_TAG=0.2.0-rc1-<short-new-sha> \
docker compose -f deploy/lekiwi-driver/compose.yaml up -d --no-build --force-recreate
```

Expected: healthy and no device mapping. Keep EMOS stopped until the operator explicitly approves the next integration phase.

**Step 2: Record evidence**

Record:

- Git SHA, image ID, and image digest;
- USB identity and resolved device;
- motor ID/model results;
- mock and real joint-name/sign tables;
- watchdog timing;
- graceful-stop and zero-speed serial-disconnect results;
- all warnings, direction changes, and failed gates;
- operator names and confirmation that wheels remained off the floor.

**Step 3: Decide the next candidate state**

- If every gate passes, commit the validation record and schedule a separate EMOS integration test while still lifted.
- If any gate fails, keep formal deployment in mock mode, open a focused correction commit, rebuild a new SHA-tagged candidate, and repeat from Task 5.
