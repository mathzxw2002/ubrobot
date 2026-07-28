# LeKiwi Driver Container Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deploy an independent ARM64 ROS 2 Jazzy LeKiwi driver container on the Raspberry Pi, initially in hardware-free mock mode, with a safe standard `/cmd_vel` path and production deployment boundaries ready for the C++ Feetech hardware implementation.

**Architecture:** EMOS and the driver remain separate host-networked containers and communicate only through ROS 2 DDS. The driver image owns `controller_manager`, `omni_wheel_drive_controller`, the robot description, command validation, and eventually the C++ `LeKiwiSystemHardware`; mock mode is the mandatory default until real serial read/write is implemented and approved.

**Tech Stack:** Ubuntu 24.04 arm64, Docker Compose, ROS 2 Jazzy, ros2_control 4.x, ros2_controllers 4.x, Python `rclpy` for the temporary Twist adapter, C++17 for the hardware plugin, pytest/ament tests.

---

## Scope and safety gate

This plan deploys the container environment and validates the ROS control chain with
`mock_components/GenericSystem`. It does not send non-zero commands to the physical
LeKiwi. The real hardware Compose override remains unusable until the C++ Feetech
transport is implemented, reviewed, and tested with wheels lifted.

The normal base image is `ros:jazzy-ros-base-noble`. Docker Hub is currently
unreachable from the Raspberry Pi, so the first build may pass the already cached
`ghcr.io/automatika-robotics/emos:jazzy-latest` image through `ROS_BASE_IMAGE` as a
temporary build bootstrap. The Dockerfile default must remain the official ROS image.

### Task 1: Add deployment contract tests

**Files:**
- Create: `tests/lekiwi_driver/test_deployment_contract.py`
- Create: `tests/lekiwi_driver/__init__.py`

**Step 1: Write failing structural tests**

Tests must assert that:

- the Dockerfile defaults to `ros:jazzy-ros-base-noble`;
- Compose uses `network_mode: host` and never uses `privileged: true`;
- mock Compose does not map `/dev` or a serial device;
- the real-hardware override maps only `/dev/lekiwi-base`;
- the default launch argument is `hardware_mode:=mock`;
- the controller timeout is `0.25` seconds and odom TF publishing is disabled;
- the udev rule targets USB serial `5A68011386` and creates `lekiwi-base`.

**Step 2: Run the tests and verify they fail**

Run:

```powershell
python -m pytest tests/lekiwi_driver/test_deployment_contract.py -v
```

Expected: failures because deployment files do not exist.

**Step 3: Commit the failing tests**

```bash
git add tests/lekiwi_driver
git commit -m "test: define LeKiwi container deployment contract"
```

### Task 2: Add the robot description and mock ros2_control system

**Files:**
- Create: `ros_depends_ws/src/lekiwi_description/CMakeLists.txt`
- Create: `ros_depends_ws/src/lekiwi_description/package.xml`
- Create: `ros_depends_ws/src/lekiwi_description/urdf/lekiwi_base.urdf.xacro`
- Create: `ros_depends_ws/src/lekiwi_description/ros2_control/lekiwi_base.ros2_control.xacro`

**Step 1: Define the three wheel joints**

Use these names consistently and list them counter-clockwise for the official omni
controller:

```text
base_back_wheel_joint
base_right_wheel_joint
base_left_wheel_joint
```

Each joint exposes velocity command and velocity state interfaces in mock mode.

**Step 2: Add a safe hardware-mode switch**

The Xacro argument defaults to `mock`. In mock mode load:

```xml
<plugin>mock_components/GenericSystem</plugin>
<param name="calculate_dynamics">true</param>
```

Do not provide a usable `real` plugin yet. An unsupported mode must fail launch rather
than silently fall back to mock.

**Step 3: Add install rules and package dependencies**

Install `urdf/` and `ros2_control/`; depend on `xacro`, `urdf`, and
`hardware_interface`.

**Step 4: Validate Xacro in a Jazzy container**

Run after the image exists:

```bash
ros2 run xacro xacro /opt/lekiwi_ws/share/lekiwi_description/urdf/lekiwi_base.urdf.xacro hardware_mode:=mock
```

Expected: valid URDF containing three velocity command interfaces and
`mock_components/GenericSystem`.

**Step 5: Commit**

```bash
git add ros_depends_ws/src/lekiwi_description
git commit -m "feat: describe LeKiwi base for ros2_control"
```

### Task 3: Add safe command adaptation and bringup

**Files:**
- Create: `ros_depends_ws/src/lekiwi_bringup/package.xml`
- Create: `ros_depends_ws/src/lekiwi_bringup/setup.py`
- Create: `ros_depends_ws/src/lekiwi_bringup/setup.cfg`
- Create: `ros_depends_ws/src/lekiwi_bringup/resource/lekiwi_bringup`
- Create: `ros_depends_ws/src/lekiwi_bringup/lekiwi_bringup/__init__.py`
- Create: `ros_depends_ws/src/lekiwi_bringup/lekiwi_bringup/cmd_vel_adapter.py`
- Create: `ros_depends_ws/src/lekiwi_bringup/test/test_cmd_vel_adapter.py`
- Create: `ros_depends_ws/src/lekiwi_bringup/config/controllers.yaml`
- Create: `ros_depends_ws/src/lekiwi_bringup/launch/lekiwi_driver.launch.py`

**Step 1: Write failing adapter unit tests**

Test a pure `sanitize_velocity(x, y, omega, limits)` helper:

- finite values pass through;
- values are clipped to `0.05`, `0.05`, and `0.20`;
- NaN and infinity produce `(0.0, 0.0, 0.0, false)`;
- negative values retain their sign.

**Step 2: Run the adapter tests and verify failure**

Run inside the driver build environment:

```bash
python3 -m pytest /ws/src/lekiwi_bringup/test/test_cmd_vel_adapter.py -v
```

Expected: failure because `sanitize_velocity` is not implemented.

**Step 3: Implement the adapter**

The ROS node must:

- subscribe to `/cmd_vel` as `geometry_msgs/msg/Twist`;
- publish `geometry_msgs/msg/TwistStamped` to
  `/lekiwi_base_controller/cmd_vel`;
- use the receipt time as the outgoing timestamp;
- validate finite numbers and clamp limits;
- publish zero every watchdog period after 250 ms without a valid command;
- start with zero state and never invent a non-zero command.

The official Jazzy omni controller receives `TwistStamped` on its private
`~/cmd_vel` topic, so the launch file remaps the adapter output to that endpoint.

**Step 4: Configure controllers**

Use:

```yaml
controller_manager:
  ros__parameters:
    update_rate: 20
    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster
    lekiwi_base_controller:
      type: omni_wheel_drive_controller/OmniWheelDriveController

lekiwi_base_controller:
  ros__parameters:
    wheel_names:
      - base_back_wheel_joint
      - base_right_wheel_joint
      - base_left_wheel_joint
    wheel_offset: 0.0
    robot_radius: 0.125
    wheel_radius: 0.05
    open_loop: false
    position_feedback: false
    enable_odom_tf: false
    cmd_vel_timeout: 0.25
```

**Step 5: Implement launch**

Launch robot description, `ros2_control_node`, both controller spawners, and the
adapter. The `hardware_mode` launch argument defaults to `mock`.

**Step 6: Run unit tests**

Expected: all adapter tests pass.

**Step 7: Commit**

```bash
git add ros_depends_ws/src/lekiwi_bringup
git commit -m "feat: add safe LeKiwi ros2_control bringup"
```

### Task 4: Add container and Compose deployment

**Files:**
- Create: `deploy/lekiwi-driver/Dockerfile`
- Create: `deploy/lekiwi-driver/compose.yaml`
- Create: `deploy/lekiwi-driver/compose.hardware.yaml`
- Create: `deploy/lekiwi-driver/entrypoint.sh`
- Create: `deploy/lekiwi-driver/healthcheck.sh`
- Create: `deploy/lekiwi-driver/99-lekiwi-base.rules`
- Create: `deploy/lekiwi-driver/.env.example`
- Create: `deploy/lekiwi-driver/README.md`

**Step 1: Implement a multi-stage Dockerfile**

Default build argument:

```dockerfile
ARG ROS_BASE_IMAGE=ros:jazzy-ros-base-noble
```

Install pinned Jazzy binary packages available for ARM64, including
`ros-jazzy-ros2-control`, `ros-jazzy-ros2-controllers`, and
`ros-jazzy-omni-wheel-drive-controller`. Build only the three LeKiwi ROS packages.

**Step 2: Implement safe default Compose**

The default service:

- uses host networking;
- starts `hardware_mode:=mock`;
- has no device mapping;
- uses a read-only root filesystem and `/tmp` tmpfs;
- sets `ROS_DOMAIN_ID=0` and `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`;
- restarts unless stopped;
- has a ROS-aware health check.

**Step 3: Implement explicit hardware override**

The override maps only:

```text
/dev/lekiwi-base:/dev/lekiwi-base
```

It must not be usable until the real C++ hardware plugin exists; document this hard
gate prominently.

**Step 4: Add udev rule**

Match vendor `1a86`, product `55d3`, serial `5A68011386`, assign `dialout`, mode
`0660`, and symlink `lekiwi-base`.

**Step 5: Run contract tests**

Run:

```powershell
python -m pytest tests/lekiwi_driver/test_deployment_contract.py -v
```

Expected: pass.

**Step 6: Validate Compose**

Run:

```bash
docker compose -f deploy/lekiwi-driver/compose.yaml config
```

Expected: valid configuration, host network, no devices, no privileged mode.

**Step 7: Commit**

```bash
git add deploy/lekiwi-driver tests/lekiwi_driver
git commit -m "feat: package LeKiwi mock driver container"
```

### Task 5: Build the ARM64 image on the Raspberry Pi

**Files:**
- Modify only if required by verified build errors.

**Step 1: Synchronize the feature worktree to the Pi**

Use the existing `/home/china/ubrobot` checkout only after confirming that the files
being replaced are clean. Prefer pushing the feature branch and pulling when registry
access is available; for this deployment, copy only the new LeKiwi package and deploy
directories.

**Step 2: Build with the official default**

```bash
docker build -f deploy/lekiwi-driver/Dockerfile \
  -t ubrobot/lekiwi-base-driver:0.1.0-mock .
```

If Docker Hub remains unreachable, repeat with the documented temporary bootstrap:

```bash
docker build -f deploy/lekiwi-driver/Dockerfile \
  --build-arg ROS_BASE_IMAGE=ghcr.io/automatika-robotics/emos:jazzy-latest \
  -t ubrobot/lekiwi-base-driver:0.1.0-mock .
```

Expected: successful `linux/arm64` image with the LeKiwi workspace installed.

**Step 3: Inspect the image**

Verify architecture, labels, entrypoint, installed packages, and that no serial device
is embedded or required for mock startup.

### Task 6: Deploy and verify mock mode alongside EMOS

**Files:**
- Raspberry Pi runtime files under `/home/china/ubrobot/deploy/lekiwi-driver/`.

**Step 1: Start the default Compose service**

```bash
docker compose -f deploy/lekiwi-driver/compose.yaml up -d
```

Expected: `lekiwi-base-driver` becomes healthy without access to `/dev/lekiwi-base`.

**Step 2: Verify ROS components**

Verify:

```text
/controller_manager
/lekiwi_cmd_adapter
/lekiwi_base_controller
/joint_state_broadcaster
```

Both controllers must be active and the hardware component must report active mock
interfaces.

**Step 3: Verify EMOS connection without motion**

Confirm `/cmd_vel` now has a subscriber. Do not start a tracking action and do not
publish a non-zero command.

**Step 4: Verify watchdog behavior**

Publish one zero Twist, stop publishing, and confirm the adapter and controller remain
at zero after 250 ms.

**Step 5: Verify independent lifecycle**

Restart the EMOS container and confirm the driver container stays running. Restart the
driver container and confirm EMOS stays running.

### Task 7: Install but do not activate stable device naming

**Files:**
- Install: `/etc/udev/rules.d/99-lekiwi-base.rules`

**Step 1: Review the exact device identity**

Confirm vendor, product, and serial from `udevadm info` before installation.

**Step 2: Install the rule and reload udev**

Install with root ownership and mode `0644`, reload rules, and trigger only the matching
device.

**Step 3: Verify permissions**

Expected:

```text
/dev/lekiwi-base -> ttyACM*
owner root, group dialout, mode 0660
```

Do not start the hardware Compose override.

### Task 8: Document deployment results and checkpoint

**Files:**
- Modify: `emos.md`
- Modify: `deploy/lekiwi-driver/README.md`
- Create: `docs/adr/0003-temporary-emos-base-image-bootstrap.md` only if the fallback image was actually used.

**Step 1: Record exact versions and verification output**

Document image ID, ROS package versions, Raspberry Pi architecture, Compose state, ROS
node/controller state, and the Docker Hub connectivity exception if applicable.

**Step 2: State the remaining hard gate**

Make clear that real hardware mode remains disabled until the C++ Feetech transport,
hardware watchdog strategy, lifted-wheel tests, and explicit approval are complete.

**Step 3: Run final checks**

Run contract tests, ROS package tests, Compose validation, image inspection, mock ROS
graph verification, and `git diff --check`.

**Step 4: Commit**

```bash
git add emos.md deploy/lekiwi-driver/README.md docs/adr
git commit -m "docs: record LeKiwi mock driver deployment"
```

