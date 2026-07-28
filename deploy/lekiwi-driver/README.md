# LeKiwi base driver container

This deployment owns the LeKiwi base `ros2_control` runtime. It communicates with
EMOS over ROS 2 DDS using host networking and consumes the standard `/cmd_vel`
topic. The Raspberry Pi host does not need a complete ROS 2 installation.

## Safety state

Only `hardware_mode:=mock` is implemented. The default Compose file has no device
mapping, drops all Linux capabilities, uses a read-only root filesystem, and limits
commands to 0.05 m/s linear and 0.20 rad/s angular velocity with a 250 ms watchdog.
It cannot drive the physical base.

`compose.hardware.yaml` is a deliberately unusable deployment boundary. It maps only
`/dev/lekiwi-base`, but launch rejects `hardware_mode:=real` until the reviewed C++
`LeKiwiSystemHardware` plugin and lifted-wheel test gate are complete. Do not start
the hardware override before that milestone.

## Build and run mock mode

From the repository root on the ARM64 Raspberry Pi:

```bash
docker build \
  -f deploy/lekiwi-driver/Dockerfile \
  -t ubrobot/lekiwi-base-driver:0.1.0-mock \
  .

docker compose -f deploy/lekiwi-driver/compose.yaml up -d
docker compose -f deploy/lekiwi-driver/compose.yaml ps
```

The only supported base is the official `ros:jazzy-ros-base-noble` image. If Docker
Hub is unavailable on the Pi, pull the `linux/arm64/v8` image on a trusted machine,
transfer it with `docker save`/`docker load`, and keep the same image tag.

## Verify

```bash
docker exec lekiwi-base-driver ros2 node list
docker exec lekiwi-base-driver \
  ros2 control list_controllers --controller-manager /controller_manager
docker inspect lekiwi-base-driver --format '{{json .State.Health}}'
```

Expected controllers are `joint_state_broadcaster` and `lekiwi_base_controller`, both
`active`. Verify only zero commands during this phase.

## Stable serial device rule

After independently confirming vendor `1a86`, product `55d3`, and serial
`5A68011386`, install `99-lekiwi-base.rules` as root under `/etc/udev/rules.d/`, reload
udev, and verify `/dev/lekiwi-base` belongs to group `dialout` with mode `0660`.
Installing the rule does not authorize starting hardware mode.
