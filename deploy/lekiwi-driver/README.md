# LeKiwi base driver container

This deployment owns the LeKiwi base `ros2_control` runtime. It communicates with
EMOS over ROS 2 DDS using host networking and consumes the standard `/cmd_vel`
topic. The Raspberry Pi host does not need a complete ROS 2 installation.
Both sides must use `rmw_fastrtps_cpp` with the shared
`deploy/fastdds/udp-only.xml` Fast DDS profile;
the containers use host networking but intentionally do not share an IPC namespace.

## Safety state

The default remains `hardware_mode:=mock`. The default Compose file has no device
mapping, drops all Linux capabilities, uses a read-only root filesystem, and limits
commands to 0.05 m/s linear and 0.20 rad/s angular velocity with a 250 ms watchdog.
It cannot drive the physical base.

`compose.hardware.yaml` maps only `/dev/lekiwi-base` and passes both
`hardware_mode:=real` and the separate `enable_real_hardware:=true` acknowledgement.
It leaves motor torque disabled so the serial bus, IDs, models, and feedback can be
checked without motion. `compose.hardware-torque-test.yaml` is the final hard gate:
it enables torque, disables automatic container restart, and must be added only
after the torque-disabled preflight passes with all wheels lifted.

The base service uses `stop_signal: SIGINT` so ROS 2 launch can shut down
ros2_control in order, deactivate the hardware, and disable motor torque before
closing the serial device.

## Build and run mock mode

From the repository root on the ARM64 Raspberry Pi:

```bash
docker build \
  -f deploy/lekiwi-driver/Dockerfile \
  -t ubrobot/lekiwi-base-driver:0.2.0 \
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

## Lifted-wheel hardware checkpoint

1. Stop EMOS so no non-zero `/cmd_vel` publisher exists.
2. Confirm `/dev/lekiwi-base` resolves to USB serial `5A68011386` and the container
   user can open it.
3. Lift and secure all three wheels, with an operator ready to cut motor power.
4. Start the hardware override together with the default file. This performs the
   real-bus preflight with motor torque disabled:

   ```bash
   docker compose \
     -f deploy/lekiwi-driver/compose.yaml \
     -f deploy/lekiwi-driver/compose.hardware.yaml up
   ```

5. Confirm IDs 8/9/7 map to back/right/left and `/joint_states` remains near zero.
6. Stop the preflight container, then add `compose.hardware-torque-test.yaml` only
   with an operator holding the independent motor-power cutoff.
7. Publish only a small, short command after verifying the zero-command path. If a
   wheel direction is wrong, stop the container and change the corresponding
   `*_direction` parameter in the ros2_control Xacro before continuing.

Never perform the first activation with the robot resting on the floor.
