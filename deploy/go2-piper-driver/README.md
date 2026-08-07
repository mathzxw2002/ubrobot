# go2-piper-driver

One-robot-one-container hardware driver: Go2 bridge + Piper arm CAN.

## Contents

- `go2_piper_driver/go2_bridge_node.py` — guarded `/cmd_vel` -> Go2 Unitree
  DDS body; publishes `/odom`, `/imu`, `/joint_states`.
- `go2_piper_driver/piper_driver_node.py` — `/piper/joint_cmd` -> Piper CAN
  (`JointCtrl`/`GripperCtrl`); torque gate via `/piper/enable` service;
  publishes `/piper/joint_states` + `/piper/arm_status`.
- `go2_piper_driver/launch/go2_piper_bringup.launch.py` — bring up both nodes.
- `compose.yaml` — container: CycloneDDS on eth0, maps `/dev/can0`.
- `cyclonedds.xml` — CycloneDDS config (eth0, Go2 DDS requirement).

## Build (on the Go2 dock)

The Unitree and Piper SDKs are **private** (not on PyPI) and live on the dock
host at `/home/unitree/unitree_sdk2_python` and `/home/unitree/piper_sdk`.
Stage them into the build context before building. The `cyclonedds` Python
package (required by unitree_sdk2py's SportClient) is compiled against a
host-built CycloneDDS C library (0.10.2, not packaged for Ubuntu Noble), so
stage that install tree + its OpenSSL 1.1 runtime deps too:

```bash
# from the repo root on the dock host:
mkdir -p .dock-build/sdks
cp -r /home/unitree/piper_sdk .dock-build/sdks/piper_sdk
cp -r /home/unitree/unitree_sdk2_python .dock-build/sdks/unitree_sdk2_python

# CycloneDDS 0.10.2 C library + headers (host source build) + libssl1.1 deps:
mkdir -p .dock-build/cyclonedds
cp -r /home/unitree/cyclonedds_ws/install/cyclonedds/* .dock-build/cyclonedds/
cp /usr/lib/aarch64-linux-gnu/libssl.so.1.1 .dock-build/cyclonedds/lib/
cp /usr/lib/aarch64-linux-gnu/libcrypto.so.1.1 .dock-build/cyclonedds/lib/

# build from the repo root (the Dockerfile COPY paths resolve against it):
sudo docker build \
  --build-arg ROS_BASE_IMAGE=ros:jazzy-ros-base-noble \
  -f deploy/go2-piper-driver/Dockerfile \
  -t ubrobot/go2-piper-driver:0.1.0 \
  .
```

`.dock-build/` is git-ignored (contains no source secrets, just staged SDKs).

> **Process isolation for the Unitree SDK**: initializing a `SportClient`
> (which uses the `cyclonedds` Python package) in the same process as the RMW
> CycloneDDS participant (`rclpy` / `rmw_cyclonedds_cpp`) segfaults. The
> `go2_bridge_node` therefore never constructs a SportClient in-process;
> `/go2/stand` shells out to `go2_stand_cli.py` (a separate python process
> with no rclpy). `/cmd_vel` forwarding from the ROS node is intentionally
> disabled for the same reason.

## Run

```bash
# bring up can0 first (Piper arm):
sudo ip link set can0 up type can bitrate 1000000

docker compose -f deploy/go2-piper-driver/compose.yaml up -d
# healthcheck: ros2 topic hz /odom
```

Torque is NOT enabled on startup: call the `/piper/enable` service
(`std_srvs/SetBool`) from the semantic layer before any grasp.

## Autostart on boot (systemd, recommended)

Two systemd units survive a power cycle: bring up `can0` first, then start
the container (the piper SDK must see a live `can0` at startup or it
degrades to telemetry-only). Units live in `systemd/`:

- `can0-up.service` — `ip link set can0 up type can bitrate 1000000` (oneshot).
- `go2-piper-driver.service` — `docker run` the container; `Requires` +
  `After=can0-up.service` so CAN is up before the SDK initializes. The
  container is created with `--restart unless-stopped` as a Docker-level
  backstop.

Install on the dock host:

```bash
sudo cp deploy/go2-piper-driver/systemd/can0-up.service \
        deploy/go2-piper-driver/systemd/go2-piper-driver.service \
        /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable can0-up.service go2-piper-driver.service
```

After a power cycle: `can0` is up, the container runs, and
`/piper/arm_status` reports `enabled=False sdk=ok` (torque off, arm safe).
