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
Stage them into the build context before building:

```bash
# from the repo root on the dock host:
mkdir -p .dock-build/sdks
cp -r /home/unitree/piper_sdk .dock-build/sdks/piper_sdk
cp -r /home/unitree/unitree_sdk2_python .dock-build/sdks/unitree_sdk2_python

# build from the repo root (the Dockerfile COPY paths resolve against it):
sudo docker build \
  --build-arg ROS_BASE_IMAGE=ros:jazzy-ros-base-noble \
  -f deploy/go2-piper-driver/Dockerfile \
  -t ubrobot/go2-piper-driver:0.1.0 \
  .
```

`.dock-build/` is git-ignored (contains no source secrets, just staged SDKs).

## Run

```bash
# bring up can0 first (Piper arm):
sudo ip link set can0 up type can bitrate 1000000

docker compose -f deploy/go2-piper-driver/compose.yaml up -d
# healthcheck: ros2 topic hz /odom
```

Torque is NOT enabled on startup: call the `/piper/enable` service
(`std_srvs/SetBool`) from the semantic layer before any grasp.
