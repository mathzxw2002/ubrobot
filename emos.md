# EMOS Vision Depth Follower Runbook

This document records the working command sequence for running an EMOS
vision-depth follower on the LeKiwi robot with:

- Raspberry Pi / LeKiwi host: `rasp_pi`
- VLM / RoboML server: `vlm_server`
- EMOS container name: `emos`
- ROS 2 distro inside the container: `jazzy`
- RMW used for this workflow: `rmw_fastrtps_cpp`
- Camera: Intel RealSense D435I
- Object detection service: `roboml-resp`
- Navigation stack pieces:
  - RealSense RGBD
  - `depthimage_to_laserscan`
  - RTAB-Map RGB-D odometry
  - EMOS recipe `vision_depth_follower`
  - Kompass vision tracking action `/track_vision_target`

Run each long-running command in its own terminal and keep it open.

## 0. Start the VLM / RoboML Server

On the VLM server:

```bash
ssh vlm_server
conda activate roboml
roboml-resp
```

Expected result:

```text
Uvicorn running on http://0.0.0.0:6379
```

If HuggingFace is unreachable, make sure the object detection model is already
available locally on the VLM server, and that the EMOS recipe points
`VisionModel.checkpoint` to the local model path.

Example:

```python
object_detection = VisionModel(
    name="object_detection",
    checkpoint="/home/sany/roboml_models/rtdetr_r50vd_coco_o365",
)
```

## 1. Login to the LeKiwi / Raspberry Pi

On your workstation:

```bash
ssh rasp_pi
```

All remaining commands are run on the Raspberry Pi unless noted otherwise.

## 2. Start the EMOS Recipe

Start the recipe with Fast DDS and skip the initial sensor check because the
camera, scan, odom, and TF sources are started manually in separate terminals.

```bash
emos run vision_depth_follower --rmw rmw_fastrtps_cpp --skip-sensor-check
```

Keep this terminal open. The recipe should start nodes such as:

```text
/detection_component
/my_controller
/my_driver
/mapper
```

Useful checks:

```bash
docker exec -it emos bash -c "source /ros_entrypoint.sh && ros2 node list"
docker exec -it emos bash -c "source /ros_entrypoint.sh && ros2 action list -t"
```

The vision tracking action should be:

```text
/track_vision_target [kompass_interfaces/action/TrackVisionTarget]
```

## 3. Start RealSense RGBD

Start RealSense inside the EMOS container. This command enables RGB, depth,
depth alignment, stream sync, and the RealSense RGBD topic.

```bash
docker exec -it emos bash -c "source /ros_entrypoint.sh && export LD_LIBRARY_PATH=/opt/ros/jazzy/lib/aarch64-linux-gnu:\$LD_LIBRARY_PATH && export RMW_IMPLEMENTATION=rmw_fastrtps_cpp && ros2 launch realsense2_camera rs_launch.py enable_color:=true enable_depth:=true align_depth.enable:=true enable_sync:=true enable_rgbd:=true"
```

If the Raspberry Pi is overloaded, use a lower camera profile:

```bash
docker exec -it emos bash -c "source /ros_entrypoint.sh && export LD_LIBRARY_PATH=/opt/ros/jazzy/lib/aarch64-linux-gnu:\$LD_LIBRARY_PATH && export RMW_IMPLEMENTATION=rmw_fastrtps_cpp && ros2 launch realsense2_camera rs_launch.py enable_color:=true enable_depth:=true align_depth.enable:=true enable_sync:=true enable_rgbd:=true rgb_camera.color_profile:=640x480x15 depth_module.depth_profile:=640x480x15"
```

Verify the key topics:

```bash
docker exec -it emos bash -c "source /ros_entrypoint.sh && ros2 topic list | grep -E 'rgbd|aligned_depth|camera_info|color/image_raw'"
docker exec -it emos bash -c "source /ros_entrypoint.sh && ros2 topic info /camera/camera/rgbd -v"
```

Expected RGBD topic:

```text
/camera/camera/rgbd
Type: realsense2_camera_msgs/msg/RGBD
Publisher count: 1
```

In the recipe, the Vision input should use this exact topic:

```python
image0 = Topic(name="/camera/camera/rgbd", msg_type="RGBD")
```

## 4. Start the Depth-to-LaserScan Bridge

The Kompass local mapper and DriveManager need `/scan`. Generate it from the
aligned depth image.

Install the package once if needed:

```bash
docker exec -it emos bash -c "apt-get update && apt-get install -y ros-jazzy-depthimage-to-laserscan"
```

Run the bridge:

```bash
docker exec -it emos bash -c "source /ros_entrypoint.sh && ros2 run depthimage_to_laserscan depthimage_to_laserscan_node --ros-args -r depth:=/camera/camera/aligned_depth_to_color/image_raw -r depth_camera_info:=/camera/camera/aligned_depth_to_color/camera_info -r scan:=/scan"
```

Verify:

```bash
docker exec -it emos bash -c "source /ros_entrypoint.sh && timeout 5 ros2 topic hz /scan"
```

A working result should show a non-zero rate, for example around `8-10 Hz`.

## 5. Start the Required Static TF

Kompass needs a transform from the robot body frame to the camera/depth frame.
The default Kompass `ControllerConfig` depth frame was observed as:

```text
robot_base = base_link
depth = camera_depth_link
```

Run this static transform publisher and keep the terminal open:

```bash
docker exec -it emos bash -c "source /ros_entrypoint.sh && ros2 run tf2_ros static_transform_publisher 0.10 0.030 0.20 0 0 0 base_link camera_depth_link"
```

Also run the RealSense root camera transform if the TF tree needs
`base_link -> camera_link`:

```bash
docker exec -it emos bash -c "source /ros_entrypoint.sh && ros2 run tf2_ros static_transform_publisher 0.10 0.030 0.20 0 0 0 base_link camera_link"
```

Verify:

```bash
docker exec -it emos bash -c "source /ros_entrypoint.sh && timeout 5 ros2 run tf2_ros tf2_echo base_link camera_depth_link"
docker exec -it emos bash -c "source /ros_entrypoint.sh && timeout 5 ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame"
```

The expected TF tree should include:

```text
odom
  -> base_link
      -> camera_link
          -> camera_color_frame
              -> camera_color_optical_frame
```

## 6. Start RTAB-Map RGB-D Odometry

RTAB-Map provides `/odom` from RealSense RGB-D data.

Install RTAB-Map once if needed:

```bash
docker exec -it emos bash -c "apt-get update && apt-get install -y ros-jazzy-rtabmap-ros ros-jazzy-rtabmap-odom"
```

Check available executables:

```bash
docker exec -it emos bash -c "source /ros_entrypoint.sh && ros2 pkg executables rtabmap_odom"
```

Expected:

```text
rtabmap_odom rgbd_odometry
```

Start RGB-D odometry:

```bash
docker exec -it emos bash -c "source /ros_entrypoint.sh && ros2 run rtabmap_odom rgbd_odometry --ros-args -p frame_id:=base_link -p odom_frame_id:=odom -p publish_tf:=true -p approx_sync:=true -p approx_sync_max_interval:=0.05 -p topic_queue_size:=50 -p sync_queue_size:=50 -p qos:=1 -p qos_camera_info:=1 -r rgb/image:=/camera/camera/color/image_raw -r depth/image:=/camera/camera/aligned_depth_to_color/image_raw -r rgb/camera_info:=/camera/camera/color/camera_info -r odom:=/odom"
```

Verify:

```bash
docker exec -it emos bash -c "source /ros_entrypoint.sh && timeout 5 ros2 topic hz /odom"
docker exec -it emos bash -c "source /ros_entrypoint.sh && timeout 5 ros2 run tf2_ros tf2_echo odom base_link"
```

Working output should include a non-zero `/odom` rate and a valid
`odom -> base_link` transform.

RTAB-Map may warn about RGB/depth timestamp differences:

```text
The time difference between rgb and depth frames is high
```

If odometry still publishes with good quality, this warning is acceptable for
initial testing. Tune `approx_sync_max_interval` later if needed.

## 7. Kompass Patch Required for Current Container Version

The current container version had this line in:

```text
/opt/ros/jazzy/lib/python3.12/site-packages/kompass/components/_vision_follower.py
```

Original line:

```python
if not self._vision_controller or not self.setup():
```

This short-circuits on the first action because `self._vision_controller` is
initially `None`, so `self.setup()` is not called.

Temporary patch:

```bash
docker exec -it emos bash -c "python3 - <<'PY'
from pathlib import Path
p = Path('/opt/ros/jazzy/lib/python3.12/site-packages/kompass/components/_vision_follower.py')
s = p.read_text()
old = '        if not self._vision_controller or not self.setup():'
new = '        if not self._vision_controller and not self.setup():'
if old not in s:
    raise SystemExit('target line not found')
p.write_text(s.replace(old, new, 1))
print('patched vision follower setup condition')
PY"
```

Restart the recipe after applying the patch.

This is a container-local patch. It may be lost if the container image is
recreated or upgraded.

## 8. Recipe Configuration Notes

For LeKiwi, use `RobotType.OMNI`, not `ACKERMANN`.

Use Python lists instead of `np.array(...)` in `RobotConfig`, because the
Kompass/ros_sugar launcher serializes config to JSON.

Recommended robot config:

```python
import math

my_robot = RobotConfig(
    model_type=RobotType.OMNI,
    geometry_type=RobotGeometry.Type.CYLINDER,
    geometry_params=[0.18, 0.35],
    ctrl_vx_limits=LinearCtrlLimits(max_vel=0.25, max_acc=0.5, max_decel=0.8),
    ctrl_vy_limits=LinearCtrlLimits(max_vel=0.25, max_acc=0.5, max_decel=0.8),
    ctrl_omega_limits=AngularCtrlLimits(
        max_vel=0.8,
        max_acc=1.0,
        max_decel=1.5,
        max_steer=math.pi / 3,
    ),
)
```

Recommended controller config:

```python
config = ControllerConfig(
    loop_rate=10.0,
    ctrl_publish_type="Parallel",
    control_time_step=0.3,
)

config.frames.robot_base = "base_link"
config.frames.depth = "camera_depth_link"
config.topic_subscription_timeout = 15.0

controller = Controller(component_name="my_controller", config=config)

controller.inputs(
    vision_detections=detections_topic,
    depth_camera_info=Topic(
        name="/camera/camera/aligned_depth_to_color/camera_info",
        msg_type="CameraInfo",
    ),
)

controller.algorithm = ControllersID.VISION_DEPTH
controller.direct_sensor = False
```

Recommended Vision input:

```python
image0 = Topic(name="/camera/camera/rgbd", msg_type="RGBD")
detections_topic = Topic(name="/vision_detections", msg_type="Detections")
```

Recommended local mapper:

```python
mapper = LocalMapper(
    component_name="mapper",
    config=LocalMapperConfig(
        map_params=MapConfig(width=4.0, height=4.0, resolution=0.1),
    ),
)

mapper.inputs(sensor_data=Topic(name="/scan", msg_type="LaserScan"))
```

## 9. Verify All Required Inputs

Before sending the action, verify:

```bash
docker exec -it emos bash -c "source /ros_entrypoint.sh && timeout 5 ros2 topic hz /scan"
docker exec -it emos bash -c "source /ros_entrypoint.sh && timeout 5 ros2 topic hz /odom"
docker exec -it emos bash -c "source /ros_entrypoint.sh && timeout 5 ros2 topic hz /local_map/occupancy_layer"
docker exec -it emos bash -c "source /ros_entrypoint.sh && timeout 5 ros2 topic hz /vision_detections --qos-reliability reliable --qos-durability volatile --qos-history keep_last --qos-depth 100"
docker exec -it emos bash -c "source /ros_entrypoint.sh && timeout 5 ros2 run tf2_ros tf2_echo base_link camera_depth_link"
docker exec -it emos bash -c "source /ros_entrypoint.sh && timeout 5 ros2 run tf2_ros tf2_echo odom base_link"
```

The controller should subscribe to:

```bash
docker exec -it emos bash -c "source /ros_entrypoint.sh && ros2 node info /my_controller"
```

Expected subscribers:

```text
/camera/camera/aligned_depth_to_color/camera_info
/local_map/occupancy_layer
/odom
/vision_detections
```

Verify detections include RGB and depth:

```bash
docker exec -it emos bash -c "source /ros_entrypoint.sh && ros2 topic echo /vision_detections --once --qos-reliability reliable --qos-durability volatile --qos-history keep_last --qos-depth 100"
```

Expected fields:

```text
labels:
- person
image:
  frame_id: camera_color_optical_frame
  height: 720
  width: 1280
depth:
  frame_id: camera_color_optical_frame
  height: 720
  width: 1280
  encoding: 16UC1
```

## 10. Send the Vision Tracking Action

Track a person:

```bash
docker exec -it emos bash -c "source /ros_entrypoint.sh && ros2 action send_goal /track_vision_target kompass_interfaces/action/TrackVisionTarget \"{label: 'person'}\" --feedback"
```

Track a chair:

```bash
docker exec -it emos bash -c "source /ros_entrypoint.sh && ros2 action send_goal /track_vision_target kompass_interfaces/action/TrackVisionTarget \"{label: 'chair'}\" --feedback"
```

Successful behavior:

```text
Goal accepted
Feedback:
    distance_error: ...
orientation_error: ...
```

`distance_error` is the distance error to the target. `orientation_error` is
the target angular error in radians.

## 11. Check Control Output

Check controller output:

```bash
docker exec -it emos bash -c "source /ros_entrypoint.sh && ros2 topic echo /control"
```

Check final velocity command:

```bash
docker exec -it emos bash -c "source /ros_entrypoint.sh && ros2 topic echo /cmd_vel"
```

If `/control` has messages but `/cmd_vel` does not, inspect `DriveManager`.

## 12. Known Warnings and Their Meaning

### `Could not initialize controller -> ABORTING ACTION`

Observed cause in this setup:

- Kompass vision follower setup was not called because of the short-circuit
  condition in `_vision_follower.py`.
- Required TF such as `base_link -> camera_depth_link` was missing.

Fix:

- Apply the temporary Kompass patch.
- Keep the static TF publisher running.
- Restart the recipe after TF is available.

### `Box updated with invalid time step, Velocity will be reset to zero`

Meaning:

- The bounding box timestamp delta was invalid or unstable.
- The target velocity estimate is reset to zero.

This can happen with heavy RGBD messages and slow detector inference. It is not
always fatal if action feedback and `/control` continue.

Mitigations:

- Lower camera resolution and FPS.
- Check detection frequency.
- Reduce load on the VLM server and Raspberry Pi.

### `The time difference between rgb and depth frames is high`

RTAB-Map warning caused by RGB/depth timestamp mismatch.

Mitigations:

- Keep `enable_sync:=true` in RealSense.
- Tune `approx_sync_max_interval`.
- Start with `0.05`, then try `0.02` if stable.

## 13. Quick Start Order

Use this order for a clean run:

1. VLM server:

```bash
ssh vlm_server
conda activate roboml
roboml-resp
```

2. Raspberry Pi: start EMOS recipe:

```bash
ssh rasp_pi
emos run vision_depth_follower --rmw rmw_fastrtps_cpp --skip-sensor-check
```

3. Raspberry Pi: start RealSense RGBD.

4. Raspberry Pi: start static TF publishers.

5. Raspberry Pi: start RTAB-Map RGB-D odometry.

6. Raspberry Pi: start `depthimage_to_laserscan`.

7. Apply the Kompass patch if needed, then restart the recipe.

8. Verify `/scan`, `/odom`, `/local_map/occupancy_layer`, `/vision_detections`,
   and TF.

9. Send `/track_vision_target`.

