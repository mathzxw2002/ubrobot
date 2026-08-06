# Go2 + Piper 接入清单（Task 1）

本清单冻结 Go2+Piper 接入所需的"已有实现位置 -> adapter 协议方法"映射，以及拓展坞环境事实。后续每个 adapter 的实际调用都必须在此有已知来源、版本与硬件验证状态；未知项不得以推测填充。

## 状态标记

每条接入项标注以下三者之一：

- **verified on hardware** —— 已在目标硬件（拓展坞/真机）上验证可用。
- **fixture only** —— 代码已存在且单测通过，但未在 Go2+Piper 目标硬件上验证。
- **unknown** —— 尚未确认，必须经 Step 3 探针或后续 Task 落实。

## 1. 拓展坞环境冻结（硬前置）

拓展坞 = Go2 拓展坞（Jetson Orin NX）。宿主 OS / ROS 不可升级。**宿主预装 ROS Noetic（ROS 1）不使用**；所有 ROS 2 在 Jazzy/Noble 容器内运行。JetPack/CUDA 冻结 -> 感知/VLM 不在本地跑（见接入映射"远程感知服务"）。

| 项 | 值 | 状态 |
|---|---|---|
| 宿主 OS | `TODO-verify-on-dock` | unknown |
| JetPack 版本 | `TODO-verify-on-dock` | unknown |
| CUDA 版本 | `TODO-verify-on-dock` | unknown |
| Docker 基础镜像 | `ros:jazzy-ros-base-noble` | verified on hardware（仓库 Dockerfile 已用；拓展坞可拉起待 Step 3 确认） |
| Noble 容器可拉起 | pending | unknown —— 在拓展坞执行：`docker run --rm --platform linux/arm64 ros:jazzy-ros-base-noble bash -c "echo ok && python3 -c 'import rclpy; print(rclpy.__file__)'"` |
| RealSense 内核可用 | pending | unknown —— 在拓展坞执行：`ls /dev/video*` 与 `rs-enumerate-devices --compact`（若安装） |
| `can0` 可用 | pending | unknown —— 在拓展坞执行：`ip link show can0` |
| ROS 域 | `ROS_DOMAIN_ID=0`（与 Pi/LeKiwi 独立，可改） | fixture only |

> 上述 `unknown` 项是本计划的硬前置：若 Noble 容器无法在拓展坞拉起、或 Go2 接口实为 ROS 1 且 `ros1_bridge` 不可用、或 dock 内 RMW 无法统一，触发计划停止条件。

## 2. 接入映射

### Go2 运动
- **运动入口 = `/cmd_vel` (geometry_msgs/Twist) -> Go2 ROS 2 bridge -> 四足**。
  - 状态：**fixture only**（设计已定：`cmd_vel_guard` 发布 `/cmd_vel`，bridge 消费；`navigate_to_object_server.py` 不改、不建运动 adapter）。
  - bridge 来源/版本/RMW：`unknown`（见 env `GO2_ROS2_BRIDGE_SOURCE`/`GO2_ROS2_BRIDGE_VERSION`/`GO2_BRIDGE_RMW`）。
  - bridge 话题名（`/odom`、`/imu`、`/joint_states`）与 TF 根帧：`unknown`。
- **Go2 站起 / stop / sport-mode 原语**：`unknown`（确认由 bridge 自带还是单独服务；`/cmd_vel` 仅在已站起 + sport velocity 模式有效）。
- **废弃路径**：`src/ubrobot/robots/unitree_go2_robot.py` 直接 `SportClient`（`Move`/`StopMove`/`StandUp`）运动方式 **废弃（deprecated）**，本计划不接入；Go2 运动一律经 `/cmd_vel`。

### Piper（拓展坞本地，`can0`）
- **start / joint state / cancel-stop / gripper / torque status** = `src/ubrobot/robots/piper/piper_sdk_interface.py`（`set_joint_positions_deg`/`GripperCtrl`/`EnablePiper`/`get_status_deg`）。
  - 状态：**fixture only**（代码深度开发且存在；Go2+Piper 组合在拓展坞上的硬件验证待 Task 2/4）。
  - `can0` 可用性：`unknown`（见环境冻结表）。
  - 不直接调用 `ros_depends_ws/src/piper_ros/src/piper/scripts/piper_ctrl_single_node.py`（ROS 1，直接 CAN）。

### RGB-D / 检测 / TF
- 彩色话题 `/camera/camera/color/image_raw`：**fixture only**（`deploy/emos/recipes/cortex_navigation/recipe.py` 已订阅）。
- 深度话题 `/camera/camera/depth/image_rect_raw` 与相机内参来源 `/camera/camera/color/camera_info`：**fixture only**（标准 RealSense 命名；拓展坞实机话题待确认）。
- TF 树完整性（`camera_*` <-> `base_link` <-> `odom`）：`unknown`。

### 远程感知服务（x86 GPU 服务器，不在拓展坞）
- 服务 = `src/service/reasoning/http_reasoning_server.py`（Flask，端口 5802）+ `grasp_plan.py`（`RobotArmMotionPlan`，GraspNet on `cuda:0`）。
  - 状态：**fixture only**（代码已存在；`/grasp_poses` 端点契约待 Task 4 落实或确认；拓展坞到服务器的网络可达性 `unknown`）。
  - `PerceptionInterface` 绑定为该服务的 HTTP 客户端；感知失败 fail-closed。

### 已执行过的测试
- Piper 遥操作 / 抓取：历史已验证（Piper standalone）。
- Go2 经 Kompass/`cmd_vel` 导航：`unknown`（本集成首次）。
- Go2+Piper 联合停止路径：`unknown`（Task 6 验证）。

## 3. 待拓展坞执行的只读探针（Step 3）

以下命令须在拓展坞（Orin NX）上执行，脱敏后回填本表与 `go2-piper.example.env` 的 `pending`/`TODO-verify-on-dock` 字段：

```bash
# 1) Jazzy/Noble 容器可拉起性
docker run --rm --platform linux/arm64 ros:jazzy-ros-base-noble \
  bash -c "echo ok && python3 -c 'import rclpy; print(rclpy.__file__)'"
# 2) RealSense 可见性
ls /dev/video* ; rs-enumerate-devices --compact 2>/dev/null || true
# 3) Piper CAN 可见性
ip link show can0
# 4) Go2 bridge 话题/RMW（bridge 启动后）
ros2 topic list ; ros2 topic info /cmd_vel ; ros2 topic info /odom
# 5) Go2 站起/停止原语来源（bridge 文档或服务列表）
ros2 service list | grep -iE 'stand|stop|sport'
```
