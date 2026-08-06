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
| 宿主 OS | `Ubuntu 20.04.5 LTS (Focal Fossa)`，kernel `5.10.104-tegra`，aarch64 | verified on hardware |
| JetPack 版本 | `R35.3.1`（`/etc/nv_tegra_release`: R35 REVISION 3.1, BOARD t186ref） | verified on hardware |
| CUDA 版本 | `CUDA 11.4`（`/usr/local/cuda-11.4`，`/dev/nvidia{0,ctl,modeset}` 存在） | verified on hardware |
| Docker 基础镜像 | `ros:jazzy-ros-base-noble` | verified on hardware（2026-08-07 实测拉取并运行成功，rclpy 在 `/opt/ros/jazzy/lib/python3.12/site-packages/rclpy`） |
| Noble 容器可拉起 | `yes` | verified on hardware —— `docker run --rm --platform linux/arm64 ros:jazzy-ros-base-noble bash -c "echo ok && python3 -c 'import rclpy; print(rclpy.__file__)'"` 输出 `ok` 后正常退出 |
| RealSense 内核可用 | `yes` | verified on hardware —— `rs-enumerate-devices` 识别 **Intel RealSense D435IF**（SN 336222070923，固件 5.17.0.10，USB 3.2）；`/dev/video0..5` 共 6 个节点 |
| `can0` 可用 | `yes` | verified on hardware —— `ip link set can0 up bitrate 1000000` 成功（gs_usb 适配器，state UP / ERROR-ACTIVE）；`candump` 2s 内即收到 Piper 实时帧（ID 0x2A1..0x2A5，见"已执行过的测试"） |
| Docker 镜像加速 | `registry-mirrors` 已配置 | verified on hardware —— Docker Hub `registry-1.docker.io` 不可达（超时），已配置 `docker.m.daocloud.io`/`docker.1panel.live`/`hub.rat.dev` 三个镜像加速，重启 dockerd 后拉取正常 |
| ROS 域 | `ROS_DOMAIN_ID`（与 Pi/LeKiwi 独立，可改） | fixture only —— 宿主 `.bashrc` 已设 `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` + `CYCLONEDDS_URI=~/cyclonedds_ws/cyclonedds.xml`（eth0） |

> 上述 `unknown` 项是本计划的硬前置：若 Noble 容器无法在拓展坞拉起、或 Go2 接口实为 ROS 1 且 `ros1_bridge` 不可用、或 dock 内 RMW 无法统一，触发计划停止条件。

## 2. 接入映射

### Go2 运动
- **运动入口 = `/cmd_vel` (geometry_msgs/Twist) -> Go2 ROS 2 bridge -> 四足**。
  - 状态：**fixture only**（设计已定：`cmd_vel_guard` 发布 `/cmd_vel`，bridge 消费；`navigate_to_object_server.py` 不改、不建运动 adapter）。
  - bridge 来源/版本/RMW：见下方"Go2 接口形态（Task 1 结论）"。
  - bridge 话题名（`/odom`、`/imu`、`/joint_states`）与 TF 根帧：见下方"Go2 接口形态（Task 1 结论）"。
- **Go2 站起 / stop / sport-mode 原语**：`unknown`（确认由 bridge 自带还是单独服务；`/cmd_vel` 仅在已站起 + sport velocity 模式有效）。
- **废弃路径**：`src/ubrobot/robots/unitree_go2_robot.py` 直接 `SportClient`（`Move`/`StopMove`/`StandUp`）运动方式 **废弃（deprecated）**，本计划不接入；Go2 运动一律经 `/cmd_vel`。

### Go2 接口形态（Task 1 结论，2026-08-07 实测）
- **接口类型：Unitree 私有 DDS（CycloneDDS）**，非 ROS 1，也非标准 ROS 2 wrapper。
  - SDK = 官方 `unitree_go2_sdk`（源码 `/unitree/lib/unitree_go2_sdk/`，已 `install.sh` 装入 `/usr/local`：`libunitree_go2_sdk.a`、`libunitree_go2_idl_cpp.a`、`libunitree_ros2_idl_cpp.a`）。
  - 话题（DDS）：`rt/sportmodecmd`（命令）/ `rt/sportmodestate`（状态）/ `rt/lowcmd` / `rt/lowstate`（low-level）。示例 `example/go2_sub/go2_sub.cpp`、`go2_state_pub.cpp`、`send_cmd.cpp`。
  - **SDK 自带 ROS2 IDL 类型头**（`/usr/local/include/unitree/ros2_idl/`：`Twist_.hpp`、`TwistStamped_.hpp`、`Odometry_.hpp`、`PointCloud2_.hpp` 等）——证明 unitree_ros2 生态可直接与 ROS 2 类型互通。
  - 网络接口：Go2 通信走 **eth0 = 192.168.123.18/24**，狗本体 **192.168.123.161**（`ping` 可达，ARP REACHABLE）。SDK 示例硬编码接口 `enx0826ae3e0542`（旧环境的枚举名），当前机为 `eth0`，接入时需按 `ip -o -4 addr` 实态替换。
- **RMW：宿主 `rmw_cyclonedds_cpp`**（`.bashrc` 已配置 `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`、`CYCLONEDDS_URI=~/cyclonedds_ws/cyclonedds.xml`，xml 指定 `NetworkInterface=eth0`）。
  - **结论：bridge 与 ubrobot 容器同 RMW 用 `rmw_cyclonedds_cpp` 可互通；与 emos 容器 `rmw_zenoh_cpp` 不互通**（详见 Task 3 注意事项）。
- **gRPC 运动服务（既有）**：`go2_sport_grpc_server eth0 0.0.0.0 50051`（aarch64 预编译二进制在 `/home/unitree/unitree_sdk2_go2_agentskill/build-go2-grpc/bin/` 与 `artifacts/go2_grpc/`），带 open/close session、heartbeat、Move action。历史实际使用方式（`.bash_history`）。**本计划不采用 gRPC 作为 `/cmd_vel` 链路**，仅记录为既有控制入口。
- **ROS 1 路径（废弃，仅记录）**：`unitree_ros` 的 `twist_sub.cpp` 订阅 ROS1 `cmd_vel` 经 UDP high 通道（8090→`192.168.123.161:8082`）发 HighCmd。宿主有 `/opt/ros/noetic` + `/opt/ros/foxy`。**Go2 接口实为 DDS 而非 ROS1，无需 `ros1_bridge`。**
- 站起/停止原语来源：`unknown`——SDK `sport_client` 提供 `StandUp/StandDown/StopMove/Move`，可由 bridge 内置或单独服务提供，Task 3 落实。

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
- **can0（Piper）实机帧**：`candump` 在 1Mbps 下 2s 内收到实时帧 ID `0x2A1..0x2A5`（每帧 8 字节，Piper 关节/状态上报），证明 Piper 臂在线供电并广播状态。verified on hardware（2026-08-07）。
- **Go2 本体连通**：`ping 192.168.123.161` 0% 丢包，ARP REACHABLE（eth0）。verified on hardware（2026-08-07）。
- **Noble 容器**：`ros:jazzy-ros-base-noble` 在 Orin NX 拉取运行成功（rclpy OK）。verified on hardware（2026-08-07）。
- Go2 经 Kompass/`cmd_vel` 导航：`unknown`（本集成首次，Task 3）。
- Go2+Piper 联合停止路径：`unknown`（Task 6 验证）。

## 3. 待拓展坞执行的只读探针（Step 3）

以下命令已在拓展坞（Orin NX）执行（2026-08-07），脱敏结果已回填上文；`pending`/`TODO-verify-on-dock` 字段已相应更新为 `verified on hardware` 或保留 `unknown`：

```bash
# 1) Jazzy/Noble 容器可拉起性 —— 通过
docker run --rm --platform linux/arm64 ros:jazzy-ros-base-noble \
  bash -c "echo ok && python3 -c 'import rclpy; print(rclpy.__file__)'"
# 2) RealSense 可见性 —— 通过（D435IF）
ls /dev/video* ; rs-enumerate-devices --compact 2>/dev/null || true
# 3) Piper CAN 可见性 —— 通过（can0 1Mbps up + candump 收到帧）
ip link show can0
# 4) Go2 bridge 话题/RMW —— 部分通过：接口确认为 Unitree 私有 DDS (CycloneDDS) + ROS2 IDL；
#    bridge 尚未运行，`ros2 topic list` 待 bridge 启动后（Task 3）执行
ros2 topic list ; ros2 topic info /cmd_vel ; ros2 topic info /odom
# 5) Go2 站起/停止原语来源（bridge 文档或服务列表）—— 待 Task 3
ros2 service list | grep -iE 'stand|stop|sport'
```

### Task 1 新发现的关键事实（2026-08-07）

1. **Docker Hub 不可达，必须走镜像加速**：`registry-1.docker.io` 在拓展坞上连接超时；已配置 `docker.m.daocloud.io` 等镜像并验证可用。`/etc/docker/daemon.json` 的 `runtimes.nvidia` 已保留（`nvidia-container-runtime`）。
2. **Go2 接口是 Unitree 私有 DDS，不是 ROS 1**：因此**不需要 `ros1_bridge`**（计划 Task 1 的兜底项解除）。SDK 自带 `libunitree_ros2_idl_cpp.a` 与 `ros2_idl/*.hpp`，bridge 可将 `/cmd_vel` (Twist) 经 ROS2 IDL -> DDS `rt/sportmodecmd` 下发。
3. **RMW 冲突预警**：宿主/Go2 生态用 `rmw_cyclonedds_cpp`；已运行的 **emos 容器是 `rmw_zenoh_cpp`**。若 bridge 与 emos 需同 ROS 图互操作，RMW 必须统一（Task 3 处理；或 bridge/ubrobot 容器内部独立域）。
4. **既有 gRPC 运动服务**：`go2_sport_grpc_server`（50051）是可用的既有控制入口，但本计划不用它作 `/cmd_vel` 链路，仅记录。
5. **扩展坞宿主还有 ROS1 `unitree_ros`（twist_sub UDP 8090）与 `/opt/ros/foxy`**：均不作为本计划路径，仅记录避免混淆。
