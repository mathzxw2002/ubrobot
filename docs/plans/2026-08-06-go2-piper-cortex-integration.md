# Go2 + Piper Cortex 编排接入 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将已开发验证过的 Piper 抓取能力与 Go2 移动能力接入现有 Cortex 语义编排链路，使 `NavigateToObject` 与 `GraspObject` 两个语义 Action 可在 Go2+Piper 上各自受控、安全地独立执行。Cortex 顺序编排（自然语言任务自动"先导航后抓取"）为产品化事项，本轮暂缓——可由操作者分别调用两个 Action。Go2 的运动**不直接调 Unitree SDK**，而是复用现有 Kompass 导航链路经 `/cmd_vel` 驱动 Go2 自带的 ROS 2 接口。

**Architecture:**
- 边界保持 `Operator Console -> Robot Edge -> /cortex_input_command -> 语义 ROS Action -> 平台 executor -> 驱动`。Cortex 只能发现导航与抓取两个语义工具，不能发现 Unitree SDK、Piper CAN、关节、扭矩、`cmd_vel` 或 sport client。
- **Go2 运动链完全复用现有导航栈**：`NavigateToObjectServer`（已存在，对底盘无感）-> `TrackVisionTargetAdapter` -> Kompass `/track_vision_target` -> Kompass `DriveManager` -> `/navigation/raw_cmd_vel` -> `cmd_vel_guard`（lease/急停/stale 发零）-> `/cmd_vel` -> **Go2 ROS 2 接口**（unitree_ros2 bridge，在拓展坞上以容器运行）-> Go2 四足。本计划**不新建 Go2 运动 adapter，不修改 `navigate_to_object_server.py`**。
- **Piper 抓取在拓展坞本地**：Piper CAN 走拓展坞 `can0`，复用 `piper_sdk_interface.py`；抓取状态机复用 `PiperGraspNetExecutor`。
- **感知/VLM 走远程服务**：GraspNet/VLM 以 HTTP 服务形式部署在 x86 GPU 服务器上（仓库已有 `src/service/reasoning/http_reasoning_server.py`），拓展坞只发图像/深度、收抓取位姿，本地无 GPU 依赖。
- **安全互斥（底线，已有）**：既有 `authority.py`/`lifecycle.py` 的 fail-closed 互斥（lease 存在或底盘不静止则拒绝抓取；抓取中 lease 出现即取消）作为安全底线保留，使两个 Action 可分别安全调用。统一的 `motion_arbitration.py` 仲裁器与 Cortex 自动顺序编排为产品化事项，本轮暂缓。

**Tech Stack:** ROS 2 Jazzy（Ubuntu Noble 容器）、EMOS Cortex、Robot Edge/FastAPI、`ubrobot_navigation`、`ubrobot_manipulation`、Kompass、Go2 unitree_ros2 bridge、Piper `piper_sdk_interface`、RealSense、Flask 推理服务（GraspNet/VLM）、pytest/unittest、Docker Compose。

---

## 拓展坞环境约束（硬前置）

开发与运行环境为 **Go2 拓展坞（Jetson Orin NX）**。以下约束在 Task 1 必须核实并记录，且不允许通过"升级系统"绕过：

1. **宿主 OS / ROS 冻结**：拓展坞宿主预装 ROS Noetic（ROS 1，Ubuntu 20.04 量级）。**本计划不使用宿主 ROS**——所有 ROS 2 逻辑在 Jazzy/Noble 容器内运行；Noetic 仅作为板载 OS 存在，不启动、不参与通信。
2. **JetPack / CUDA 冻结**：Orin NX 的 JetPack 与 CUDA 版本不可升级。因此**感知/VLM 不在拓展坞本地跑**（GraspNet 的 aarch64 依赖编译 + CUDA 版本冲突），一律走远程 HTTP 服务。
3. **容器化是唯一路径**：拓展坞宿主必须能 `docker run ros:jazzy-ros-base-noble`（aarch64）。CPU 工作负载（Kompass、导航、抓取控制、Robot Edge、Go2 bridge）在容器内跑；拓展坞不承担 GPU 推理。
4. **Go2 接口形态待确认**：Go2 提供 ROS 2 接口（unitree_ros2）。Task 1 必须确认其 ROS 版本、RMW、话题名；若实为 ROS 1 Noetic 话题，则需在拓展坞上跑 `ros1_bridge` 容器做话题桥。
5. **Pi 无关**：Pi（LeKiwi 本体）是独立机器人，不参与本集成。Go2 栈与 Pi 栈各自独立 ROS 域、可各自 RMW，互不通信。

## 当前基础与目标缺口

已存在的基础（均已核实）：

- `NavigateToObject` Action（`ros_depends_ws/src/ubrobot_interfaces/action/NavigateToObject.action`）与 server（`navigate_to_object_server.py`）已运行：经 `TrackVisionTargetAdapter` 调 Kompass `/track_vision_target`，Kompass `DriveManager` 输出 `/navigation/raw_cmd_vel`，经 `cmd_vel_guard` 出 `/cmd_vel`。lease 在 `/navigation/command_lease`（0.1s 心跳），并订阅 Cortex 状态以在 Cortex goal 结束时取消下游。**该链路在 `/cmd_vel` 上游对底盘类型无感，LeKiwi 与 Go2 通用。**
- `cmd_vel_guard`（`ubrobot_navigation/cmd_vel_guard.py`）：订阅 `/navigation/raw_cmd_vel` + `/navigation/command_lease`，20Hz 发 `/cmd_vel`；lease stale(0.25s)/撤销/NaN 即发零；速度经 `policy.sanitize_twist()` 限幅。测试 `test_cmd_vel_guard.py`。
- `GraspObject` Action（`GraspObject.action`）与 server（`grasp_object_server.py`）fail-closed：`build_executor()` 仅在 `UBROBOT_GRASP_EXECUTOR=fixture` 时构造 `DeterministicGraspExecutor`，否则 `NotImplementedError`；未知 `UBROBOT_GRASP_PLATFORM` 直接中止启动。已用 `AuthorityTracker`（lease + cmd_vel）gating。
- `PiperGraspNetExecutor`（`executors/piper_graspnet.py`）：状态机 approach->align->grasp->retreat，注入式 `PerceptionInterface`（`locate_grasp_poses`）/`MotionInterface`（`execute_grasp`+`hold_position`），无真实 binding。测试 `test_piper_graspnet.py`。
- `GraspLifecycleCoordinator`（`lifecycle.py`）：reserve/execute/abandon，抓取前查 `grasp_may_start()`，抓取中 lease 出现即取消。
- `authority.py` 的 `AuthorityTracker`：`navigation_lease_active()`、`base_is_stationary()`（无证据=不静止，fail-closed），纯 Python，已测。
- `policy.py`：`go2_piper` profile（`requires_stationary_base=True`、`max_approach_speed_mps=0.05`）、`PLATFORM_PROFILES`、`sanitize_twist()`、`grasp_may_start()`。
- `CORTEX_ENABLE_GRASP`（`recipe.py`，默认关）、`GRASP_TOOL_DESCRIPTION`、`grasp_exposure_enabled()`。
- `piper_sdk_interface.py`（`src/ubrobot/robots/piper/`）：`set_joint_positions_deg()`、`GripperCtrl()`、`EnablePiper()`、`get_status_deg()`，Python 级可用。
- 远程推理服务 `src/service/reasoning/http_reasoning_server.py`（Flask，端口 5802，VLM `/eval_reasoning_vqa_cosmos`）与 `grasp_plan.py`（`RobotArmMotionPlan`，`cuda:0` 跑 GraspNet，`generate_6d_grasp_pose()`）已存在，属 x86 GPU 服务器代码。
- Robot Edge FastAPI（`src/robot_edge/app.py`）已有 health/capabilities/telemetry/commands/lease/safety 端点；`MobileBaseHealth`（`mobile_base_health.py`，`SUPPORTED_PROFILES=("lekiwi",)`）与 `PiperHealth`（注入 `PiperSystemProbe`，无 SDK 导入）已有只读范式。
- `docs/validation/` 有 18 份既定模板报告；`docs/hardware/` 不存在（Task 1 创建）。

因此本计划**不是重做 Go2/Piper 控制，也不是重做导航**，而是：把拓展坞环境固化、把 Go2 经 ROS 2 接口接入既有 `cmd_vel` 链路、把 Piper executor 真实 binding（感知远程 + 运动本地）、补齐共享仲裁与受控开关、做联合验收。

## 允许的 Cortex 工具面

| Cortex 可见 | 参数 | 执行者 | 禁止暴露 |
|---|---|---|---|
| `NavigateToObject` | `target`、`timeout_sec` | Kompass -> /cmd_vel -> Go2 ROS 2 bridge | sport client、Unitree SDK、`cmd_vel` 原始发布 |
| `GraspObject` | `target`、`timeout_sec` | Piper `piper_sdk_interface` + 远程 GraspNet 服务 | CAN、关节目标、扭矩、夹爪原始命令、`piper_ctrl_single_node` |
| `describe_scene` | 可选 query | 受控相机预览 -> 远程 VLM 服务 | 原始图像对象、相机句柄 |

## Task 1: 盘点并冻结拓展坞环境与 Go2+Piper 接入清单

**Files:**
- Create: `docs/hardware/go2-piper-integration-inventory.md`
- Create: `deploy/robot-edge/config/go2-piper.example.env`
- Create: `tests/robot_edge/test_go2_piper_inventory_contract.py`

**Step 1: 写失败的配置完整性测试**

测试 `go2_piper.example.env` 必须定义且不含真实值：拓展坞宿主 OS 与 JetPack/CUDA 版本、Docker 基础镜像（`ros:jazzy-ros-base-noble`）并在 inventory 中标注"已验证可拉起"、Go2 ROS 2 bridge 来源与版本、Go2 接口是 ROS 2 还是 ROS 1（若 ROS 1 需 `ros1_bridge`）、bridge 的 RMW、Go2 `/cmd_vel`/`/odom`/`/imu`/`/joint_states` 话题名与 TF 根帧、Go2 站起/停止原语、Piper `can0`、Piper driver 启动方式、RGB-D 话题与内参、远程感知服务 URL 与契约、`ROS_DOMAIN_ID`、标定版本。token/IP/序列号不入库。

Run: `python -m unittest tests.robot_edge.test_go2_piper_inventory_contract -v`

Expected: FAIL，因为清单与受版本控制的 example.env 尚不存在。

**Step 2: 记录环境与已验证实现的映射，而非复制代码**

在 inventory 中分两节填入：

(1) **拓展坞环境冻结**：宿主 OS/JetPack/CUDA、Docker `ros:jazzy-ros-base-noble` 在 Orin NX 上的 `docker run --rm` 验证结果、RealSense 在 JetPack 内核下的可用性、`can0` 可用性、网络与 ROS 域规划。

(2) **接入映射**，每条标 `verified on hardware` / `fixture only` / `unknown`，未知项不得推测填充：
```text
Go2 运动入口 = /cmd_vel (Twist) -> Go2 ROS 2 bridge -> 四足
Go2 站起 / stop / sport-mode 原语（确认由 bridge 还是单独服务提供）
Go2 /odom, /imu, /joint_states 来源与帧名
Piper start / joint state / cancel-stop / gripper / torque status = piper_sdk_interface.py
RGB-D / detection / TF
远程感知服务：http_reasoning_server URL, 端点契约, 已验证?
已执行过的遥操作、导航、抓取与停止测试
```
**显式标注**：`src/ubrobot/robots/unitree_go2_robot.py` 的直接 `SportClient` 运动方式在本计划中**废弃**，Go2 运动一律经 `/cmd_vel`。

**Step 3: 验证 Docker 与硬件可达性（只读）**

在拓展坞上执行并记录输出：`docker run --rm --platform linux/arm64 ros:jazzy-ros-base-noble bash -c "echo ok && python3 -c 'import rclpy; print(rclpy.__file__)'"`；`ls /dev/video*` 与 RealSense `rs-enumerate-devices`（若可用）；`ip link show can0`。脱敏后写入 inventory。

**Step 4: 验证并提交**

```powershell
python -m unittest tests.robot_edge.test_go2_piper_inventory_contract -v
git add docs/hardware deploy/robot-edge/config tests/robot_edge
git commit -m "docs: inventory dock environment and verified Go2 Piper inputs"
```

**Acceptance:** 后续每个 adapter 调用都有已知来源、版本和硬件验证状态；Go2 运动入口明确为 `/cmd_vel` 而非 SportClient；拓展坞能否跑 Noble 容器、Go2 接口 ROS1/ROS2、RMW 三项有明确结论或被停止条件触发。

## Task 2: 为 Go2+Piper 增加只读平台 profile 与健康证据

**Files:**
- Create: `src/robot_edge/platforms.py`
- Create: `src/robot_edge/hardware/go2_health.py`
- Create: `src/robot_edge/hardware/go2_telemetry.py`
- Modify: `src/robot_edge/hardware/mobile_base_health.py`（将 `SUPPORTED_PROFILES` 扩展到包含 `go2`，复用既有 `TelemetrySnapshot` 范式）
- Modify: `src/robot_edge/hardware/piper_health.py`（仅在需要时补充 torque/ driver 状态字段，不改注入范式）
- Create: `tests/robot_edge/test_go2_health.py`
- Create: `tests/robot_edge/test_go2_piper_profile.py`

**Step 1: 写失败测试**

使用 fake probe/ROS graph；workstation 测试中**不得导入** `unitree_sdk2py`、`piper_sdk`、`rclpy`。覆盖：Go2 断开、非站立状态、里程计/IMU 过期、`body_velocity` 非零、姿态超限、Piper CAN/driver 缺失、扭矩状态不明、TF 不完整、急停未绑定。任何一种情况都使 `go2_piper` 的硬件 authority 为 false。

**Step 2: 实现 read-only probes**

定义 `Go2SystemProbe`：`connected`、`standing`、`odometry`、`body_velocity`、`imu`、`body_orientation`、`local_stop_ready`；**不得具有 movement 方法**。数据来源为 Go2 ROS 2 bridge 的话题（`/odom`、`/imu`、`/joint_states` 等，话题名以 Task 1 inventory 为准），在 ROS 端 factory 中订阅；workstation 侧用注入的 fake。断开、过期和异常一律不得报告为 healthy。`platforms.py` 定义 `go2_piper` 平台装配（base=Go2 bridge, arm=Piper, perception=remote-service），作为 Robot Edge 唯一平台枚举来源。

**Step 3: 进行无运动真机验证**

只启动状态/遥测/TF 组件（含 Go2 ROS 2 bridge 的上行话题），检查 Go2、Piper、相机、TF、时钟、急停和错误语义。Piper 保持扭矩禁用。保存脱敏报告 `docs/validation/YYYY-MM-DD-go2-piper-readonly.md`。

**Step 4: 验证并提交**

```powershell
python -m unittest tests.robot_edge.test_go2_health tests.robot_edge.test_go2_piper_profile tests.robot_edge.test_hardware_health_mapping -v
git add src/robot_edge tests/robot_edge docs/validation
git commit -m "feat: add read-only Go2 Piper platform health"
```

**Acceptance:** Console/Robot Edge 能真实呈现 Go2+Piper 是否可服务；只读阶段不能运动。

## Task 3: Go2 底盘 bring-up 与 Kompass Go2 配置（非新 executor）

> **关键约束**：本 Task **不创建 `go2_adapter.py` 运动 adapter，不修改 `navigate_to_object_server.py`**。导航语义层、`TrackVisionTargetAdapter`、`cmd_vel_guard`、lease 全部原样复用。Go2 只是 `/cmd_vel` 的一个新消费者，与 LeKiwi 并列。

**Files:**
- Create: `deploy/go2-driver/compose.yaml`（拉起 Go2 ROS 2 bridge 容器，RMW/域与 ubrobot 容器一致）
- Create: `deploy/go2-driver/launch/go2_bringup.launch.py`（如 bridge 需 ROS 2 launch；否则记录外部启动方式）
- Modify: `deploy/emos/recipes/cortex_navigation/recipe.py`（Kompass 机器人模型与速度限值按 `UBROBOT_PLATFORM` 参数化；Go2 保守限速）
- Modify: `ros_depends_ws/src/ubrobot_navigation/ubrobot_navigation/policy.py`（为 `go2_piper` profile 增加 base velocity 限值，供 `sanitize_twist()` 使用）
- Create: `tests/robot_edge/test_go2_bringup_contract.py`
- Create: `docs/validation/YYYY-MM-DD-go2-navigation-cortex.md`

**Step 1: 写失败的 bring-up 契约测试**

断言配置层面的事实（不依赖真机）：Go2 bridge 容器订阅 `/cmd_vel`（`geometry_msgs/Twist`）并发布 `/odom`、`/imu`、`/joint_states`；`cmd_vel_guard` 存在且输出 `/cmd_vel`；Kompass `DriveManager` 输出 `/navigation/raw_cmd_vel`；`go2_piper` profile 的 base velocity 限值非 LeKiwi 默认值（如 `max_base_linear_mps=0.2`、`max_base_angular_radps=0.5`，具体以 inventory 为准）；`/navigation/command_lease` 心跳与 `cmd_vel_guard` 的 lease 门已接。bridge 容器与 ubrobot 容器 `RMW_IMPLEMENTATION`、`ROS_DOMAIN_ID` 一致。

**Step 2: 配置 Go2 bring-up 与 Kompass Go2 模型**

- 在 `deploy/go2-driver/compose.yaml` 中以容器运行 Go2 ROS 2 bridge（unitree_ros2，ROS 版本/RMW 以 Task 1 结论为准；若 bridge 仅 CycloneDDS，则**拓展坞内全部容器**改用 `rmw_cyclonedds_cpp`，Pi 那套 FastDDS 不受影响）。
- 在 `recipe.py` 中按 `UBROBOT_PLATFORM` 参数化 Kompass 机器人模型：Go2 保留 `RobotType.OMNI`（可横移），保守限速（线性 ≤0.2 m/s、角速度 ≤0.5 rad/s 起步），`loop_rate`/`control_time_step` 维持现有值或在 inventory 标注调整。
- 定义 **Go2 站起/sport-mode 生命周期**：`/cmd_vel` 仅在 Go2 已站起且处于 sport velocity 模式时有效。站起在 bring-up 阶段由 operator 手动或 bridge 自带原语触发（本计划不自动站起）；lease 释放只发零速度 = 停走但保持站立；stand-down 为会话结束的独立动作，不与 lease 挂钩。该生命周期写入 `docs/hardware/go2-piper-integration-inventory.md`。

**Step 3: 逐级硬件验证**

先验证零输出、stop、lease 过期与急停；再在空旷区域执行低速短距离导航；最后从 Cortex 提交"靠近 <目标>"。每轮只变更一个因素，记录 Cortex goal、Action feedback、odometry、停止延迟与视频索引。

**Step 4: 验证并提交**

```powershell
python -m unittest tests.robot_edge.test_go2_bringup_contract -v
colcon test --packages-select ubrobot_navigation
git add deploy/go2-driver deploy/emos ros_depends_ws/src/ubrobot_navigation tests docs/validation
git commit -m "feat: bring up Go2 base via Kompass cmd_vel (no nav server change)"
```

**Acceptance:** Cortex 能经**既有** `NavigateToObject` Action 完成受限 Go2 导航；`navigate_to_object_server.py` 无改动；浏览器、Console、Cortex 均无原始运动权限；Go2 站起/停止生命周期已定义并经验证。

## Task 4: 接入 Piper/GraspNet executor（感知远程、运动本地），并验证静止底盘抓取

**Files:**
- Create: `ros_depends_ws/src/ubrobot_manipulation/ubrobot_manipulation/executors/go2_piper.py`（`RemoteGraspPerception` + `PiperMotionBinding`）
- Modify: `ros_depends_ws/src/ubrobot_manipulation/ubrobot_manipulation/grasp_object_server.py`（`build_executor()` 在 `UBROBOT_GRASP_PLATFORM=go2_piper` + `UBROBOT_GRASP_EXECUTOR=hardware` + profile health 全通过时构造真实 binding）
- Modify: `ros_depends_ws/src/ubrobot_manipulation/ubrobot_manipulation/policy.py`（`go2_piper` profile 增加远程感知服务 URL 与 base velocity 字段）
- Modify: `deploy/emos/recipes/cortex_navigation/recipe.py`（`GRASP_TOOL_DESCRIPTION` 补充限制，见 Step 3）
- Create: `ros_depends_ws/src/ubrobot_manipulation/test/test_go2_piper_executor_contract.py`
- Create: `docs/validation/YYYY-MM-DD-go2-piper-stationary-grasp.md`
- 可能 Modify: `src/service/reasoning/http_reasoning_server.py`（补 `/grasp_poses` 端点契约，见 Step 3）

**Step 1: 写失败的 executor contract 测试**

覆盖 perception->规划->预抓取->接近->夹爪->撤退 phase feedback；测试可达工作空间、最大接近速度、取消、超时、异常、Piper stop acknowledgement。以下任一条件应在 `start` 前拒绝或在过程中取消：navigation lease 存在、Go2 非静止、姿态异常、TF/相机过期、Piper status 异常。远程感知服务不可达/契约不符应 fail-closed 拒绝（不降级为本地推测）。

**Step 2: 实现真实 binding，仅在 robot factory 延迟加载**

- `RemoteGraspPerception`（实现 `PerceptionInterface.locate_grasp_poses`）：HTTP 客户端，POST `{color, depth, camera_intrinsic, workspace, target}` 到远程 `http_reasoning_server` 的抓取端点，解析返回为 `GraspCandidate` 列表。超时/错误即抛异常（fail-closed）。不在本地导入 torch/graspnetAPI。
- `PiperMotionBinding`（实现 `MotionInterface.execute_grasp` + `hold_position`）：将 6-DOF grasp 位姿经 IK 解算（pinocchio + `assets/urdf/piper_description.urdf`，或 `piper_moveit`）为关节角，经 `piper_sdk_interface.set_joint_positions_deg()` + `GripperCtrl()` 执行；`hold_position()` 维持当前关节。**不直接调用 `piper_ctrl_single_node.py` 的 CAN 对象，不让 Cortex 选择关节目标。**
- `build_executor()` 仅在 `UBROBOT_GRASP_PLATFORM=go2_piper`、`UBROBOT_GRASP_EXECUTOR=hardware`、且 Task 2 的 profile health 全通过时创建真实 binding；其余维持现有 fixture/`NotImplementedError` 行为。真实 SDK 导入仅在 factory 中延迟发生。

**Step 3: 对齐远程感知服务契约与受控开关**

- 确认 `http_reasoning_server.py` 是否已有返回抓取位姿的端点；若无或契约不符，在其上补一个 `/grasp_poses` 端点（入参：RGBD + 内参 + workspace + target；出参：抓取位姿列表）。该改动在 x86 服务器侧，不在拓展坞；Task 1 inventory 必须记录该端点 URL 与契约已验证。
- 仅在上述 server 已部署、健康检查和 fixture e2e 通过后设置 `CORTEX_ENABLE_GRASP=true`。`GRASP_TOOL_DESCRIPTION` 限定：仅用于底盘静止且健康时的固定底盘抓取；不得用于移动底盘。Cortex 自动"先导航后抓取"顺序编排属产品化，本轮不实现。

**Step 4: 验证并提交**

```powershell
python -m unittest ros_depends_ws.src.ubrobot_manipulation.test.test_go2_piper_executor_contract -v
colcon test --packages-select ubrobot_manipulation ubrobot_interfaces
git add ros_depends_ws/src/ubrobot_manipulation deploy/emos tests docs/validation src/service/reasoning
git commit -m "feat: bind Go2 Piper grasp (remote perception, local motion)"
```

**Acceptance:** Cortex 可调用 `GraspObject`，但只能在 Go2 已安全停稳且动作健康时执行固定底盘抓取；感知失败 fail-closed，不产生运动。

## Task 5: 共享仲裁与 Cortex 顺序编排（产品化，本轮暂缓）

> **本轮不实施。** 安全底线已由既有 `authority.py` / `lifecycle.py` 提供（fail-closed：lease 存在或底盘不静止则拒绝抓取；抓取中 lease 出现即取消），操作者分别手动调用 `NavigateToObject` 与 `GraspObject` 已安全。下列产品化事项留待后续，见文末"产品化（后续）"：
> - 统一 `motion_arbitration.py` 共享 authority source（含 Go2 `body_velocity` 上行、settling 连续静止窗口）；
> - Cortex 系统提示自动"先导航后抓取"顺序编排；
> - `tests/e2e/test_go2_piper_cortex_fixture.py` 链式任务与六条失败路径。

## Task 6: 联合真机验收与部署固化

**Files:**
- Create: `deploy/robot-edge/compose.go2-piper.hardware.yaml`
- Create: `scripts/hardware/validate_go2_piper_cortex.sh`
- Create: `tests/hardware/test_go2_piper_cortex_acceptance.py`
- Create: `docs/validation/YYYY-MM-DD-go2-piper-cortex-combined.md`
- Modify: `deploy/robot-edge/README.md`

**Step 1: 部署前门槛**

`compose.go2-piper.hardware.yaml` 必须要求 `UBROBOT_PLATFORM=go2_piper`、`UBROBOT_EDGE_MODE=hardware`、`UBROBOT_EDGE_HARDWARE_AUTHORITY=true`、已审核 checklist、绑定的本地急停、只读报告和配置 hash。RMW 与 `ROS_DOMAIN_ID` 按 Task 1/Task 3 结论设置；远程感知服务 URL 作为必需环境变量。缺任一项则拒绝启动控制 executor；不得使用 `privileged` 作为绕过条件。

**Step 2: 分阶段试验**

按顺序执行：只读健康->零输出/停止->低速导航->静止预抓取->轻量抓取。两个 Action **分别**验证（操作者手动依次调用，不做自动链式编排）。每次只测一种故障注入：正常取消、lease 失效、Console/Edge/Cortex 断连、本地急停、物理急停。另测互斥安全：导航 lease 活跃时调用 `GraspObject` 应被拒；抓取进行时调用 `NavigateToObject` 应使抓取 fail-closed 取消。远程感知服务断连作为单独一轮，验证 fail-closed 不产生运动。

**Step 3: 验收与回滚**

报告记录 commit、镜像/tag、profile/config hash、RMW/域、操作者、观察员、目标、限速、状态转移、Action feedback、stop latency、视频引用和已知限制。失败时回滚为 `CORTEX_ENABLE_GRASP=false` 且 `UBROBOT_EDGE_HARDWARE_AUTHORITY=false`；不得保留激活任务、lease 或扭矩。

**Acceptance:** `NavigateToObject` 与 `GraspObject` 可在 Go2+Piper 上分别由 Console/Cortex 安全完成；既有 fail-closed 互斥在真机可重复验证（一方运行时另一方被拒或取消）；所有停止路径可重复验证；远程感知断连不产生运动。

## 实施顺序与 SO101 的位置

先完成本计划 Task 1–4、Task 6；Task 5 为产品化，本轮暂缓。SO101 不与 Go2+Piper 并行改动；待本计划的 bring-up 模板、健康检查、权限 gate 和验收模板稳定后，再复用该模板实现 `so101_station`，从而避免两种机械臂适配层分叉。

## 产品化（后续，本轮不实施）

本轮仅保证两个语义 Action 各自安全可用。以下产品化事项留待后续：

1. **统一仲裁器** `src/robot_edge/motion_arbitration.py`：以 `authority.py` 的 `AuthorityTracker` 为唯一 authority source，纳入 Go2 `body_velocity`（`/odom` 上行）、IMU 姿态、Piper 执行状态与安全 latch，状态机 `idle -> navigating -> settling -> manipulating -> idle`，任一 stale/stop 锁存急停；导航结束后需满足连续静止窗口才允许抓取。
2. **Cortex 顺序编排**：在 `recipe.py` 系统提示中加入"先导航、等待成功、再抓取；失败/取消不继续；只用被发现的工具"。
3. **链式 fixture E2E** `tests/e2e/test_go2_piper_cortex_fixture.py`：验证"靠近桌上杯子并抓取"成功路径，以及导航失败、抓取拒绝、抓取中 lease 出现、UI cancel、`safety.stop`、Cortex cancel 六条失败路径。
4. 重新评估 `GRASP_TOOL_DESCRIPTION` 增加"导航完成后"前置与 settling 窗口的强约束。

## 停止条件

- 拓展坞无法 `docker run ros:jazzy-ros-base-noble`（aarch64），且无等效 Noble 容器路径；
- Go2 接口实为 ROS 1 且 `ros1_bridge` 在拓展坞不可用或不可稳定运行；
- dock 内 RMW 无法统一（unitree bridge 与 ubrobot 容器无法同 RMW 同域通信）；
- 远程感知服务不可达，或其抓取位姿契约不符且无法在服务器侧补齐；
- 旧 Go2/Piper 测试实现无法确定其控制入口、版本或停止原语（其中 Go2 运动入口必须为 `/cmd_vel`，直接 SportClient 不再接受）；
- 本地/物理急停未接入或未验证；
- Go2 静止、姿态、TF、相机或 Piper 健康任一证据不可靠；
- Cortex 能发现 raw control 工具，或真实 executor 在 fixture/只读模式被加载；
- 既有 `authority.py`/`lifecycle.py` 的 fail-closed 互斥在 Go2+Piper 上验证失败（lease 存在/底盘不静止仍允许抓取，或抓取中 lease 出现未取消）。
