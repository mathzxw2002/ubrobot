# 多机器人本体支持 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在不改变上层任务语义与安全边界的前提下，使 UBRobot 以部署时选择的平台 profile 支持 SO101 单臂和 Unitree Go2 + Piper 移动操作平台，并保留已验证的 LeKiwi 路径。

**Architecture:** 保持 `Operator Console -> Robot Edge -> ROS/厂商驱动` 边界。上层只发现版本化的能力、遥测和语义动作；每个本体在 Robot Edge 和 ROS 侧实现受限的 profile/adapter，不让 SDK、ROS 消息或原始速度/关节命令穿透到 Console。抓取继续使用 `/ubrobot/manipulation/grasp_object`，移动操作平台通过共享运动仲裁保证“行走”和“机械臂运动”互斥。

**Tech Stack:** Python 3.10、Pydantic contracts、FastAPI Robot Edge、ROS 2 Jazzy、LeRobot SO101 host/client、Unitree SDK2/ROS bridge（待确认）、Piper ROS/SDK、pytest/unittest、Docker Compose。

---

## 范围、原则与里程碑

| 里程碑 | 结果 | 是否允许运动 |
|---|---|---|
| M0 | 统一平台清单、配置和契约测试 | 否 |
| M1 | SO101 Ubuntu 主机只读接入与遥操作闭环 | 仅经人工确认的单臂低速测试 |
| M2 | SO101 采集、回放与策略评估可复现 | 单臂、固定工位 |
| M3 | Go2 + Piper 只读集成、坐标与安全互锁验证 | 否 |
| M4 | Go2 低风险移动与 Piper 固定底座抓取分别验证 | 不联合 |
| M5 | Go2 + Piper 受限联合任务验收 | 是，逐级放权 |

所有 profile 默认 `hardware_authority=false`。每一次从只读到可运动的切换，都必须有独立验证报告、物理急停和人工批准；成功的 Mock/Fixture 测试不能作为硬件验收证据。

## 目标平台定义

| profile | 控制主体 | 首期能力 | 关键互锁 |
|---|---|---|---|
| `lekiwi` | ROS 2 LeKiwi base | navigation、observation、stop | 现有速度 guard/lease |
| `so101_station` | Ubuntu 上的 LeRobot SO101 host | observation、grasp、stop | 单一控制 owner、串口断连即停止、软限位 |
| `piper_station` | Piper 固定工位 | observation、grasp、stop | CAN/驱动健康、扭矩状态、工作空间 |
| `go2_piper` | Unitree Go2 + Piper | navigation、observation、grasp、stop | 行走与抓取互斥、底盘静止确认、机身姿态/急停 |

`follow` 不在本计划的硬件放权范围内；只有在 navigation 与 grasp 验收完成后再独立立项。

### Task 1: 冻结多本体 profile 契约与部署配置（M0）

**Files:**
- Modify: `src/ubrobot_contracts/capabilities.py`
- Modify: `src/ubrobot_contracts/telemetry.py`
- Modify: `src/robot_edge/hardware/mobile_base_health.py`
- Create: `src/robot_edge/platforms.py`
- Create: `deploy/robot-edge/config/platforms.example.yaml`
- Create: `tests/robot_edge/test_platform_profiles.py`

**Step 1: 写失败测试**

覆盖下列不变量：

- profile 名称只接受 `lekiwi`、`so101_station`、`piper_station`、`go2_piper`；
- 缺少 profile、重复资源（如两个 CAN 控制者）、未知 profile 均启动失败；
- SO101 暴露关节/相机遥测但不暴露 navigation；
- `go2_piper` 暴露 navigation 与 grasp，但默认两个 capability 均无硬件控制权；
- 任何硬件 profile 断开时返回 `disconnected`，不沿用旧遥测假装可用。

Run: `python -m unittest tests.robot_edge.test_platform_profiles -v`

Expected: FAIL，因为 profile registry 尚不存在。

**Step 2: 实现最小 profile registry**

在 `src/robot_edge/platforms.py` 定义冻结的 `PlatformProfile`，至少包含：`name`、`capabilities`、`required_resources`、`telemetry_channels`、`motion_domains`、`default_authority`。从 YAML 加载部署选择，但不允许 YAML 覆盖安全默认值或能力名称。

`mobile_base_health.py` 不能再把 `go2` 作为“未来拒绝项”隐藏在代码中：改为由 profile registry 明确报告 `unavailable`，直到 Go2 读写分离 adapter 实现完成。

**Step 3: 验证并提交**

Run:
```powershell
python -m unittest tests.robot_edge.test_platform_profiles tests.robot_edge.test_hardware_health_mapping -v
git diff --check
git add src/ubrobot_contracts src/robot_edge deploy/robot-edge/config tests/robot_edge
git commit -m "feat: add explicit robot platform profiles"
```

**Acceptance:** 一个部署只能选择一个顶层平台；能力、资源、遥测通道和 authority 均可由 profile 解释且有测试。

### Task 2: SO101 Ubuntu 主机的只读 Robot Edge adapter（M1）

**Files:**
- Create: `src/robot_edge/hardware/so101_health.py`
- Create: `src/robot_edge/hardware/so101_telemetry.py`
- Create: `src/robot_edge/adapters/so101_host.py`
- Modify: `src/robot_edge/app.py`
- Create: `tests/robot_edge/test_so101_health.py`
- Create: `tests/robot_edge/test_so101_host_adapter.py`
- Create: `scripts/hardware/so101_preflight.sh`
- Create: `docs/validation/YYYY-MM-DD-so101-readonly.md`

**Step 1: 写失败测试**

使用 fake host client，不导入 `lerobot` 或串口库。验证：主机离线、host ID 不匹配、关节数量不匹配、相机时间戳过期、串口未发现分别生成真实的 `disconnected`/`unavailable`/`stale` 状态；测试还须断言只读路径没有 `connect`、`send_action`、扭矩使能或运动调用。

**Step 2: 实现只读协议和 adapter**

以小协议封装远程主机：`health()`、`joint_state()`、`camera_metadata()`、`close()`。将结果映射到 JSON-safe DTO；原始图像只走受控预览流，不进入状态 JSON。Ubuntu 上保持既有 `lerobot ... so101_host` 启动方式，先只增加一个经过认证的本地 bridge，不修改上层 Console。

**Step 3: 执行硬件 preflight（无运动）**

在 SO101 Ubuntu 主机上执行只读检查：OS/Python/LeRobot 版本、USB stable path、设备 ID、相机、网络、时间同步、软限位配置、物理急停/断电方式。报告中脱敏 IP、序列号和 token。不得连接 follower 控制循环或发送动作。

**Step 4: 验证并提交**

Run:
```powershell
python -m unittest tests.robot_edge.test_so101_health tests.robot_edge.test_so101_host_adapter -v
git add src/robot_edge tests/robot_edge scripts/hardware docs/validation
git commit -m "feat: add read-only SO101 robot edge adapter"
```

**Acceptance:** Console 能显示 SO101 状态与过期语义；硬件控制权仍为 false。

### Task 3: SO101 受限遥操作、数据闭环与策略基线（M1/M2）

**Files:**
- Create: `src/robot_edge/executors/so101.py`
- Modify: `src/robot_edge/safety.py`
- Modify: `examples/so101_to_so101/teleoperate_networked.py`
- Modify: `examples/so101_to_so101/record.py`
- Create: `tests/robot_edge/test_so101_executor.py`
- Create: `tests/e2e/test_so101_fixture.py`
- Create: `docs/validation/YYYY-MM-DD-so101-teleop-and-dataset.md`

**Step 1: 写失败状态机测试**

测试单一 arm lease、软关节限位、每步速度/增量限幅、命令 TTL、取消、host 断开、Robot Edge 退出及 `safety.stop`。每个失败路径均须向 host 发送安全停止/断连，而不是等待上层 UI。

**Step 2: 实现最小 SO101 executor**

仅支持关节目标的受限增量命令；不把原始控制端口暴露给 Console。配置中固定最大速度、最大增量、命令刷新时间和软限位；所有值需经代码定义的上限二次夹紧。先在 fixture 中验证，再在空载、低速、人工在场条件下进行单关节和回零位测试。

**Step 3: 固化数据协议**

将 `examples/so101_to_so101/record.py` 改为读取 profile 配置和运行元数据（机器人 profile、相机标定版本、任务、操作者匿名 ID、代码提交、控制频率）。每段数据先离线回放并通过 schema/帧率/动作范围检查，才进入训练集。保留原始数据不可变，分离训练/验证/测试任务场景。

**Step 4: 评估策略**

先使用固定的 3–5 个桌面任务和成功判据，比较 teleop 回放、规则基线和已有 ACT/LeRobot policy。策略评估必须使用与训练集隔离的物体摆放、光照或初始位姿；策略输出仍经过 SO101 executor 的限幅和急停。

**Acceptance:** 形成可复现记录→校验→训练/推理→受限执行链路；SO101 单臂验收不授予任何移动本体权限。

### Task 4: Go2 + Piper 的只读发现、坐标树和安全互锁（M3）

**Files:**
- Create: `src/robot_edge/hardware/go2_health.py`
- Create: `src/robot_edge/hardware/go2_telemetry.py`
- Create: `src/robot_edge/adapters/go2_piper.py`
- Modify: `src/robot_edge/hardware/mobile_base_health.py`
- Modify: `ros_depends_ws/src/ubrobot_manipulation/ubrobot_manipulation/policy.py`
- Modify: `ros_depends_ws/src/ubrobot_manipulation/ubrobot_manipulation/authority.py`
- Create: `tests/robot_edge/test_go2_health.py`
- Create: `tests/robot_edge/test_go2_piper_profile.py`
- Create: `docs/validation/YYYY-MM-DD-go2-piper-readonly.md`

**Step 1: 写失败测试**

验证 Go2 SDK/ROS bridge 不在 workstation 或 fixture import；缺失站立状态、里程计、IMU、机身姿态、Piper CAN/扭矩关闭状态任一项时，`go2_piper` 的执行 authority 均为 false。验证 navigation lease 活跃、底盘速度超过阈值、机身姿态异常时 `grasp_may_start` 为 false。

**Step 2: 实现 read-only Go2 adapter**

通过注入 probe/ROS graph 读取 Go2 的连接、站立模式、里程计、线速度、IMU、姿态和本地急停状态；不调用 sport/client 控制接口。Piper 延续 `PiperHealth` 的只读约束。确认真实 ROS topic/service/action 名称后写进已审核的 profile 配置，不凭猜测硬编码。

**Step 3: 标定与坐标系审计**

建立并记录 `map/odom -> base -> piper_base -> camera -> tool` 坐标树、时间戳来源、外参与单位。用固定标志物检查手眼变换和重力方向；任何 TF 丢失、超过阈值的时间偏差或标定版本不一致都必须阻止抓取计划。

**Step 4: 验证并提交**

Run:
```powershell
python -m unittest tests.robot_edge.test_go2_health tests.robot_edge.test_go2_piper_profile ros_depends_ws.src.ubrobot_manipulation.test.test_policy -v
git add src/robot_edge ros_depends_ws/src/ubrobot_manipulation tests/robot_edge docs/validation
git commit -m "feat: add read-only Go2 Piper platform profile"
```

**Acceptance:** 实机只读报告能同时证明 Go2、Piper、相机、TF 和急停状态；没有任何运动或扭矩使能。

### Task 5: 分离验证 Go2 导航和 Piper 抓取（M4）

**Files:**
- Create: `src/robot_edge/executors/go2_navigation.py`
- Create: `ros_depends_ws/src/ubrobot_manipulation/ubrobot_manipulation/executors/piper.py`
- Modify: `ros_depends_ws/src/ubrobot_manipulation/ubrobot_manipulation/grasp_object_server.py`
- Modify: `src/robot_edge/safety.py`
- Create: `tests/robot_edge/test_go2_navigation_executor.py`
- Create: `ros_depends_ws/src/ubrobot_manipulation/test/test_piper_executor_contract.py`
- Create: `docs/validation/YYYY-MM-DD-go2-navigation.md`
- Create: `docs/validation/YYYY-MM-DD-piper-stationary-grasp.md`

**Step 1: Go2 导航先写失败测试**

验证 lease 过期、网络断开、姿态异常、低电量/驱动异常、本地急停和外部 `safety.stop` 都立即取消控制并调用本地 stop；测试速度/角速度上限以及一次只允许一个 navigation owner。

**Step 2: Go2 低风险硬件验收**

先验证零指令、停止、lease 超时和急停，再在清空场地中以经过批准的极低速度/距离执行直行、转向、取消和失联试验。禁止在 Piper 扭矩开启或抓取任务存在时移动。

**Step 3: Piper executor 先写假驱动测试**

为 start、feedback、cancel、result、stop、关节/工作空间限制、扭矩关闭和超时写 contract test。只在 robot-side factory 延迟导入 `piper_sdk`；先无载低速，再单姿态，最后使用柔软物体的轻量抓取。执行期间保持 Go2 足端站立、速度为零且无 navigation lease。

**Acceptance:** 两条运动能力分别通过安全报告；这不是联合移动操作批准。

### Task 6: Go2 + Piper 联合任务编排与回归测试（M5）

**Files:**
- Create: `src/robot_edge/motion_arbitration.py`
- Modify: `src/robot_edge/runtime.py`
- Modify: `ros_depends_ws/src/ubrobot_manipulation/ubrobot_manipulation/lifecycle.py`
- Modify: `ros_depends_ws/src/ubrobot_manipulation/ubrobot_manipulation/grasp_object_server.py`
- Create: `tests/robot_edge/test_motion_arbitration.py`
- Create: `tests/e2e/test_go2_piper_fixture.py`
- Create: `tests/hardware/test_go2_piper_acceptance.py`
- Create: `docs/validation/YYYY-MM-DD-go2-piper-combined.md`

**Step 1: 定义并测试仲裁状态机**

实现明确状态：`idle -> navigating -> settling -> manipulating -> idle`，以及任意状态到 `emergency_stopped`。`navigating` 与 `manipulating` 永不可重叠；只有连续静止窗口、姿态在限值内、Piper/感知健康、导航 lease 已释放时才可进入 `manipulating`。所有状态转移写结构化审计事件。

**Step 2: 端到端 fixture 测试**

从 Console 提交“到近处目标再抓取”的语义任务，验证 capability snapshot、lease、Go2 停稳、抓取 feedback、正常取消、UI 急停、本地急停、浏览器重连和 Edge 重连。测试中使用 fake executor，不能接入任何厂商 SDK。

**Step 3: 分阶段真机验收**

先执行“导航→停止→等待→仅预抓取姿态”，再执行低风险抓取；每一类故障注入单独试验，记录版本、配置哈希、视频索引、遥测、动作/停止延迟和人工观察。出现意外姿态、遥测陈旧、校准不一致或任一停止路径失败即停止升级。

**Acceptance:** 一个经过批准的、低风险的导航后抓取任务成功，且所有停止和互斥测试可复现。

### Task 7: 文档、CI 矩阵与发布准则（贯穿 M0–M5）

**Files:**
- Modify: `README.md`
- Create: `docs/platforms.md`
- Create: `docs/hardware/so101-ubuntu.md`
- Create: `docs/hardware/go2-piper.md`
- Modify: `scripts/validate_operator_console.ps1`
- Create: `.github/workflows/platform-contracts.yml`（若仓库启用 GitHub Actions）

**Step 1: 编写平台矩阵文档**

记录每个 profile 的硬件、OS、ROS/SDK、启动命令、能力、遥测、控制资源、急停方案、标定版本、已知限制和验收状态。机密、IP 和设备序列号不得入库。

**Step 2: 建立分层 CI**

工作站必跑 profile/contract/fixture/仲裁测试；ROS 测试在 ROS 容器中运行；硬件测试默认跳过，仅由显式环境变量和审核报告启用。每个 PR 检查 `git diff --check` 与不导入厂商 SDK 的 fixture 保护测试。

**Acceptance:** 新增机器人遵循“profile + 只读遥测 + executor contract + fixture + 硬件 gate”的模板，不复制 Console 或任务语义代码。

## 建议节奏与资源依赖

1. 第 1 周：Task 1；同时收集 SO101 Ubuntu 主机、Go2/Piper 的设备清单和急停方案。
2. 第 2–3 周：Task 2–3，优先把 SO101 变成第一个完整单臂参考实现，沉淀数据协议。
3. 第 4 周：Task 4，只读完成 Go2+Piper 的传感、TF、姿态和安全互锁，不运动。
4. 第 5–6 周：Task 5，分离验收 Go2 与 Piper。
5. 第 7 周起：Task 6 的 fixture 与分阶段真机验证；每次权限升级以验收报告为门槛。

## 实施前必须由负责人确认

- SO101 Ubuntu 主机的 LeRobot 版本、稳定串口路径、相机型号及物理断电/急停方式；
- Unitree 具体型号、官方 SDK/ROS bridge 版本、控制接口、站立/停止原语及本地急停；
- Piper 型号/固件、CAN 接口、机械安装方式、工作空间和 Go2 机身上的重量/重心约束；
- 三个平台各自的 ROS domain、网络拓扑与时间同步方案；
- 首个验收任务的场地、速度/力限制、观察员和允许的物体。

## 不可突破的停止条件

- 任何 profile 的设备、急停、驱动或坐标系未确认；
- 只读模式意外初始化运动 SDK、扭矩或控制连接；
- 断网、lease 失效、遥测陈旧或取消不能 fail-closed；
- Go2 未确认静止/稳定时开始 Piper 动作；
- 无单独硬件报告就把 fixture 成功标为硬件可用。

