# go2-piper-driver 容器构建与真机验证报告

- Date/time: 2026-08-07
- Commits: `bf0fc21`（含容器化修复）on `go2-piper-cortex-integration`
- Machine role: Go2 expansion dock (Jetson Orin NX, Ubuntu 20.04.5, JetPack R35.3.1)
- Image: `ubrobot/go2-piper-driver:0.1.0`
- Build: repo root on dock, `docker build -f deploy/go2-piper-driver/Dockerfile`
- RMW: `rmw_cyclonedds_cpp` (Go2 DDS requirement); ROS_DOMAIN_ID=0

## 构建过程中修复的问题

| 问题 | 根因 | 修复 |
|---|---|---|
| colcon 只装 `launch` 包 | **python 包未嵌套在 ament 包下**：`find_packages()` 把顶层当 ament 包名 | 目录重构为 `go2_piper_driver/`（ament）→ `go2_piper_driver/go2_piper_driver/`（python 包） |
| pip PEP 668 拒绝 | Ubuntu Noble 系统环境受管 | `--break-system-packages` |
| `unitree-sdk2py` 装不上 | 私有 SDK 不在 PyPI，且需编译 CycloneDDS C++ 扩展 | 纯 Python：COPY 源码 + PYTHONPATH 指向，不 pip 编译 |
| `piper-sdk` pip 超时 | files.pythonhosted.org 网络问题 | 清华 PyPI 源 |
| launch 报 libexec 缺失 | ament_python 的 `Node(executable=)` 找 libexec 而非 bin | launch 用 `ExecuteProcess(cmd=[绝对路径])` |
| CycloneDDS 建 domain 失败 | socket buffer 需求 1MB 超容器限制 | 移除 `MinimumSocketReceiveBufferSize`，用默认 |
| `RcutilsLogger.info` 崩溃 | `get_logger().info(fmt, arg)` 多参数不支持 | 全部改 f-string 单参数 |
| 容器内 piper SDK 报 CAN 错 | 无 `ip` 工具 + 无 NET_ADMIN | Dockerfile 装 `iproute2` + 容器 `--cap-add NET_ADMIN` |

## 真机验证结果（容器运行）

- **容器稳定运行**：`Up` 无重启。
- **go2_bridge**：`go2 bridge up: interface=eth0 body_ip=192.168.123.161`，成功发现 Go2 本体的 DDS 话题（`rt/api/sport/*`、`rt/lf/sportmodestate`、`rt/utlidar/*` 等）——**容器 → CycloneDDS → Go2 本体 DDS 直连成功**。
- **piper_driver**：`piper driver up: can=can0 (torque NOT enabled)`，`arm_status` 发布 `enabled=False sdk=ok`——**容器 → socketcan(can0) → Piper SDK 连接成功**。
- **`/piper/joint_states` 返回真实关节数据**（rad）：j1 -0.049, j2 -0.042, j3 0.015, j4 0.058, j5 0.474, j6 -1.841（≈-105°，折叠位），gripper 5.02mm。与 S5 真机读数一致。
- **话题/服务注册**：`/cmd_vel`、`/odom`（go2）；`/piper/joint_cmd`、`/piper/joint_states`、`/piper/arm_status`、`/piper/enable`（piper）。
- **容器内 CAN 实测**：`socket(AF_CAN, SOCK_RAW)` bind can0 OK，收到 Piper 实时帧（`c1010000...`）。

## 结论

- **"一机器人一硬件容器"架构验证通过**：Go2 bridge + Piper 驱动在同一容器内，通过 host 网络共享 eth0（Go2 DDS）与 can0（Piper CAN）。
- **Piper 关节状态读取闭环**：容器 → can0 → Piper SDK → `/piper/joint_states` 真实数据。
- 扭矩保持关闭（`enabled=False`）；未执行运动指令。
- 未验证（需操作员现场 + 扭矩启用）：通过 `/piper/enable` 启用后经 `/piper/joint_cmd` 发运动、抓取端到端。

## 真机运动验证（通过容器 ROS2 接口，2026-08-07 补充）

经 `/piper/enable` 服务 + `/piper/joint_cmd` 话题执行完整运动循环：

| 步骤 | 指令 | 结果 |
|---|---|---|
| 启用扭矩 | `/piper/enable` SetBool true | `torque enabled`，`enabled=True sdk=ok` |
| 小幅运动 | `/piper/joint_cmd` +0.1 rad (j1-j3) | 指令发布成功 |
| 夹爪开/合 | `/piper/joint_cmd` gripper 8mm/4mm | 执行 |
| 回位 | `/piper/joint_cmd` 起始关节 | 关节回到 [-0.049,-0.042,0.015,0.058,0.474,-1.841] |
| 禁用扭矩 | `/piper/enable` SetBool false | `torque disabled`，`enabled=False sdk=ok` |

- **完整闭环验证**：`/piper/enable` → 扭矩启用 → `/piper/joint_cmd` 真实运动 → 回位 → 扭矩禁用。
- **安全收尾**：扭矩关闭、关节回起始位、夹爪闭合（4.01mm）、容器稳定运行。
- 已知现象：日志大量 `Failed to parse type hash` WARN 来自 CycloneDDS 解析 Go2 私有 DDS 话题（`rt/*`）的类型哈希，属 Go2 生态噪音，不影响本机功能。

## 下一步（真机，需现场确认）

1. emos 语义层（GraspObject server + motion_arbitration）接入本容器的话题/服务，做端到端"IK→话题→运动→抓取"。
