# LeKiwi 独立驱动容器设计

## 状态

已确认。2026-07-28，决定以独立容器承载 LeKiwi 底盘驱动，并以 C++
`ros2_control` 硬件插件作为正式运行内核。现有 Python
`lekiwi_base.py` 仅用于诊断、校准和一致性测试。

## 目标

将当前已经分别验证的 EMOS 控制输出和 LeKiwi 串口控制连接起来，同时保持：

- EMOS 感知/规划与底盘硬件故障隔离；
- 使用标准 ROS 2 接口，不引入 UDP、HTTP 等自定义控制协议；
- 上电、启动、超时、退出和故障时默认停车；
- 串口只有一个进程持有；
- 驱动可独立构建、测试、发布、重启和回滚；
- 后续能接入遥控、急停、Nav2、仿真和里程计融合。

## 当前约束

- 树莓派宿主机运行 Ubuntu 24.04.4 LTS 和 ARM64 Docker。
- ROS 2 Jazzy 完整运行环境分别由 EMOS 容器和 LeKiwi 驱动容器提供，
  不依赖宿主机安装完整 ROS 2。
- EMOS Recipe 在 `emos` 容器中运行，并由 `my_driver` 发布
  `geometry_msgs/msg/Twist` 类型的 `/cmd_vel`。
- EMOS 容器当前不映射 LeKiwi 串口，也不包含 `lerobot` 或 `ubrobot`。
- LeKiwi 三个轮电机 ID 为 7、8、9，通过 Feetech 总线连接。
- 当前稳定设备标识为
  `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A68011386-if00`。
- RTAB-Map 当前拥有 `/odom` 和 `odom -> base_link`，新驱动不能发布冲突 TF。

## 高层架构

```text
EMOS container
  Vision -> Controller -> DriveManager
                               |
                         /cmd_vel (Twist)
                               |
                         ROS 2 DDS / host network
                               |
LeKiwi driver container        v
  command adapter -> command arbitration / software stop
                               |
                    omni_wheel_drive_controller
                               |
                       controller_manager
                               |
                    LeKiwiSystemHardware (C++)
                               |
                         /dev/lekiwi-base
                               |
                    Feetech motors 7, 8, 9
```

两个容器使用同一个 `ROS_DOMAIN_ID` 和兼容的 RMW 实现，通过 ROS 2 DDS
通信。驱动容器独占串口；EMOS 容器没有串口访问权限。

## 组件

### `lekiwi_description`

提供底盘 URDF/Xacro、三个轮关节、轮半径、底盘半径、轮子安装角度和
`<ros2_control>` 配置。硬件参数必须来自配置，不能散落在代码中。

### `lekiwi_hardware`

实现 `hardware_interface::SystemInterface`：

- `on_init` 校验串口、电机和单位参数；
- `on_configure` 打开串口并探测 ID 7、8、9；
- `on_activate` 连续写入零速度后允许控制；
- `read` 读取三个轮子的实际速度；
- `write` 写入三个轮子的目标速度；
- `on_deactivate`、`on_error` 和 `on_shutdown` 尽力停车并释放资源。

硬件插件只转换 `rad/s <-> Feetech raw value`，不实现底盘运动学。

### `omni_wheel_drive_controller`

负责 `linear.x`、`linear.y`、`angular.z` 到三个轮速的运动学转换，并根据
轮速反馈生成 `/odom/wheel`。控制器必须启用命令超时和速度限制。

### `lekiwi_cmd_adapter`

第一版用于兼容 EMOS 当前发布的非时间戳 `Twist`：

- 拒绝 NaN、Inf 和不完整数据；
- 添加接收时间戳；
- 执行输入边界检查；
- 转发到全向轮控制器的标准参考接口。

后续 EMOS 原生输出标准带时间戳命令后，可删除此适配器。

### `lekiwi_bringup`

负责启动 `robot_state_publisher`、`controller_manager`、硬件插件、
`joint_state_broadcaster`、全向轮控制器、命令适配器和诊断节点。

## 仓库结构

```text
ros_depends_ws/src/
  lekiwi_description/
  lekiwi_hardware/
  lekiwi_bringup/

deploy/lekiwi-driver/
  Dockerfile
  compose.yaml
  entrypoint.sh
  healthcheck.sh
  README.md
```

驱动镜像基于 `ros:jazzy-ros-base-noble`，使用多阶段构建，仅复制运行所需
产物。ARM64 镜像使用语义化版本标签，正式部署固定镜像 digest。

## 运行和权限

- Docker 使用 `network_mode: host`，以便与 EMOS DDS 通信。
- 通过 udev 将串口稳定命名为 `/dev/lekiwi-base`。
- Compose 只映射 `/dev/lekiwi-base`，不使用 `privileged` 或整个 `/dev`。
- 进程以非 root 用户运行，加入对应的 `dialout` 数字 GID。
- 根文件系统只读，运行时目录使用 `tmpfs`。
- 丢弃全部默认 capabilities；确需实时优先级时只增加 `SYS_NICE`。
- 使用 `restart: unless-stopped`、`init: true` 和有限的退出宽限期。

## 安全要求

第一阶段参数：

| 项目 | 初始值 |
|---|---:|
| 控制频率 | 20 Hz |
| 命令超时 | 250 ms |
| 停车目标 | 500 ms 内 |
| `linear.x` | +/-0.05 m/s |
| `linear.y` | +/-0.05 m/s |
| `angular.z` | +/-0.20 rad/s |

必须满足以下不变量：

1. 上电、容器启动和硬件激活前命令均为零。
2. 无有效命令、命令过期、非有限数值或通信错误时命令为零。
3. 硬件未进入 `ACTIVE` 状态时不能接受非零命令。
4. EMOS 停止或 DDS 中断后，底盘在超时时间内停车。
5. 正常退出、ROS 生命周期切换和可处理异常均执行停车。
6. 容器外必须保留实体急停或独立硬件看门狗；不能依赖析构函数处理
   `SIGKILL`、内核崩溃或整机掉电。

## 话题和里程计

建议逐步采用以下命名：

```text
/cmd_vel/emos
/cmd_vel/teleop
/cmd_vel/safety
/lekiwi_base_controller/reference
/lekiwi_base_controller/odom
/joint_states
/diagnostics
/lekiwi/estop
```

第一阶段将轮式里程计重映射到 `/odom/wheel`，且禁止其发布
`odom -> base_link`，避免与当前 RTAB-Map 冲突。后续引入
`robot_localization`，融合 `/odom/wheel` 和 `/odom/visual`，由融合节点唯一
发布主 `/odom` 和 TF。

## 故障模式

| 故障 | 期望行为 | 验证方式 |
|---|---|---|
| EMOS 停止发布 | 250 ms 后停车 | 停止 Recipe |
| DDS 断连 | 250 ms 后停车 | 停止 EMOS 容器 |
| 非法速度 | 拒绝并停车 | 发布 NaN/Inf/越界数据 |
| 串口超时 | 硬件进入错误状态并停车 | 故障注入/拔线 |
| 单个电机离线 | 整个底盘停止 | 屏蔽一个电机 ID |
| 驱动正常退出 | 写零并关闭串口 | `docker stop` |
| 驱动被 `SIGKILL` | 软件不保证停车 | 验证硬件看门狗/急停 |
| 容器更新失败 | 回滚到固定旧镜像 | 镜像回滚演练 |

## 测试和发布阶段

### A. 纯软件和 Mock Hardware

- 验证 Recipe 输出到三个虚拟轮速的完整 ROS 链路；
- 用 Python 实现生成黄金输入/输出样本，验证 C++ 原始值转换；
- 覆盖运动学方向、单位、限速、超时和非法消息；
- CI 构建 ARM64 镜像并运行无硬件测试。

### B. 真实硬件零速度

- 映射稳定串口；
- 探测三个电机并读取轮速；
- 全程只写零速度；
- 验证启动、停止、EMOS 重启和 DDS 中断。

### C. 轮子架空

- 每个轮子分别进行低速正反转；
- 校验电机方向、速度单位和轮式里程计符号；
- 测试前进、横移、旋转、超时和软件急停；
- 注入进程崩溃、串口断开和单电机离线故障。

### D. 地面低速和发布

- 维持初始限速进行地面测试；
- 验证手动控制优先于 EMOS；
- 完成运行手册、回滚手册和故障排查表；
- 通过验收后再分级提高速度。

## 验收标准

- `/cmd_vel` 到硬件写入的本机 p95 延迟低于 50 ms；
- 连续运行期间控制循环无持续丢周期；
- 所有可处理的软件故障在 500 ms 内产生零速度；
- EMOS 可独立重启，驱动容器不重启且底盘保持安全；
- 串口不能被第二个进程打开；
- 三轮反馈、控制器、硬件生命周期和诊断状态可查询；
- 镜像可在干净树莓派环境重复部署和回滚。

## 暂不包含

- 机械制动或安全认证；
- 高速运动参数调优；
- SO-101 机械臂驱动；
- 将视觉推理移入驱动容器；
- Kubernetes 或跨机器底盘控制。
