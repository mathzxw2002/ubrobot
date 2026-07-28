# ADR-0001：LeKiwi 使用独立 ros2_control 驱动容器

## 状态

Accepted，2026-07-28。

## 背景

EMOS Recipe 已能产生 `/cmd_vel`，现有 Python `lekiwi_base.py` 也已验证能通过
串口连接 LeKiwi、发送零速度和读取轮速，但两条链路尚未连接。

EMOS 感知与规划负载较高，且经常需要重启 Recipe、更新 Kompass 或重建环境。
底盘硬件驱动需要稳定生命周期、唯一串口所有权、超时停车和独立发布/回滚。

## 决策

采用一个独立的 `lekiwi-base-driver` Docker 容器。容器内使用 ROS 2 Jazzy、
`controller_manager`、官方全向轮控制器和自研 C++
`hardware_interface::SystemInterface` 插件。

EMOS 与驱动容器仅通过标准 ROS 2 DDS 话题通信。EMOS 无权访问底盘串口。
现有 Python `lekiwi_base.py` 不进入生产控制循环，仅保留作诊断、校准和 C++
实现的一致性测试基准。

## 结果

### 正面

- 感知/规划与硬件驱动可独立更新、重启和回滚；
- 使用 ROS 2 标准硬件生命周期和控制器接口；
- 不需要自定义 UDP 或 HTTP 控制协议；
- 运动学、硬件协议和部署职责边界清晰；
- 后续可以接入遥控、Nav2、仿真、诊断和里程计融合。

### 负面

- 需要实现和维护 C++ Feetech 硬件插件；
- 增加一个镜像、Compose 服务和 ARM64 构建流程；
- 必须解决 USB 热插拔、容器 GID 和设备稳定命名；
- 第一版开发量高于直接复用 Python 类。

### 中性

- Python 和 C++ 驱动会在一段时间内并存，但不能同时打开串口；
- EMOS 当前的 `Twist` 输出需要一个临时标准化适配器。

## 备选方案

### 将 Python `lekiwi_base.py` 放进 EMOS 容器

未采用。实现较快，但把视觉/规划重启与底盘生命周期绑定，并扩大 EMOS 的
USB 权限和依赖范围。

### Docker 到宿主机 Python 的 UDP 桥

未采用为正式方案。适合原型验证，但增加自定义协议、双进程状态和额外故障面。

### 宿主机原生 ros2_control

保留为可回退方案。故障隔离最直接，但不符合本阶段统一容器化部署目标。

## 参考

- [ROS 2 Control 架构](https://control.ros.org/jazzy/doc/getting_started/getting_started.html)
- [编写 ros2_control 硬件组件](https://control.ros.org/jazzy/doc/ros2_control/hardware_interface/doc/writing_new_hardware_component.html)
- [设计文档](../plans/2026-07-28-lekiwi-driver-container-design.md)
