# ADR-0002：LeKiwi 安全边界和里程计所有权

## 状态

Proposed，2026-07-28。

## 背景

驱动容器能处理 ROS 命令超时和正常进程退出，但无法在 `SIGKILL`、内核崩溃、
整机掉电等情况下保证执行软件停车。与此同时，RTAB-Map 和全向轮控制器都能
发布里程计及 TF，若不指定唯一所有者会产生冲突。

## 拟议决策

1. 软件层在 250 ms 命令超时后归零，并在所有可处理生命周期转换中停车。
2. 正式地面运行前增加独立于容器和树莓派进程的硬件急停或电机看门狗。
3. 第一阶段轮式里程计发布为 `/odom/wheel`，不发布主 `odom -> base_link` TF。
4. 后续由 `robot_localization` 融合轮式和视觉里程计，并唯一拥有主 `/odom` 与
   `odom -> base_link`。

## 结果

### 正面

- 软件崩溃和硬件级故障分别有明确防线；
- 避免多个节点同时发布相同 TF；
- 轮式与视觉信息可独立诊断并逐步融合。

### 负面

- 需要新增实体急停/看门狗硬件；
- 需要维护 EKF 参数和传感器协方差；
- 增加验收和故障注入测试工作。

## 待确认

- 轮式与视觉里程计的最终融合参数。

## 2026-08-02 决策：实体急停方案（M7，已被 2026-08-03 修订）

已由 owner 确认，解除"电机看门狗 / 外置 MCU / 电源继电器"三选一悬念：

1. **电气最终防线**：NC（常闭）蘑菇头急停按钮，主触点通过接触器直接切断
   LeKiwi 电机电源，完全不依赖树莓派、容器或 ROS 软件栈（覆盖 SIGKILL、
   内核崩溃、整机掉电——补上本 ADR 指出的软件缺口）。
2. **软件输入**：急停按钮辅助触点接 3.3 V 与 GPIO 线（libgpiod，内部
   PULL_DOWN）。常态闭合=高=安全；按下**或断线**=低=停止（fail-closed）。
3. **软件链路**：`robot_edge.hardware.local_stop.LocalStopButton`
   （去抖、fail-closed）→ `SafetySupervisor.on_local_stop()`（latched，
   必须显式授权复位）→ stop fan-out（零 `/cmd_vel` ×3 + 驱动容器 SIGINT
   停 torque）→ `safety.emergency_stop` 事件。
4. **恢复流程**：按下后 latch 不自动解除；必须操作员在 Operator Console
   显式授权 `/v1/safety/reset`，且接触器重新闭合、按钮确认复位后才能恢复。
5. **可选增强（未实现）**：树莓派心跳保持继电器方案（Pi 死机 → 心跳停 →
   电机断电），作为后续可选看门狗，不阻塞 M7。

分段延迟测量见 `scripts/hardware/measure_stop_latency.py`（M7 Task 12）。

## 2026-08-03 修订：无实体急停按钮，最终防线改为人工拔电源

owner 确认：**当前没有实体急停按钮，不接线、不实现、不验证**（本 ADR
2026-08-02 方案第 1–2 条废弃）。

1. **最终断电手段（人工）**：操作员**直接拔电机电源线**（或按驱动电源
   开关）作为最终切断层。该动作不依赖树莓派、容器或 ROS 软件栈，等效
   于原接触器方案，但由**人工**完成，不是自动触发。
2. **软件急停（保留，主用）**：Operator Console UI / `/v1/safety/stop`
   的 latched 紧急停止仍然有效，绕过规划、队列和 lease。
3. **LeKiwi 本地停止原语（M7 Task 12 第 2 条）**：依赖 ros2_control
   的 deactivate（`docker stop` SIGINT 路径）+ `/cmd_vel` 零速守卫；
   在扭矩使能前的预检阶段以**扭矩禁用**验证。
4. **约束变更**：M7 计划中"物理 E-stop 人工验证"门禁被 owner 明确豁免，
   代之以：操作员随时可拔电源、双人在场、轮子抬起/受控区域、软件急停
   各路径先单独通过。任何一次运动试验前，操作员确认电源线可及。
5. 原计划中的 GPIO/libgpiod 接线、`measure_stop_latency.py` 的
   `--execute` 模式、看门狗继电器，全部**不做现场验证**（代码保留）。

**安全模型结论**：软件急停（latched，显式复位）是主要停止机制；人工拔
电源线是最终切断层；两者都不依赖云端。M7 Task 13 导航验证按此模型执行。

## 2026-08-03 实测：里程计话题名与设计预期不同

抬起车轮预检（扭矩禁用、`hardware_mode:=real`）实测发现：

- 驱动实际发布 **`/lekiwi_base_controller/odom`**（ros2_control 控制器
  命名空间），**不是**本 ADR 早先设计的 `/odom/wheel`。
- `/joint_states` 与设计一致；关节名为 `base_back_wheel_joint`、
  `base_left_wheel_joint`、`base_right_wheel_joint`（注意 `dynamic_joint_states`
  的顺序为 back/right/left，与 `/joint_states` 的 back/left/right 不同，
  消费方必须按 name 配对而非按顺序）。
- odom 消息包含 `pose.position`（x/y）、`pose.orientation`（四元数，
  可转 yaw）、`twist.linear.x`（vx）。

已据此修正 Robot Edge 只读适配器：`mobile_base_health.py` 与
`ros/telemetry.py` 的 lekiwi 里程计话题改为
`/lekiwi_base_controller/odom`（保留 `/odom/wheel`、`/odom` 为兼容），
并提取 yaw。`ros/actions.py` 的 FOLLOW 能力检查同步更新。

## 参考

- [设计文档](../plans/2026-07-28-lekiwi-driver-container-design.md)
