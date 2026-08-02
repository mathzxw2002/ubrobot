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

## 2026-08-02 决策：实体急停方案（M7）

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

## 参考

- [设计文档](../plans/2026-07-28-lekiwi-driver-container-design.md)
