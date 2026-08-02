# Operator Console M4 软件验证

日期：2026-08-02

## 验证边界

M4 仅验证硬件就绪的软件合同。当前电脑没有连接树莓派、Piper、Go2 或 RealSense；测试未导入或初始化其 SDK，也未连接 ROS、Robot Edge 或 Qwen 云服务。

## 实现验证

- Capability Registry 包含 navigation、grasp、observation、follow 和 stop。
- 每项能力具有 availability、health、execution mode、required resources 和 hardware authority。
- Mock/Fixture 模式不能声明硬件权限。
- Camera、Depth、Odometry、Joint States、Navigation Lease 和 Capability Health 使用 JSON 安全 DTO。
- 未连接通道为 `disconnected`，明确不可用通道保持 `unavailable`，超时样本变为 `stale`。
- TelemetryHub 拒绝 bytes、相机 handle 和任意 SDK/runtime 对象。
- Fixture Telemetry/Cortex Adapter 不包含 ROS、RealSense、Piper 或 Go2 SDK import。
- Operator Snapshot、Health 和 `/api/operator/capabilities` 只返回序列化状态。

## 自动化结果

执行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/validate_operator_console.ps1
```

结果：

```text
软件单元/集成测试：Ran 150 tests — OK
独立进程 Mock E2E：Ran 3 tests — OK
总结果：PASS
Hardware authority: false
```

详细的无凭据运行报告：`logs/validation/operator-console-software-20260802-144856.md`。

## 后续真机前置条件

M1–M4 通过仅说明 Operator Console 软件基线和适配边界可用。恢复硬件后仍需依次完成：Robot Edge 认证与心跳、只读传感器验证、导航 lease、物理/本地急停、低风险 Go2 导航、Piper 低速无负载动作，以及最终联合任务验证。

本记录不能作为任何硬件能力已经通过测试的证据。
