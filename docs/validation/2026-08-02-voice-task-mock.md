# M3 语音与任务期交互 Mock 验证

日期：2026-08-02

## 范围

本次验证仅覆盖 Operator Console 上层软件链路。Cortex、导航、抓取和语音 Provider 均使用 Mock；未连接或初始化树莓派、Piper、Go2、RealSense、ROS 驱动及云端 Qwen 服务。

## 已验证场景

以“导航到前面的椅子”为主任务，验证了：

- UI/API 请求通过 `InteractionRuntime -> TaskRuntime -> Cortex Mock` 完成规划、运行、固定序列反馈和成功状态。
- 任务执行期间的语音状态查询直接读取 TaskRuntime，不创建第二个 Cortex 请求。
- 普通语音取消只控制当前任务，不创建动作任务。
- 语音“紧急叫停机器人”及 UI 急停走独立安全通道，不进入普通任务队列。
- 急停事件带有 `critical` 优先级和 `bypass_queue=true`，当时等待中的任务被标记为 `superseded`。
- Mock 模式 UI 始终显示 `MOCK / NO HARDWARE AUTHORITY`。
- 浏览器断开后，Operator 事件游标重连可以恢复状态；Mock 语音 WebSocket 可以断开并重新建立会话。

## 自动化结果

一键命令：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/validate_operator_console.ps1
```

结果：

```text
软件单元/集成测试：Ran 136 tests — OK
独立进程 E2E：Ran 3 tests — OK
总结果：PASS
```

进程级 E2E 使用动态 localhost 端口启动真实 Uvicorn 进程，通过结构化 HTTP API 和 WebSocket 验证，结束后使用受保护的关闭接口退出。运行产生的详细、无凭据报告位于 `logs/validation/operator-console-m3-20260802-142739.md`。

## 安全边界

云端语音识别属于便利交互通道，不能作为硬件安全保障。进入真机测试前仍必须实现并验收：

- 常驻本地急停关键词检测，不依赖网络或云端 ASR。
- 物理急停装置及其独立控制链路。
- 真机停止延迟、失联行为和导航控制权租约测试。

本记录不能作为任何真实机器人运动、抓取或传感器能力已经通过的证据。
