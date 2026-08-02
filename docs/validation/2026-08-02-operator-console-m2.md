# Operator Console M2 验证记录

日期：2026-08-02

## 验证范围

本轮验证覆盖 M2 的上层软件链路：统一事件流、Operator Console 实时更新、TaskRuntime/Cortex 任务关联、Qwen Realtime 半双工语音状态机、浏览器音频播放确认和断线重连。

由于当前电脑与树莓派、Piper 机械臂、宇树 Go2、RealSense 均处于断开状态，本轮不包含硬件真机、真实导航或真实抓取测试。导航与能力执行继续使用 Mock 后端。

## 自动化测试

执行命令：

```powershell
$env:UBROBOT_CHAT_LOG_LEVEL='WARNING'
python -m unittest discover -s tests/cortex_navigation -p "test_*.py" -q
```

结果：

```text
Ran 131 tests in 5.115s
OK
```

覆盖内容包括：

- 事件编号、历史窗口、游标重放、慢消费者丢弃策略。
- Operator WebSocket 的快照、重放和实时事件。
- Voice → Interaction → Task 的 correlation ID 贯通。
- Qwen Realtime 协议夹具中的 VAD、实时转写、工具调用、音频、错误和断线事件。
- 半双工输入门控、旧会话事件隔离和实际播放完成确认。
- Voice WebSocket 的 PCM、`provider.speech_done`、`playback.done` 闭环。
- Operator Console 组件 ID 与浏览器端实时更新契约。

## 静态检查

- `python -m compileall -q src\chat_ui tests\cortex_navigation`：通过。
- `voice_client.js` Node.js 语法检查：通过。

## 独立进程 Mock 验证

在端口 `17864` 启动受管 Operator Console 进程，配置如下：

- backend：`cortex-mock`
- media：关闭
- voice：`mock`

进程就绪信息：

```text
Operator Console listener: PID 41632, port 17864 (managed).
Health: ready; backend=cortex-mock; voice=mock; mode=mock.
```

测试客户端同时连接 `/api/operator/events?after=0` 和 `/api/voice/stream`，发送测试 PCM 及麦克风电平消息。实际收到的实时事件包括：

```text
telemetry.updated
voice.state
voice.microphone_level
```

验证结束后服务正常关闭，端口释放；日志显示 VoiceRuntime 和 Uvicorn 均完成清理。

## 未执行项与限制

- 当前环境未配置 `DASHSCOPE_API_KEY` 和 `DASHSCOPE_WORKSPACE_ID`，因此未进行 Qwen-Omni-Realtime 云端实连。当前结论基于协议夹具、Mock provider 和进程级 WebSocket 验证。
- 未执行树莓派、Piper、Go2、RealSense 真机验证。
- Gradio 5.50 在测试中仍会输出 `head`、`show_api` 相关未来弃用警告及少量测试进程资源警告，不影响本轮功能测试通过。

## 结论

M2 在无硬件、无云端凭据的边界内通过：Operator Console 已从定时轮询为主转为统一事件流实时驱动，语音部分具备半双工状态控制、实时转写事件、播放完成确认和有界自动重连。云端 Qwen 和硬件能力仍需在对应环境恢复后单独验收。
