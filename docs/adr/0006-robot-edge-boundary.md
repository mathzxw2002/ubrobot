# ADR-0006: Robot Edge 与 Operator Console 边界

- 状态：Accepted
- 日期：2026-08-02

## 背景

Operator Console 可能运行在开发电脑、平板或其他终端，而 Go2、Piper、RealSense 和 ROS 2 驱动运行在机器人侧。UI 进程不能依赖硬件 SDK 对象，也不能因浏览器刷新、网络断开或进程迁移而改变 TaskRuntime 的任务语义。

当前阶段所有硬件均断开，因此只实现数据合同和 Fixture Adapter，不实现 Robot Edge 服务或 ROS 客户端。

## 决策

系统保持以下责任边界：

```text
Browser / Gradio
       │ serialized HTTP + WebSocket only
       ▼
Operator Console Runtime
  InteractionRuntime / TaskRuntime / TelemetryHub
       │ authenticated Robot Edge contract (future)
       ▼
Robot Edge
  lease enforcement / safety gate / ROS Action adapters
       │
       ▼
Go2 / Piper / RealSense / physical E-stop
```

Operator Console 只保存并传输：

- Capability Descriptor。
- Cortex Command、Feedback 和 Result DTO。
- Camera、Depth、Odometry、Joint States、Navigation Lease 和 Capability Health DTO。
- Task、Interaction、Voice 与 Telemetry 事件。

禁止通过 Operator Snapshot、Health API 或 Event Stream 返回 ROS node、Action client、相机 frame 对象、SDK manager、文件描述符或可执行 callback。

## Capability 合同

固定能力名称为：

- `navigation`
- `grasp`
- `observation`
- `follow`
- `stop`

每项必须声明 `availability`、`health`、`execution_mode`、`required_resources`、`hardware_authority` 和更新时间。Mock/Fixture 模式禁止声明硬件权限。

状态不能根据缺失数据推断为正常。未建立 Robot Edge 连接时使用 `disconnected`；设备明确拒绝或不存在时使用 `unavailable`；超过通道时限时使用 `stale`。

## 未来 API 与 ROS Action 映射

| Capability | Robot Edge 合同 | ROS/驱动映射 |
|---|---|---|
| navigation | 提交目标对象、任务 ID、correlation ID、lease ID | `/ubrobot/navigation/navigate_to_object` Action |
| grasp | 提交目标对象、抓取约束、任务 ID | `/ubrobot/manipulation/grasp_object` Action |
| observation | 订阅序列化 RGB/Depth 元数据及受控预览流 | RealSense topics/driver，由 Edge 转码 |
| follow | 提交跟随目标、距离约束、lease ID | 独立 Follow Action 或导航编排 |
| stop | 无需规划的高优先级停止请求 | Edge safety gate，再映射各驱动停止原语 |

具体 URL 和消息 schema 在 Robot Edge 实现前冻结为版本化 OpenAPI/ROS interface 包；Operator Console 通过 adapter 使用合同，不直接导入 ROS interface。

## 认证与授权

未来 Robot Edge 必须满足：

- TLS，部署环境优先使用双向 TLS 标识 Console 与 Edge。
- 短期访问令牌，按 `observe`、`task.submit`、`task.cancel`、`safety.stop`、`lease.manage` 分 scope。
- 密钥只保存在服务端，不进入浏览器快照、事件、日志或回放。
- 所有控制请求携带 correlation ID、操作者身份、时间戳和防重放信息。
- `safety.stop` 独立授权且优先于普通任务和导航租约。

## Navigation Lease

会产生移动的 navigation/follow 请求必须携带有效 lease：

- 同一机器人同一时刻最多一个运动控制 owner。
- lease 具有 ID、owner、签发时间、过期时间和续租心跳。
- Edge 是最终裁决者；UI 显示 lease 但不能自行授予硬件权限。
- lease 过期、连接失联或 owner 改变时，Edge 进入已定义的安全停止状态。
- grasp 是否需要独立 manipulation lease，在 Piper 真机设计阶段决定。

## 心跳与失联

- Telemetry 每个通道独立计算陈旧状态。
- Console 与 Edge 连接中断时，UI 显示 `disconnected`，不得复用最后一帧并标记为实时。
- 浏览器断开不直接取消活动任务；任务归 Robot Edge/TaskRuntime 生命周期管理。
- Edge 与硬件驱动断开时，由 Edge 执行本地安全策略，不能等待云端或浏览器决定。

## 急停责任

云端 ASR、普通 UI 按钮和网络 API 都不是唯一硬件安全保障。真机启用前必须具备：

- 物理急停回路。
- Robot Edge 本地停止原语。
- 不依赖云端的本地急停关键词检测（如启用语音安全入口）。
- 已测量的停止延迟和断网行为。

Operator Console 的 `critical` 急停事件用于编排、审计和尽快请求停止，不能替代上述本地保障。

## 当前实现

- `capability_registry.py` 保存纯数据描述。
- `adapters/telemetry.py` 定义 JSON 安全 DTO 和 Fixture Adapter。
- `adapters/cortex.py` 定义 Cortex DTO 和 Fixture TaskBackend。
- `TelemetryHub` 拒绝 bytes 和任意 SDK 对象。
- `/api/operator/capabilities`、Operator Snapshot 和 Health 只返回序列化数据。

当前实现不连接任何硬件或 ROS 服务，也不证明真机能力可用。

## 后果

后续可增加 Robot Edge HTTP/gRPC/ROS adapter，而无需修改 Gradio、InteractionRuntime、TaskRuntime、Interaction 分类或语音 Provider。代价是所有新遥测与能力状态必须先定义 DTO 和显式失联语义，不能把 SDK 对象快速透传到 UI。
