# ADR-0004: Operator Console uses TaskRuntime and TelemetryHub boundaries

- Status: Accepted
- Date: 2026-08-01

## Context

The original Gradio `ChatPipeline` owns UI callbacks, ASR/TTS, Cortex calls,
cancellation, media queues, and robot observations. That coupling makes it
difficult to inspect task state, accept a status/cancel utterance while a
motion task is running, or move the UI away from the robot computer later.

The current development environment has no connected Raspberry Pi, Piper,
Go2, or RealSense devices. The software design must therefore be testable with
the in-process Cortex mock and must not infer hardware authorization.

## Decision

Use a modular monolith for the first deployment:

```text
Gradio Operator Console
  -> InteractionRuntime -> TaskRuntime -> Cortex -> Semantic Capabilities
  <- Task events         <- TelemetryHub <- sensor/capability adapters
```

`TaskRuntime` owns the lifecycle of semantic tasks. It allows one active task
with physical side effects, retains a pending queue, stores parent/root task
links, and records an append-only in-memory event timeline. It never imports
Gradio, ROS, or hardware SDKs.

`InteractionRuntime` owns text/voice turns. Status queries and cancel commands
operate on `TaskRuntime` and do not create a second Cortex request. Other text
is submitted as a task. ASR is only an input adapter: its transcript follows
the same route as typed text.

`TelemetryHub` owns timestamped channel samples and bounded history. Missing
and stale samples are explicit. UI rendering reads snapshots and never reads
hardware SDKs directly. The initial adapter publishes camera/depth
availability from the existing observation hook; odometry, joints, lease, and
capability health retain stable channel names for later ROS adapters.

The legacy ASR/TTS/video queues remain an output compatibility layer in
`ChatPipeline`; they are not task authority or telemetry storage.

## Execution and safety invariants

1. At most one task may call the Cortex backend at a time.
2. A conflicting new task is retained as `queued`; it is not automatically
   dispatched or allowed to preempt the active task.
3. Status queries remain available while a task is active and are answered
   from runtime state.
4. Cancel is bounded by the Cortex backend contract and updates the task
   timeline. Emergency-stop hardware wiring remains a separate future safety
   channel and must not depend on Cortex or Gradio.
5. Observation polling and telemetry publication do not acquire task motion
   authority.
6. No mock result authorizes real navigation, arm motion, torque, or device
   access.

## State model

```text
queued -> planning -> running -> succeeded
                     |       \-> failed
                     \-> cancelling -> cancelled

queued/running -> superseded   (reserved for target correction)
```

Every task carries `task_id`, `parent_task_id`, `root_task_id`,
`sequence_no`, `dependencies`, and `priority`. The first implementation uses
only strict single-active execution and ordered metadata; it does not execute
a general dependency graph or automatically drain the queue.

## Evolution from local to distributed deployment

The Python objects expose dictionary snapshots and callback-oriented ports so
the process boundary can change without changing task semantics:

| Local module boundary | Distributed replacement |
|---|---|
| method call into TaskRuntime | HTTP command endpoint or ROS 2 service |
| TaskEvent deque | WebSocket/SSE event stream plus persistent event store |
| TelemetryHub publish/latest | ROS 2 subscriptions and WebSocket telemetry |
| in-process Cortex backend | ROS 2 Action gateway on the robot computer |

In the distributed form, the Raspberry Pi owns TaskRuntime, TelemetryHub, ROS
adapters, and hardware-adjacent capability servers. Gradio may run on a PC or
tablet-facing server and receives only serialized snapshots/events.

## Failure handling

- Cortex unavailable or execution exception: mark the task `failed`, retain
  the error type/message, and release the active slot in `finally`.
- Cancel race: retain `cancel_requested`; even a late successful backend
  return is represented as cancelled.
- Sensor disconnect: keep the last sample but mark it stale; never substitute
  fabricated live data.
- UI disconnect: task execution remains owned by TaskRuntime. Persistent
  recovery is deferred; the current in-memory implementation loses history on
  process restart.
- Queue accumulation: queued work requires an explicit future approve/drop
  operation; there is no automatic motion after an operator leaves.

## Consequences

The current system gains testable task state, task-time query/cancel routing,
operator timeline rendering, and transport-neutral telemetry. It also adds an
explicit runtime layer that must be persisted and authenticated before remote
operation is enabled. Pause/resume, target correction, queue approval,
multi-user authority, durable replay, and direct emergency-stop integration
are intentionally deferred.

