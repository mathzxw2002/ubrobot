# ADR-0005: Operator state uses one bounded event stream

- Status: Accepted
- Date: 2026-08-02

## Context

The M1 console refreshed task, telemetry, and voice state with a one-second
Gradio timer. That was adequate for mock validation but delayed partial speech
transcription, duplicated state transport, and coupled remote UI evolution to
Gradio callback behavior.

## Decision

TaskRuntime, InteractionRuntime, VoiceSessionManager, and TelemetryHub publish
transport-neutral `EventEnvelope` records through one in-process EventStream.
Every envelope carries a monotonic event ID, UTC timestamp, kind, source,
correlation ID, optional task ID, and JSON payload.

FastAPI exposes:

- `GET /api/operator/snapshot` for initial state and recovery;
- `WS /api/operator/events?after=<event_id>` for ordered replay and live events.

History and each subscriber mailbox are bounded. A slow subscriber drops its
oldest pending records and receives a gap response plus a fresh snapshot. The
publisher never waits for a browser. Gradio's timer remains a slow fallback
for sensor image refresh and temporary browser-event failures.

## Consequences

The runtime semantics no longer depend on Gradio and can later be transported
through SSE, another WebSocket server, or a persistent broker. In-memory
history is intentionally lost on process restart; durable replay remains a
future concern. Event payloads must remain serialized DTOs and must not carry
hardware SDK objects or cloud credentials.
