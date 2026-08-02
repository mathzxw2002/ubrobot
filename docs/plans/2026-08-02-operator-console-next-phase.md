# Operator Console Next Phase Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Turn the current mock-capable Gradio/Cortex/Qwen prototype into a stable, observable, repeatably testable operator-console baseline while keeping all real hardware execution disabled.

**Architecture:** Keep the current modular monolith: Gradio talks only to transport-neutral interaction, task, voice, and telemetry runtimes; TaskRuntime remains the sole owner of Cortex task execution. Add explicit HTTP/WebSocket contracts and persistence seams now so the UI can later move off the Raspberry Pi without changing task semantics, but do not split processes or add infrastructure prematurely.

**Tech Stack:** Python, FastAPI, Gradio, WebSocket PCM audio, Qwen-Omni-Realtime, EMOS Cortex adapter, `unittest`, ROS 2 interface contracts, in-process mock backends.

---

## Current baseline

- Text and voice requests share `InteractionRuntime -> TaskRuntime -> Cortex`.
- One physical-side-effect task may be active; additional tasks are retained as queued metadata.
- Status and cancel utterances do not create a second Cortex task.
- Qwen realtime uses a provider-neutral half-duplex state machine.
- Gradio displays task state, timeline, telemetry placeholders, voice state, sensor previews, and emergency stop.
- Navigation and grasp have mock/contract tests; Raspberry Pi, Piper, Go2, and RealSense are disconnected and remain outside this phase.
- On 2026-08-02, 22 focused runtime/voice/UI tests passed with `python -m unittest`; `pytest` is not installed in the active Python environment.

## Non-functional targets for this phase

- A duplicate start must fail with a clear operator-facing message identifying the occupied port/process.
- UI command acknowledgement and state changes should appear within 500 ms locally; partial ASR text should appear within 1 s of a provider event.
- Every interaction, task transition, voice transition, backend request, and error must carry a correlation ID in structured logs.
- Refreshing or reconnecting the browser must not cancel an active task.
- Cloud credentials remain server-side and must never be returned by status endpoints or written to logs.
- Mock mode must never initialize ROS/hardware SDKs or grant motion authority.

### Task 1: Freeze a reproducible software baseline

**Files:**
- Modify: `requirements.in`
- Modify: `requirements.txt`
- Create: `requirements-dev.txt`
- Modify: `src/chat_ui/README.md`
- Create: `tests/cortex_navigation/test_dependency_contract.py`

**Steps:**

1. Add a dev-test dependency set containing `pytest` while preserving `unittest` compatibility.
2. Pin the currently supported FastAPI, Gradio, Starlette, websockets, and Qwen client-facing dependencies.
3. Add a dependency contract test that imports the UI, creates the FastAPI app in mock mode, and checks the versions against supported ranges.
4. Remove current Gradio deprecation warnings (`Blocks(js=...)` and private event API usage) or pin the last version that supports the chosen integration until migration is complete.
5. Run `python -m unittest discover -s tests/cortex_navigation -p "test_*.py" -v`; expect all software-only tests to pass.
6. Record the validated Windows startup command and environment variables in the README.

**Acceptance:** A fresh local environment has one documented install command and one test command, with no ambiguous package combination.

### Task 2: Make process startup, shutdown, and health diagnosable

**Files:**
- Modify: `src/chat_ui/app.py`
- Create: `src/chat_ui/service_lifecycle.py`
- Create: `scripts/operator_console.ps1`
- Create: `tests/cortex_navigation/test_service_lifecycle.py`
- Modify: `src/chat_ui/README.md`

**Steps:**

1. Write failing tests for free-port startup, occupied-port diagnosis, readiness, and graceful shutdown.
2. Add `/api/health/live` and `/api/health/ready`; readiness reports backend, voice provider, mock/hardware mode, and sanitized capability health.
3. Add a PowerShell launcher with `start`, `status`, `logs`, and `stop`; retain PID/log files under `logs/` and reject duplicate starts cleanly.
4. Register FastAPI lifespan cleanup for voice sessions, worker threads, and backend cancellation hooks.
5. Emit structured startup/shutdown logs and include the selected URL and log paths.
6. Verify that stopping the launcher closes port 7863 and stops browser refresh requests.

**Acceptance:** Repeated start/stop cycles do not leave a listener behind, and a second start explains the existing process instead of appearing to hang.

### Task 3: Replace polling-only state visibility with an event contract

**Files:**
- Create: `src/chat_ui/event_stream.py`
- Modify: `src/chat_ui/task_runtime.py`
- Modify: `src/chat_ui/interaction_runtime.py`
- Modify: `src/chat_ui/voice_runtime.py`
- Modify: `src/chat_ui/telemetry.py`
- Modify: `src/chat_ui/app.py`
- Modify: `src/chat_ui/voice_client.js`
- Create: `tests/cortex_navigation/test_operator_event_stream.py`
- Create: `docs/adr/0005-operator-event-stream.md`

**Steps:**

1. Define a serialized event envelope with `event_id`, `timestamp`, `kind`, `source`, `correlation_id`, `task_id`, and `payload`.
2. Publish task, interaction, voice, and telemetry changes through one bounded in-memory event stream.
3. Expose browser updates over WebSocket or SSE; retain snapshot endpoints for initial load and reconnection.
4. Update Gradio-side JavaScript so partial transcript, voice state, task state, and timeline update from events rather than waiting for the timer.
5. Keep the timer only as a slow recovery/health fallback.
6. Test ordered delivery, bounded history, reconnect from the last event ID, and slow-client behavior.

**Acceptance:** Qwen partial transcript and task transitions are visibly incremental, and a browser refresh reconstructs the current state without affecting execution.

### Task 4: Complete the Qwen half-duplex interaction loop

**Files:**
- Modify: `src/chat_ui/qwen_realtime.py`
- Modify: `src/chat_ui/voice_runtime.py`
- Modify: `src/chat_ui/voice_client.js`
- Modify: `src/chat_ui/app.py`
- Modify: `tests/cortex_navigation/test_voice_runtime.py`
- Create: `tests/cortex_navigation/test_qwen_realtime_fixture.py`

**Steps:**

1. Capture representative Qwen server events as sanitized fixtures: connected, partial transcript, final transcript, tool call, audio delta, speech done, error, and disconnect.
2. Test and implement partial/final transcript propagation independently of tool-call completion.
3. Add microphone-level/VAD indicators, explicit listening/thinking/speaking states, and browser audio playback completion acknowledgement.
4. Enforce half-duplex input rejection while thinking/speaking and automatically resume listening only after playback finishes.
5. Add bounded reconnect with backoff, session timeout, stale-event rejection, and a visible retry control.
6. Preserve a provider-neutral protocol so a future Volcengine ASR+TTS adapter maps into the same events.

**Acceptance:** A live session shows incremental transcription, executes one Cortex request, plays the reply, and returns to listening without requiring upload/click-to-record interaction.

### Task 5: Validate safety and task-time voice interaction

**Files:**
- Modify: `src/chat_ui/interaction_runtime.py`
- Modify: `src/chat_ui/task_runtime.py`
- Modify: `src/chat_ui/app.py`
- Create: `tests/cortex_navigation/test_voice_task_scenarios.py`
- Create: `docs/validation/2026-08-xx-voice-task-mock.md`

**Steps:**

1. Add deterministic scenarios for navigation plus concurrent status query, normal cancel, spoken emergency stop, and UI emergency stop.
2. Verify status queries read runtime state and never dispatch another Cortex command.
3. Verify emergency stop bypasses normal queued work and produces a high-priority safety event.
4. Keep spoken cloud recognition classified as a convenience path, not the hardware safety guarantee.
5. Add a prominent UI banner stating `MOCK / NO HARDWARE AUTHORITY` whenever mock mode is active.
6. Document the future requirement for an always-on local keyword detector and physical E-stop before hardware testing.

**Acceptance:** “导航到前面的椅子” can be interrupted by status/cancel/emergency interactions while the navigation action itself remains mocked.

### Task 6: Add repeatable end-to-end operator-console acceptance tests

**Files:**
- Create: `tests/e2e/test_operator_console_mock.py`
- Create: `tests/fixtures/operator_scenarios.json`
- Create: `scripts/validate_operator_console.ps1`
- Modify: `src/chat_ui/mock_backend.py`
- Modify: `src/chat_ui/README.md`

**Steps:**

1. Add a deterministic mock clock/feedback sequence for navigation and grasp.
2. Start the console on a dynamically allocated port from the test process.
3. Submit “导航到前面的椅子” through the same HTTP/UI callback path used by Gradio.
4. Assert the full sequence: interaction accepted, task planning, running, mock feedback, succeeded, timeline retained.
5. Run status, queue, cancel, emergency-stop, reconnect, and provider-disconnect scenarios.
6. Generate a timestamped validation report containing commands, environment, assertions, and sanitized logs.

**Acceptance:** One command validates UI/API, InteractionRuntime, TaskRuntime, Cortex mock, voice fixtures, telemetry state, and replay without ROS or hardware.

### Task 7: Prepare capability and telemetry adapters without touching hardware

**Files:**
- Create: `src/chat_ui/capability_registry.py`
- Create: `src/chat_ui/adapters/telemetry.py`
- Create: `src/chat_ui/adapters/cortex.py`
- Modify: `src/chat_ui/telemetry.py`
- Create: `tests/cortex_navigation/test_capability_registry.py`
- Create: `tests/cortex_navigation/test_telemetry_adapters.py`
- Create: `docs/adr/0006-robot-edge-boundary.md`

**Steps:**

1. Define capability descriptors for navigation, grasp, observation, follow, and stop, including availability, health, execution mode, and required resources.
2. Define transport-neutral telemetry DTOs for camera, depth, odometry, joints, navigation lease, and capability health.
3. Implement fixture adapters only; do not import or connect to RealSense, Piper, Go2, or ROS drivers in workstation tests.
4. Define the future robot-edge API/ROS Action mapping and authentication/lease requirements in ADR-0006.
5. Ensure the remote UI receives only serialized state and cannot access hardware SDK objects.
6. Add contract tests proving unavailable/stale/disconnected states are explicit and never replaced with fabricated live data.

**Acceptance:** Hardware-facing implementation can later be added behind adapters without changing Gradio, TaskRuntime, interaction classification, or voice providers.

## Deferred until hardware is reconnected

- Real Go2/LeKiwi navigation movement and odometry validation.
- Piper torque enable, trajectory execution, grasp force, and collision tests.
- RealSense RGB-D synchronization, calibration, and latency measurement.
- Physical and local-software emergency-stop latency validation.
- Navigation lease arbitration and multi-device network-loss behavior.

Mock results must not be promoted as evidence for any of these items.

## Recommended delivery order

1. Milestone M1 — Tasks 1-2: stable baseline and lifecycle.
2. Milestone M2 — Tasks 3-4: genuinely realtime, observable half-duplex voice.
3. Milestone M3 — Tasks 5-6: complete software-level functional acceptance.
4. Milestone M4 — Task 7: hardware-ready boundaries, still fixture-only.

Each milestone should end with a clean full software test run, a short validation record, and a separate reviewable commit. Do not start hardware integration until M1-M4 pass and the physical safety checklist is approved.
