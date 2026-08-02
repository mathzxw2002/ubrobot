# Robot Edge and Safe Hardware Bring-up Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an authenticated, lease-controlled Robot Edge between the Operator Console and ROS/hardware, validate it entirely with fixtures first, then bring RealSense, one mobile-base profile, and Piper online through explicit read-only and guarded-motion gates.

**Architecture:** Extract the M4 transport DTOs into a lightweight shared contract package. Run a standalone Robot Edge service on the robot-side computer; the Operator Console uses a `TaskBackend`/telemetry client adapter and never imports ROS or hardware SDKs. Robot Edge owns authentication, replay protection, navigation lease enforcement, local safety supervision, ROS Action adapters, hardware health, and fail-closed disconnect behavior.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic v2, httpx, WebSocket, existing `TaskRuntime`/`TelemetryHub`, ROS 2 Actions (`NavigateToObject`, `GraspObject`), Docker Compose, `unittest`, fixture backends. Hardware SDKs remain robot-side optional dependencies only.

---

## 0. Handoff state — read before changing code

As of 2026-08-02:

- Repository: `C:\Users\china\ubrobot`
- Branch: `main`
- HEAD before M1–M4 work: `3281c80`
- M1–M4 changes are **not committed**. The worktree contains both modified and untracked implementation files.
- Validated software result:
  - `150` tests under `tests/cortex_navigation`: PASS
  - `3` process-level tests under `tests/e2e`: PASS
  - `hardware_authority=false`
- Raspberry Pi, Piper, Go2, RealSense, and local ROS/hardware environments are disconnected.
- Qwen cloud credentials are not present in the validated environment.
- Known non-blocking warnings: Gradio 5.50 deprecations and test-only event-loop resource warnings.

Read these documents before implementation:

- `docs/adr/0004-operator-console-runtime-boundaries.md`
- `docs/adr/0005-operator-event-stream.md`
- `docs/adr/0006-robot-edge-boundary.md`
- `docs/validation/2026-08-02-operator-console-m4.md`
- `src/chat_ui/README.md`

Current core files:

- `src/chat_ui/task_runtime.py`
- `src/chat_ui/interaction_runtime.py`
- `src/chat_ui/event_stream.py`
- `src/chat_ui/telemetry.py`
- `src/chat_ui/capability_registry.py`
- `src/chat_ui/adapters/telemetry.py`
- `src/chat_ui/adapters/cortex.py`
- `src/chat_ui/pipeline.py`
- `src/chat_ui/app.py`

Existing ROS contracts and servers:

- `ros_depends_ws/src/ubrobot_interfaces/action/NavigateToObject.action`
- `ros_depends_ws/src/ubrobot_interfaces/action/GraspObject.action`
- `ros_depends_ws/src/ubrobot_navigation/ubrobot_navigation/navigate_to_object_server.py`
- `ros_depends_ws/src/ubrobot_navigation/ubrobot_navigation/cmd_vel_guard.py`
- `ros_depends_ws/src/ubrobot_manipulation/ubrobot_manipulation/grasp_object_server.py`
- `ros_depends_ws/src/emos_bringup/launch/cortex_navigation_bringup.launch.py`

### Non-negotiable constraints

1. Do not initialize ROS, RealSense, Piper, Go2, LeKiwi, CAN, serial, or motor SDKs in workstation tests.
2. Do not claim hardware authority from Mock/Fixture mode.
3. Do not expose SDK objects, ROS clients, binary frames, callbacks, secrets, or file descriptors through UI/health/event payloads.
4. Browser disconnect must not cancel an active task. Robot Edge disconnect or lease expiry must execute a local fail-closed policy.
5. `safety.stop` bypasses planning, queues, and normal lease requirements.
6. Cloud ASR is a convenience input, never the only emergency-stop mechanism.
7. Do not proceed from software fixtures to hardware merely because tests pass. Hardware phases require the explicit gates below.
8. Choose exactly one mobile-base profile for the first live run: `go2` or `lekiwi`. Do not silently mix their drivers or command topics.

---

## Delivery milestones

| Milestone | Scope | Hardware allowed? | Exit condition |
|---|---|---:|---|
| M5 | Robot Edge contracts, fixture service, auth, lease, safety, remote Console adapter | No | Two-process fixture E2E passes |
| M6 | Robot-side deployment plus read-only health for sensors/base/arm | Read-only only | Explicit inventory and stale/disconnect tests pass |
| M7 | Local/physical safety validation and one guarded mobile-base motion | Limited, supervised | Stop latency and low-risk navigation pass |
| M8 | Piper low-speed motion, grasp, then combined task | Limited, supervised | Separate signed validation reports pass |

**Mandatory checkpoint:** Stop and request owner review after each milestone. Never continue automatically into the next hardware authority level.

---

## Milestone M5 — fixture-only Robot Edge

### Task 1: Freeze the M1–M4 baseline before new implementation

**Files:**

- Review: all output of `git status --short`
- Verify: `docs/validation/2026-08-02-operator-console-m4.md`
- Do not modify implementation files in this task

**Step 1: Capture the current state**

Run:

```powershell
git branch --show-current
git rev-parse --short HEAD
git status --short
git diff --check
```

Expected:

- Branch `main`
- Base HEAD `3281c80` unless the owner has already created a baseline commit
- No `git diff --check` errors

**Step 2: Re-run the accepted baseline**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/validate_operator_console.ps1
```

Expected: `150` software tests and `3` E2E tests pass. If counts changed, inspect why; do not blindly update this plan.

**Step 3: Review ownership before committing**

M1–M4 are currently mixed in one dirty worktree. Show the owner `git diff --stat` and `git status --short`. If any file is not part of Operator Console M1–M4, stop and ask how to separate it.

**Step 4: Create an approved baseline commit**

Only after owner approval:

```powershell
git add .gitignore pyproject.toml requirements.in requirements.txt requirements-dev.txt requirements-operator-console.txt
git add src/chat_ui scripts tests/cortex_navigation tests/e2e tests/fixtures
git add docs/adr docs/plans docs/validation
git commit -m "feat: establish operator console M1-M4 baseline"
```

Expected: clean worktree. If the owner does not authorize a commit, create a new worktree from an owner-provided baseline SHA and stop this task until the changes are available there.

---

### Task 2: Extract versioned shared transport contracts

**Files:**

- Create: `src/ubrobot_contracts/__init__.py`
- Create: `src/ubrobot_contracts/capabilities.py`
- Create: `src/ubrobot_contracts/telemetry.py`
- Create: `src/ubrobot_contracts/edge_api.py`
- Modify: `src/chat_ui/capability_registry.py`
- Modify: `src/chat_ui/adapters/telemetry.py`
- Modify: `src/chat_ui/adapters/cortex.py`
- Test: `tests/robot_edge/test_contracts.py`

**Step 1: Write failing import and JSON-schema tests**

Test these exact properties:

```python
from ubrobot_contracts import PROTOCOL_VERSION
from ubrobot_contracts.edge_api import CommandRequest, CommandAccepted

def test_protocol_contract_is_versioned_and_json_safe():
    request = CommandRequest(
        text="导航到前面的椅子",
        correlation_id="trace-1",
        operator_id="operator-test",
        lease_id="lease-1",
    )
    assert PROTOCOL_VERSION == "1.0"
    assert request.model_dump(mode="json")["text"] == "导航到前面的椅子"
    assert CommandAccepted(command_id="cmd-1").model_dump(mode="json")
```

Also test:

- capability names are restricted to navigation/grasp/observation/follow/stop;
- Mock/Fixture cannot set `hardware_authority=true`;
- telemetry state is available/unavailable/stale/disconnected;
- all timestamps are timezone-aware;
- unknown protocol major versions are rejected.

**Step 2: Verify failure**

```powershell
python -m unittest tests.robot_edge.test_contracts -v
```

Expected: FAIL because `ubrobot_contracts` does not exist.

**Step 3: Implement minimal Pydantic contracts**

`src/ubrobot_contracts/__init__.py` must export:

```python
PROTOCOL_VERSION = "1.0"
```

`edge_api.py` must define, at minimum:

- `CommandRequest`
- `CommandAccepted`
- `CommandEvent`
- `CancelRequest`
- `EmergencyStopRequest`
- `LeaseAcquireRequest`
- `LeaseRecord`
- `Heartbeat`
- `ErrorResponse`

Every command/control request must include correlation ID, operator identity, request timestamp, nonce, and protocol version. Do not include API keys in any model.

**Step 4: Preserve compatibility**

Make current `src/chat_ui/...` contract modules import/re-export the shared models where practical. Do not change Gradio, TaskRuntime, InteractionRuntime, or VoiceSessionManager APIs.

**Step 5: Run tests**

```powershell
python -m unittest tests.robot_edge.test_contracts tests.cortex_navigation.test_capability_registry tests.cortex_navigation.test_telemetry_adapters -v
```

Expected: PASS.

**Step 6: Commit**

```powershell
git add src/ubrobot_contracts src/chat_ui/capability_registry.py src/chat_ui/adapters tests/robot_edge/test_contracts.py
git commit -m "refactor: extract versioned robot edge contracts"
```

---

### Task 3: Build a fixture-only Robot Edge service

**Files:**

- Create: `src/robot_edge/__init__.py`
- Create: `src/robot_edge/app.py`
- Create: `src/robot_edge/runtime.py`
- Create: `src/robot_edge/event_stream.py`
- Create: `src/robot_edge/fixture_backend.py`
- Create: `tests/robot_edge/test_fixture_service.py`
- Create: `requirements-robot-edge.txt`

**Step 1: Write failing FastAPI contract tests**

Create a `TestClient` fixture and assert these endpoints:

```text
GET  /v1/health/live
GET  /v1/health/ready
GET  /v1/capabilities
GET  /v1/telemetry/snapshot
WS   /v1/events?after=<event_id>
POST /v1/commands
POST /v1/commands/{command_id}/cancel
POST /v1/safety/stop
```

Initial readiness must contain:

```json
{
  "status": "ready",
  "execution_mode": "fixture",
  "hardware_authority": false
}
```

Test a deterministic navigation fixture sequence:

```text
command.accepted
command.planning
command.running
command.feedback
command.succeeded
```

**Step 2: Verify failure**

```powershell
python -m unittest tests.robot_edge.test_fixture_service -v
```

Expected: FAIL because `robot_edge.app` does not exist.

**Step 3: Implement the fixture runtime**

Requirements:

- bounded event history with monotonic integer event IDs;
- cursor replay and snapshot reconstruction;
- one active side-effect command;
- bounded pending metadata;
- deterministic feedback, no sleeps longer than 100 ms in tests;
- cancellation and emergency stop;
- all six telemetry channels present with explicit state;
- no ROS/hardware imports.

Do not import `src.chat_ui.app` into Robot Edge. Shared code may come only from `ubrobot_contracts` or a deliberately extracted neutral module.

**Step 4: Add import-boundary test**

Fail if `src/robot_edge` imports any of:

```text
rclpy
pyrealsense2
piper_sdk
unitree_sdk2py
lerobot.cameras.realsense
Go2Manager
```

**Step 5: Run tests and commit**

```powershell
python -m unittest tests.robot_edge.test_fixture_service -v
git add src/robot_edge requirements-robot-edge.txt tests/robot_edge
git commit -m "feat: add fixture robot edge service"
```

---

### Task 4: Add authentication, scopes, and replay protection

**Files:**

- Create: `src/robot_edge/auth.py`
- Modify: `src/robot_edge/app.py`
- Create: `tests/robot_edge/test_auth.py`
- Create: `deploy/robot-edge/.env.example`

**Step 1: Write failing tests**

Test:

- missing token returns 401;
- wrong scope returns 403;
- expired request timestamp returns 409;
- reused nonce returns 409;
- `observe` cannot submit/cancel/stop;
- `task.submit` cannot call safety stop;
- `safety.stop` can stop without a navigation lease;
- health/live reveals no secrets;
- errors/logs never contain the token.

Required scopes:

```text
observe
task.submit
task.cancel
safety.stop
lease.manage
```

**Step 2: Implement a development token verifier**

Use server-side configured bearer tokens mapped to scopes. Keep the verifier behind a protocol so mTLS/OIDC can replace it later. Do not invent custom encryption. Store only a hash/fingerprint in logs.

Environment names:

```text
UBROBOT_EDGE_TOKENS_FILE
UBROBOT_EDGE_REQUEST_MAX_AGE_SEC
UBROBOT_EDGE_NONCE_TTL_SEC
```

No default control token is allowed. Fixture tests inject configuration directly.

**Step 3: Run and commit**

```powershell
python -m unittest tests.robot_edge.test_auth -v
git add src/robot_edge/auth.py src/robot_edge/app.py tests/robot_edge/test_auth.py deploy/robot-edge/.env.example
git commit -m "feat: secure robot edge control endpoints"
```

---

### Task 5: Implement navigation lease and local safety supervision

**Files:**

- Create: `src/robot_edge/lease.py`
- Create: `src/robot_edge/safety.py`
- Modify: `src/robot_edge/runtime.py`
- Modify: `src/robot_edge/app.py`
- Create: `tests/robot_edge/test_lease.py`
- Create: `tests/robot_edge/test_safety_supervisor.py`

**Step 1: Write lease state-machine tests**

Test exact transitions:

```text
none -> active -> renewed -> released
none -> active -> expired
active(owner A) + acquire(owner B) -> conflict
```

Lease record fields:

```text
lease_id, owner, issued_at, expires_at, last_renewed_at, state
```

Use an injected monotonic clock; never use `sleep()` for lease unit tests.

**Step 2: Write fail-closed safety tests**

Test that each condition invokes an injected stop fan-out exactly once:

- lease expiry during a motion command;
- Robot Edge loses downstream/ROS heartbeat;
- local stop input activates;
- `/v1/safety/stop` is called;
- shutdown occurs with an active command.

Emergency stop must:

- set a latched safety state;
- cancel active execution;
- supersede pending work;
- revoke the lease;
- emit `safety.emergency_stop` with `priority=critical`;
- reject new commands until an explicit, authorized reset;
- never auto-reset because a client reconnects.

**Step 3: Implement minimal state machines**

Keep stop outputs injected. M5 uses only a fixture stop sink that records calls; no motor/ROS function is allowed.

**Step 4: Run and commit**

```powershell
python -m unittest tests.robot_edge.test_lease tests.robot_edge.test_safety_supervisor -v
git add src/robot_edge tests/robot_edge
git commit -m "feat: enforce robot edge lease and safety latch"
```

---

### Task 6: Add the Operator Console Robot Edge backend

**Files:**

- Create: `src/chat_ui/adapters/robot_edge.py`
- Modify: `src/chat_ui/pipeline.py`
- Modify: `src/chat_ui/service_lifecycle.py`
- Modify: `src/chat_ui/README.md`
- Create: `tests/cortex_navigation/test_robot_edge_backend.py`

**Step 1: Write failing adapter tests**

The adapter must implement TaskRuntime's existing backend API:

```python
def execute(self, task: str, *, on_feedback) -> str: ...
def cancel_active(self) -> bool: ...
def emergency_stop(self) -> bool: ...
def close(self) -> None: ...
```

Test:

- command submission and ordered feedback;
- cancellation;
- emergency stop;
- auth failure is sanitized;
- edge disconnect fails the task clearly;
- browser refresh does not close the adapter;
- adapter close releases subscriptions;
- no credential appears in snapshots/events/logs.

**Step 2: Add backend selection**

New environment:

```text
UBROBOT_CHAT_BACKEND=robot-edge
UBROBOT_EDGE_URL=http://127.0.0.1:8780
UBROBOT_EDGE_TOKEN_FILE=<server-side path>
UBROBOT_EDGE_OPERATOR_ID=<stable id>
```

Do not accept a control token from browser JavaScript or Gradio inputs.

**Step 3: Preserve runtimes**

Do not modify public APIs of TaskRuntime, InteractionRuntime, VoiceSessionManager, Gradio callbacks, or semantic capability names.

**Step 4: Run and commit**

```powershell
python -m unittest tests.cortex_navigation.test_robot_edge_backend tests.cortex_navigation.test_operator_runtimes -v
git add src/chat_ui tests/cortex_navigation/test_robot_edge_backend.py
git commit -m "feat: connect operator runtime to robot edge"
```

---

### Task 7: Bridge Robot Edge telemetry and capability health

**Files:**

- Create: `src/chat_ui/adapters/robot_edge_telemetry.py`
- Modify: `src/chat_ui/pipeline.py`
- Modify: `src/chat_ui/telemetry.py`
- Modify: `src/chat_ui/app.py`
- Create: `tests/cortex_navigation/test_robot_edge_telemetry.py`

**Step 1: Write failing tests**

Test:

- initial snapshot hydration;
- monotonic event cursor replay;
- reconnect after dropped WebSocket;
- stale after per-channel deadline;
- edge disconnect changes every edge-backed channel to `disconnected`;
- last value may remain for diagnosis but must not be marked available/live;
- capability hardware authority is true only when Edge reports hardware mode and local config explicitly permits it;
- an SDK-like object is rejected before reaching EventStream.

**Step 2: Implement one background telemetry client**

It may update `TelemetryHub` and `CapabilityRegistry`; it must not update Gradio components directly. Use bounded reconnect/backoff and a stop event registered with FastAPI lifespan cleanup.

**Step 3: Run and commit**

```powershell
python -m unittest tests.cortex_navigation.test_robot_edge_telemetry tests.cortex_navigation.test_operator_event_stream -v
git add src/chat_ui tests/cortex_navigation/test_robot_edge_telemetry.py
git commit -m "feat: stream robot edge telemetry to operator console"
```

---

### Task 8: Add fixture deployment and two-process acceptance

**Files:**

- Create: `deploy/robot-edge/Dockerfile`
- Create: `deploy/robot-edge/compose.fixture.yaml`
- Create: `deploy/robot-edge/compose.hardware.yaml`
- Create: `deploy/robot-edge/README.md`
- Create: `scripts/robot_edge.ps1`
- Create: `tests/e2e/test_operator_robot_edge_fixture.py`
- Modify: `scripts/validate_operator_console.ps1`
- Create: `docs/validation/YYYY-MM-DD-robot-edge-fixture.md`

**Step 1: Add a fixture-only compose profile**

Defaults:

```text
UBROBOT_EDGE_MODE=fixture
UBROBOT_EDGE_HARDWARE_AUTHORITY=false
```

`compose.hardware.yaml` must not start by default. It must require all of:

```text
UBROBOT_EDGE_MODE=hardware
UBROBOT_EDGE_HARDWARE_AUTHORITY=true
UBROBOT_EDGE_SAFETY_CHECKLIST=<approved file path>
```

Even then, M5 contains no hardware bindings.

**Step 2: Write process-level E2E**

Start Robot Edge and Operator Console on dynamically allocated ports. Validate:

1. authenticate;
2. acquire lease;
3. submit “导航到前面的椅子” through Operator API;
4. observe task and Edge event timelines;
5. query status without a second command;
6. cancel;
7. expire lease using test clock/control fixture;
8. trigger safety stop;
9. verify latch blocks new work;
10. reset with authorized fixture control;
11. reconnect both event streams;
12. stop both processes with no listeners left.

**Step 3: Update one-command validation**

The script must run:

```text
tests/cortex_navigation
tests/robot_edge
tests/e2e/test_operator_console_mock.py
tests/e2e/test_operator_robot_edge_fixture.py
```

and produce one sanitized report under `logs/validation/`.

**Step 4: Run and commit**

```powershell
powershell -ExecutionPolicy Bypass -File scripts/validate_operator_console.ps1
git add deploy/robot-edge scripts tests/e2e docs/validation src/chat_ui/README.md
git commit -m "test: validate fixture robot edge end to end"
```

### M5 acceptance gate

All must be true:

- clean test run;
- two independent processes exercised;
- auth/scope/replay tests pass;
- lease expiry is fail-closed;
- safety latch requires explicit reset;
- no hardware/ROS SDK imported in workstation processes;
- `hardware_authority=false` everywhere;
- no secret appears in health/snapshot/event/report;
- no listener remains after test;
- owner reviews the validation report.

Stop here and request review. M6 is blocked while hardware remains disconnected.

---

## Milestone M6 — robot-side deployment and read-only hardware

> **BLOCKED UNTIL:** the owner confirms the Raspberry Pi/robot-side computer, network, RealSense, one mobile-base profile, Piper, and physical safety controls are connected and available for supervised testing.

### Task 9: Capture robot-side inventory without motion authority

**Files:**

- Create: `scripts/hardware/robot_edge_preflight.sh`
- Create: `docs/validation/YYYY-MM-DD-robot-edge-inventory.md`
- Modify: `deploy/robot-edge/README.md`

**Steps:**

1. Ask the owner for: robot-side host/IP, OS version, ROS distro, chosen mobile profile (`go2` or `lekiwi`), expected camera serial, expected CAN interface, and physical E-stop description.
2. Run only read-only inventory: OS, CPU architecture, disk, network interfaces, ROS domain/RMW, USB IDs, camera enumeration, CAN interface state, ROS topic/action names, and time synchronization.
3. Do not activate CAN, enable torque, publish `/cmd_vel`, send ROS goals, or open SDK control sessions.
4. Redact IPs, serials, tokens, Wi-Fi credentials, and user names from committed reports.
5. Compare actual ROS Actions with:
   - `/ubrobot/navigation/navigate_to_object`
   - `/ubrobot/manipulation/grasp_object`
6. STOP on any unexpected device/interface/topic or if the selected mobile-base profile is ambiguous.

**Acceptance:** signed inventory report, no motion, no torque enable, no authority granted.

---

### Task 10: Add ROS-side Robot Edge adapters in read-only mode

**Files:**

- Create: `src/robot_edge/ros/__init__.py`
- Create: `src/robot_edge/ros/context.py`
- Create: `src/robot_edge/ros/telemetry.py`
- Create: `src/robot_edge/ros/actions.py`
- Create: `tests/robot_edge/test_ros_adapter_contract.py`
- Create: `deploy/robot-edge/compose.ros-readonly.yaml`

**Steps:**

1. First test adapters with fake ROS clients; workstation tests must skip importing `rclpy` until the hardware-mode factory is called.
2. Map ROS messages into shared DTOs; never return ROS messages to FastAPI.
3. In read-only mode, do not construct Action clients for command execution and do not publish control topics.
4. Report missing Actions/topics as `disconnected` or `unavailable`, never healthy.
5. Validate startup/shutdown repeatedly on robot-side host.

**Acceptance:** capability/telemetry snapshots reflect the real ROS graph, while all command endpoints reject with “hardware authority disabled.”

---

### Task 11: Validate RealSense and robot health read-only

**Files:**

- Create: `src/robot_edge/hardware/realsense_ros.py`
- Create: `src/robot_edge/hardware/mobile_base_health.py`
- Create: `src/robot_edge/hardware/piper_health.py`
- Create: `tests/robot_edge/test_hardware_health_mapping.py`
- Create: `docs/validation/YYYY-MM-DD-hardware-readonly.md`

**Steps:**

1. RealSense: subscribe/read metadata only; validate RGB/depth timestamps, dimensions, encoding, frame IDs, calibration presence, and stale behavior. Do not stream raw depth through JSON.
2. Mobile base: use only the owner-selected `go2` or `lekiwi` profile. Read state/odometry/driver health; do not publish movement commands.
3. Piper: verify CAN/driver/arm status with torque disabled. Do not call enable, go-zero, trajectory, gripper, or SDK motion methods.
4. Disconnect each source deliberately and confirm UI changes to `disconnected` within the specified timeout.
5. Record measured telemetry rates and age, but do not claim motion capability.

**Acceptance:** observation/odometry/joints show truthful live/stale/disconnected state; command authority remains false.

Stop and request owner review before M7.

---

## Milestone M7 — safety gate and one guarded navigation motion

> **BLOCKED UNTIL:** M6 passes, test area is cleared, robot is supported against falls/collisions as appropriate, physical E-stop is verified by a human, and a second observer is present.

### Task 12: Bind and measure local safety controls

**Files:**

- Create: `src/robot_edge/hardware/local_stop.py`
- Modify: `src/robot_edge/safety.py`
- Create: `scripts/hardware/measure_stop_latency.py`
- Create: `docs/validation/YYYY-MM-DD-safety-stop-latency.md`

**Steps:**

1. Bind physical E-stop state to SafetySupervisor.
2. Bind the selected base's local stop primitive.
3. Bind Piper stop/disable only after confirming the vendor-safe method.
4. If local voice keyword stop is implemented, run it fully on Robot Edge and ensure it never depends on Qwen/cloud/network.
5. Test stop fan-out first with motion outputs disabled.
6. Measure input detection, supervisor dispatch, driver acknowledgement, and physical stop latency separately.
7. Test network loss and lease expiry.

**Acceptance:** owner-approved latency limits pass; physical E-stop remains authoritative; safety latch/reset behavior is documented.

---

### Task 13: Execute one low-risk navigation validation

**Files:**

- Modify only the selected mobile-base deployment profile
- Create: `docs/validation/YYYY-MM-DD-navigation-hardware.md`

**Steps:**

1. Select `go2` or `lekiwi` from the approved inventory; disable the other profile.
2. Verify `/cmd_vel` guard, command lease heartbeat, odometry, and local stop while wheels/feet cannot create uncontrolled travel.
3. Send zero commands only; verify timeout returns to zero.
4. Execute a minimal bounded movement in a cleared area at the owner-approved speed/distance.
5. Trigger normal cancel, lease expiry, local stop, UI emergency stop, and physical E-stop in separate trials.
6. Only after those pass, run `NavigateToObject` for a nearby unambiguous target.
7. Record goal, feedback, odometry, lease, stop latency, result, video reference, and operator observations.

**Acceptance:** all stop paths work, lease loss is fail-closed, and one bounded navigation goal succeeds. This does not approve Piper or combined operation.

Stop and request owner review before M8.

---

## Milestone M8 — Piper and combined task

> **BLOCKED UNTIL:** M7 passes and Piper has an approved fixture/work envelope with no person inside the reachable space.

### Task 14: Implement Piper executor behind the existing GraspObject server

**Files:**

- Create: `ros_depends_ws/src/ubrobot_manipulation/ubrobot_manipulation/executors/piper.py`
- Modify: `ros_depends_ws/src/ubrobot_manipulation/ubrobot_manipulation/grasp_object_server.py`
- Modify: `ros_depends_ws/src/ubrobot_manipulation/ubrobot_manipulation/policy.py`
- Create: `ros_depends_ws/src/ubrobot_manipulation/test/test_piper_executor_contract.py`
- Create: `docs/validation/YYYY-MM-DD-piper-hardware.md`

**Steps:**

1. Define a narrow executor protocol: start, feedback, cancel, result, stop.
2. Test it with a fake Piper driver before importing `piper_sdk`.
3. Bind the SDK only in robot-side hardware factory.
4. Validate torque disabled status and joint limits.
5. Enable torque only with explicit operator action and physical E-stop ready.
6. Run no-load, low-speed, single-joint/pose motions within approved limits.
7. Validate cancel and emergency stop before closing the gripper on any object.
8. Run a compliant low-force grasp with retreat and collision limits.

**Acceptance:** Piper-only validation passes; no base motion occurs; all stop paths are measured.

---

### Task 15: Validate the combined Operator Console task

**Files:**

- Create: `tests/hardware/test_operator_combined_acceptance.py`
- Create: `scripts/hardware/validate_combined_task.sh`
- Create: `docs/validation/YYYY-MM-DD-combined-hardware.md`

**Steps:**

1. Start Robot Edge, ROS Actions, EMOS Cortex, and Operator Console with recorded versions/configuration.
2. Verify readiness, authority, lease owner, telemetry freshness, and physical E-stop before submitting a task.
3. Submit a bounded combined request approved by the owner, for example navigating to a nearby object and grasping it.
4. Verify UI timeline, Cortex plan, Action feedback, navigation lease, stationary-base grasp gate, Piper execution, result, and replay.
5. Repeat separate trials for status query, normal cancel, UI emergency stop, local keyword stop, physical E-stop, browser reconnect, Edge reconnect, and selected network-loss cases.
6. Never combine failure injections in one trial until each individual path is safe and understood.

**Acceptance:** signed combined report with exact versions, configuration hashes, logs, measured safety behavior, and explicit limitations.

---

## Global test commands

Workstation software baseline:

```powershell
python -m compileall -q src\chat_ui src\ubrobot_contracts src\robot_edge tests
node --check src\chat_ui\voice_client.js
python -m unittest discover -s tests/cortex_navigation -p "test_*.py" -q
python -m unittest discover -s tests/robot_edge -p "test_*.py" -q
python -m unittest tests.e2e.test_operator_console_mock -v
python -m unittest tests.e2e.test_operator_robot_edge_fixture -v
git diff --check
```

Preferred one-command validation:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/validate_operator_console.ps1
```

After every process-level test, verify no residual service:

```powershell
Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -match 'chat_ui[\\/]app\.py|robot_edge' } |
    Select-Object ProcessId,Name,CommandLine
```

Expected: no test-owned process remains.

## Required validation report contents

Every milestone report must include:

- date/time/timezone;
- commit SHA and dirty/clean state;
- machine role and OS/architecture;
- selected execution mode and mobile-base profile;
- dependency/container image versions;
- exact commands;
- assertions and results;
- sanitized logs and event correlation IDs;
- hardware authority state;
- safety controls present/absent;
- known limitations and deferred tests;
- explicit statement that Mock evidence is not hardware evidence.

## Stop conditions for the implementing model

Stop immediately and ask the owner if:

- the M1–M4 baseline cannot be separated or reproduced;
- a credential, IP, device serial, CAN interface, ROS domain, or mobile-base profile is unknown;
- an unexpected hardware SDK initializes in workstation/fixture mode;
- Mock/Fixture reports hardware authority;
- lease expiry or disconnect does not invoke fail-closed stop;
- physical E-stop is absent or unverified before motion;
- Piper torque is enabled unexpectedly;
- ROS Action/topic names differ from the inventory;
- a test leaves a process, listener, ROS goal, lease, or torque-enabled device active;
- the next step would increase hardware authority without a reviewed milestone report.

## Suggested commit sequence

```text
feat: establish operator console M1-M4 baseline
refactor: extract versioned robot edge contracts
feat: add fixture robot edge service
feat: secure robot edge control endpoints
feat: enforce robot edge lease and safety latch
feat: connect operator runtime to robot edge
feat: stream robot edge telemetry to operator console
test: validate fixture robot edge end to end
feat: add read-only robot edge ROS telemetry
test: validate robot hardware inventory and safety gates
feat: bind guarded mobile navigation
feat: bind guarded piper grasp executor
test: validate combined hardware task
```

Do not squash fixture-only and hardware-authority changes into one commit. Reviewers must be able to identify exactly where authority increased.
