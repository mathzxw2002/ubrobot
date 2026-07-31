# Cortex UI Routing and Hardware Gate Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Route ordinary Chat UI text through the EMOS Cortex Action without keyword matching, propagate feedback and cancellation correctly, retain an explicit legacy rollback path, and prepare—but do not execute—the next real-hardware validation.

**Architecture:** Add a small synchronous `CortexClient` above an injectable Action transport. The production transport owns its ROS 2 node/executor and talks only to `/cortex_input_command`; `ChatPipeline` depends on the client rather than constructing `Go2Manager`, and its Stop path cancels the active ROS goal before stopping media workers. The existing `Go2Manager` remains available only when `UBROBOT_CHAT_BACKEND=legacy` is explicitly selected.

**Tech Stack:** Python 3.12, ROS 2 Jazzy `rclpy`, `automatika_embodied_agents/action/VisionLanguageAction`, Gradio, `unittest`, Docker/Fast DDS.

---

## Working rules

- Work in `C:\Users\china\ubrobot-cortex-navigation` on `codex/cortex-navigation`.
- Keep Raspberry Pi `emos`, `lekiwi-base-driver`, and hardware overrides stopped except for explicitly named no-device smoke containers.
- Do not map `/dev/lekiwi-base`, enable torque, or infer hardware authorization from Task 8 mock success.
- Write tests before implementation and make one focused commit per task.
- Preserve ASR, TTS, audio/video rendering, and the old navigation research code unless a task explicitly changes its call site.
- Every normal UI exit and Stop action must cancel the active Cortex goal and wait for bounded acknowledgement.

### Task 9.1: Define the UI-facing Cortex client contract

**Files:**

- Create: `src/chat_ui/cortex_client.py`
- Create: `tests/cortex_navigation/test_chat_cortex_client.py`

**Step 1: Write failing transport-independent tests**

Create fake `Goal` and `Transport` types. Require:

- input text is passed unchanged, including text that does not start with `nav:`;
- feedback is forwarded in order through a callback;
- the final completed feedback text is returned;
- unsuccessful results raise `CortexRequestError` with the last feedback;
- only one request can be active;
- `cancel_active()` calls transport cancellation and waits for acknowledgement;
- the active-goal reference is cleared in `finally`, including exceptions.

Representative test:

```python
transport = FakeTransport(
    feedback=["planning", "moving", "arrived"],
    success=True,
)
client = CortexClient(transport, result_timeout_sec=30.0)
seen = []

reply = client.execute("请走到椅子旁边", on_feedback=seen.append)

self.assertEqual(transport.tasks, ["请走到椅子旁边"])
self.assertEqual(seen, ["planning", "moving", "arrived"])
self.assertEqual(reply, "arrived")
```

**Step 2: Run the test and verify it fails**

Run:

```powershell
python -m unittest tests.cortex_navigation.test_chat_cortex_client -v
```

Expected: import failure because `cortex_client.py` does not exist.

**Step 3: Implement the minimal pure client**

Define protocols rather than importing ROS at module import time:

```python
class CortexGoal(Protocol):
    def wait(self, timeout_sec: float) -> CortexResult: ...
    def cancel(self, timeout_sec: float) -> bool: ...

class CortexTransport(Protocol):
    def send(self, task: str, on_feedback: Callable[[str], None]) -> CortexGoal: ...
```

Use a lock around `_active_goal`. Reject empty text locally, but do not rewrite,
prefix, classify, or otherwise change non-empty text.

**Step 4: Run focused and repository tests**

Expected: new tests and all existing tests pass.

**Step 5: Commit**

```powershell
git add src/chat_ui/cortex_client.py tests/cortex_navigation/test_chat_cortex_client.py
git commit -m "feat: add cancellable Cortex chat client"
```

### Task 9.2: Implement the ROS 2 Cortex Action transport

**Files:**

- Modify: `src/chat_ui/cortex_client.py`
- Modify: `tests/cortex_navigation/test_chat_cortex_client.py`

**Step 1: Add failing adapter tests around a fake rclpy boundary**

Require the adapter to:

- use `/cortex_input_command` by default;
- create `VisionLanguageAction.Goal.task` from the exact input;
- translate `feedback.feedback` and remember the most recent completed text;
- reject an unavailable server after a bounded timeout;
- use `cancel_goal_async()` and inspect `goals_canceling`;
- shut down its executor thread cleanly.

Keep ROS imports inside `RosCortexTransport.__init__()` or a private factory so
Windows unit tests do not require a ROS installation.

**Step 2: Verify the new tests fail**

Expected: `RosCortexTransport` is missing.

**Step 3: Implement one private ROS node and executor thread**

Configuration comes from environment variables with bounded defaults:

```text
CORTEX_ACTION_NAME=/cortex_input_command
CORTEX_SERVER_TIMEOUT_SEC=5
CORTEX_RESULT_TIMEOUT_SEC=180
CORTEX_CANCEL_TIMEOUT_SEC=2
```

Do not read planner credentials in the UI. They remain owned by the EMOS
container.

**Step 4: Run tests and static compilation**

```powershell
python -m unittest tests.cortex_navigation.test_chat_cortex_client -v
python -m py_compile src/chat_ui/cortex_client.py
python -m unittest discover -s tests -v
```

Expected: all pass.

**Step 5: Commit**

```powershell
git add src/chat_ui/cortex_client.py tests/cortex_navigation/test_chat_cortex_client.py
git commit -m "feat: connect chat client to Cortex Action"
```

### Task 9.3: Switch the primary ChatPipeline path to Cortex

**Files:**

- Modify: `src/chat_ui/pipeline.py:1-36,60-75,79-140,257-258`
- Modify: `src/chat_ui/app.py:13-18`
- Create: `tests/cortex_navigation/test_chat_pipeline_routing.py`

**Step 1: Write failing routing and cancellation tests**

Avoid constructing ASR, TTS, Gradio, cameras, or robot hardware in tests. Use a
small injected fake backend and test a factored `request_text()` method.

Prove:

- default backend receives plain text unchanged;
- `nav:` is not required and is not stripped;
- feedback is stored in a thread-safe status queue;
- final text enters the existing TTS/VLM queue exactly once;
- Stop calls `cancel_active()` before joining media workers;
- default construction does not import or instantiate `Go2Manager`;
- `get_robot_observation()` safely returns `(None, None)` for the Cortex path.

**Step 2: Verify tests fail**

Expected: pipeline still constructs `Go2Manager` and calls `agent_response()`.

**Step 3: Add backend injection and default Cortex selection**

Use:

```text
UBROBOT_CHAT_BACKEND=cortex   # default
UBROBOT_CHAT_BACKEND=legacy   # explicit rollback only
```

Import `Go2Manager` only inside the legacy branch. In the default branch,
construct `CortexClient(RosCortexTransport(...))` and do not connect cameras,
LeKiwi, or the legacy VLM.

`stop_pipeline()` ordering must be:

1. set the local stop event;
2. call `cancel_active()` with the bounded Action deadline;
3. join media workers with bounded joins;
4. flush queues and clear UI state.

**Step 4: Preserve feedback and media behavior**

Keep intermediate Cortex feedback separate from the final TTS queue so the UI
does not speak every repeated monitoring update. Expose the newest feedback as
status text while the final completed response follows the current TTS/video
pipeline once.

**Step 5: Run tests**

```powershell
python -m unittest tests.cortex_navigation.test_chat_pipeline_routing -v
python -m unittest discover -s tests -v
```

Expected: all pass without initializing physical devices.

**Step 6: Commit**

```powershell
git add src/chat_ui/pipeline.py src/chat_ui/app.py tests/cortex_navigation
git commit -m "refactor: route Chat UI requests through Cortex"
```

### Task 9.4: Retain a rollback-only legacy path

**Files:**

- Modify: `src/ubrobot/robots/ubrobot.py:284-363`
- Modify: `src/chat_ui/README.md`
- Modify: `tests/cortex_navigation/test_chat_pipeline_routing.py`

**Step 1: Add a failing legacy-boundary test**

Assert the default pipeline contains no `nav:` prefix routing and no eager
`Go2Manager` creation. Assert the explicit legacy backend still delegates to
`Go2Manager.agent_response()` without changing its old behavior.

**Step 2: Mark the old method deprecated**

Add a `DeprecationWarning` to `Go2Manager.agent_response()` explaining that it
is rollback/research code. Do not delete its navigation, grasping, camera, or
VLM methods in this milestone.

**Step 3: Document rollback**

Document:

```powershell
$env:UBROBOT_CHAT_BACKEND = "legacy"
python src/chat_ui/app.py
```

Make clear that legacy mode reconnects local hardware dependencies and must not
be used as the production Cortex path.

**Step 4: Run all tests and commit**

```powershell
python -m unittest discover -s tests -v
git add src/ubrobot/robots/ubrobot.py src/chat_ui/README.md tests/cortex_navigation
git commit -m "chore: isolate legacy keyword routing"
```

### Task 9.5: Run a no-motion UI-to-Cortex integration smoke test

**Files:**

- Create: `deploy/emos/test/chat_cortex_smoke_test.py`
- Create: `docs/validation/2026-07-30-chat-cortex-smoke.md`
- Modify: `deploy/emos/README.md`

**Step 1: Write the smoke client**

Use the production `RosCortexTransport` and submit a non-motion prompt such as:

```text
Report whether orchestration is ready. Do not navigate or call tools.
```

Record Action feedback and result. Always cancel in `finally`, even when a
local timeout or assertion fails.

**Step 2: Build a clean SHA-tagged UI/EMOS test artifact**

Use a checksummed `git archive`. Start only disposable no-device containers
with `start_sensors:=false`; do not start the LeKiwi container because this is a
non-motion transport test.

**Step 3: Execute and verify**

Require:

- plain text reaches Cortex unchanged;
- Action feedback and a successful final response return to the client;
- the model request lists only the semantic navigation Action among Action
  tools;
- `/navigation/command_lease` has no non-empty sample;
- `/cmd_vel` has no non-zero sample;
- Stop/cancel returns within two seconds.

**Step 4: Stop containers and document evidence**

Record source SHA, image ID, Action result, feedback, cancellation result, and
zero-motion observations. Verify no test container or related listener remains.

**Step 5: Commit**

```powershell
git add deploy/emos/test deploy/emos/README.md docs/validation
git commit -m "test: validate Chat UI Cortex transport"
```

### Task 10: Complete the software gate and write the hardware validation plan

**Files:**

- Modify: `README.md`
- Modify: `emos.md`
- Create: `docs/plans/2026-07-30-cortex-navigation-hardware-validation.md`

**Step 1: Run the complete software test matrix**

Run repository tests, navigation package tests with the correct source
`PYTHONPATH`, LeKiwi tests, Task 8 mock regression, and Task 9 no-motion UI
smoke. Expected: all pass.

**Step 2: Review security and ownership boundaries**

Confirm from source and `docker inspect` evidence:

- Cortex and UI have no hardware device mapping;
- only the semantic capability is visible to Cortex;
- capability server never publishes final `/cmd_vel`;
- guard cannot enable torque;
- only the LeKiwi real-hardware override maps `/dev/lekiwi-base`;
- all containers use the same ROS domain and UDP-only Fast DDS profile;
- planner credentials remain runtime environment/secrets;
- prior recipe and image tags remain available for rollback.

**Step 3: Write—but do not execute—the hardware plan**

The plan must require, in order:

1. formal services stopped and verified torque-off read-back;
2. operator at the physical motor-power cutoff;
3. torque-disabled serial and joint-state preflight;
4. lifted-wheel zero-command torque test;
5. lifted-wheel direction pulses below the existing first-test limits;
6. cancel, timeout, capability-loss, and client-loss tests while lifted;
7. torque off and operator inspection;
8. a separately authorized ground move of at most 1 cm;
9. final graceful stop, torque-off read-back, and container shutdown.

Mock or lifted-wheel success must not automatically authorize the next gate.

**Step 4: Update architecture and operations documentation**

Document the primary path as:

```text
Chat UI -> Cortex Action -> semantic capability -> Kompass -> raw command
        -> lease guard -> LeKiwi driver -> hardware
```

Describe the explicit legacy rollback switch and image rollback tags.

**Step 5: Run documentation/source checks and commit**

```powershell
python -m unittest discover -s tests -v
git diff --check
git add README.md emos.md docs/plans
git commit -m "docs: define Cortex navigation hardware gate"
```

## Completion criteria

- Ordinary UI text reaches Cortex unchanged; no `nav:` prefix is required.
- Default Chat UI startup does not construct `Go2Manager` or connect local
  camera/base hardware.
- Cortex feedback is observable and final text enters the existing media path
  once.
- UI Stop cancels the active ROS Action and waits for bounded acknowledgement.
- Legacy keyword routing exists only behind an explicit rollback setting.
- Non-motion UI integration produces no navigation lease or non-zero velocity.
- The full software matrix passes and all disposable Pi containers are stopped.
- Real-hardware validation remains a separate operator-authorized plan.
