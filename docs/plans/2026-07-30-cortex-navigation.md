# Cortex Navigation Capability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace UBRobot's prefix-based navigation entry with a Cortex-planned `NavigateToObject` ROS 2 Action that safely delegates to Kompass while an independent command guard controls access to `/cmd_vel`.

**Architecture:** Cortex sees one semantic navigation capability, never raw velocity or hardware controls. The capability server owns the downstream Kompass goal and a short-lived command lease; a separate guard forwards `/navigation/raw_cmd_vel` to `/cmd_vel` only while the lease and raw command are fresh. The independent LeKiwi container retains final limits, watchdog, and torque lifecycle.

**Tech Stack:** Python 3.12, ROS 2 Jazzy, `rclpy`, ROS 2 Actions, `geometry_msgs`, EMOS Cortex, Kompass, Docker Compose, Fast DDS, `unittest`/`pytest`.

---

## Working rules

- Work only in `C:\Users\china\ubrobot-cortex-navigation` on branch
  `codex/cortex-navigation`.
- Keep Raspberry Pi `emos`, `lekiwi-base-driver`, and `emos-dashboard` stopped
  until Task 8 explicitly starts mock containers.
- Do not map `/dev/lekiwi-base`, enable motor torque, or start a real-hardware
  override in this plan.
- Run repository tests after every task. Make one focused commit per task.
- Treat `docs/validation/2026-07-29-emos-real-motion-integration.md` as a
  regression requirement: client exit must not leave command authority alive.

### Task 1: Pin and probe the Cortex API used by the development image

**Files:**
- Create: `deploy/emos/verify_cortex_api.py`
- Modify: `deploy/emos/Dockerfile`
- Create: `tests/cortex_navigation/__init__.py`
- Create: `tests/cortex_navigation/test_cortex_api_contract.py`

**Step 1: Write the failing deployment-contract test**

```python
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


class CortexApiContractTest(unittest.TestCase):
    def test_image_verifies_required_cortex_symbols(self):
        dockerfile = (ROOT / "deploy/emos/Dockerfile").read_text(encoding="utf-8")
        self.assertIn("COPY deploy/emos/verify_cortex_api.py", dockerfile)
        self.assertIn("RUN /ros_entrypoint.sh python3 /opt/ubrobot/verify_cortex_api.py", dockerfile)

    def test_probe_requires_cortex_action_discovery_surface(self):
        probe = (ROOT / "deploy/emos/verify_cortex_api.py").read_text(encoding="utf-8")
        for token in ("Cortex", "CortexConfig", "Action", "Launcher"):
            self.assertIn(token, probe)
```

**Step 2: Run the test and verify it fails**

Run:

```powershell
python -m unittest tests.cortex_navigation.test_cortex_api_contract -v
```

Expected: FAIL because `verify_cortex_api.py` does not exist.

**Step 3: Add the minimal build-time probe**

Create a probe that imports `Cortex`, `CortexConfig`, `agents.ros.Action`, and
the launcher class used by the recipe. Use `inspect.signature()` to assert the
constructor parameters required by the chosen integration, and print the
detected signatures. Do not instantiate ROS nodes at image-build time.

Add to the Dockerfile:

```dockerfile
COPY deploy/emos/verify_cortex_api.py /opt/ubrobot/verify_cortex_api.py
RUN /ros_entrypoint.sh python3 /opt/ubrobot/verify_cortex_api.py
```

**Step 4: Run repository tests**

Run:

```powershell
python -m unittest discover -s tests -v
```

Expected: all tests PASS.

**Step 5: Build only the development image on the Pi**

Create a checksummed archive from this commit and build a tag such as
`ubrobot/emos:cortex-nav-api-<sha>`. Do not replace the stopped `emos`
container. Expected: the build-time probe succeeds. If it fails, stop here and
pin a compatible EMOS base image before writing navigation code.

**Step 6: Commit**

```powershell
git add deploy/emos/verify_cortex_api.py deploy/emos/Dockerfile tests/cortex_navigation
git commit -m "build: verify Cortex navigation API surface"
```

### Task 2: Define the stable `NavigateToObject` Action

**Files:**
- Create: `ros_depends_ws/src/ubrobot_interfaces/CMakeLists.txt`
- Create: `ros_depends_ws/src/ubrobot_interfaces/package.xml`
- Create: `ros_depends_ws/src/ubrobot_interfaces/action/NavigateToObject.action`
- Modify: `deploy/emos/Dockerfile`
- Create: `tests/cortex_navigation/test_navigation_interface_contract.py`

**Step 1: Write the failing interface test**

Assert that the Action contains exactly these fields and status constants:

```text
string target
float32 timeout_sec
---
uint8 SUCCEEDED=0
uint8 CANCELLED=1
uint8 TIMED_OUT=2
uint8 REJECTED=3
uint8 FAILED=4
uint8 status
string message
---
string phase
float32 distance_error
float32 orientation_error
```

Also assert that the Dockerfile copies `ubrobot_interfaces` into the overlay
workspace before running `colcon build`.

**Step 2: Verify the test fails**

Run:

```powershell
python -m unittest tests.cortex_navigation.test_navigation_interface_contract -v
```

Expected: FAIL because the package is missing.

**Step 3: Create the minimal `rosidl` package**

Use `ament_cmake`, `rosidl_default_generators`, and
`rosidl_generate_interfaces()` for `action/NavigateToObject.action`. Export
`rosidl_default_runtime` and add the package to the EMOS overlay build.

**Step 4: Build the overlay in the development image**

Expected: generated Python module
`ubrobot_interfaces.action.NavigateToObject` imports successfully.

**Step 5: Run all tests and commit**

```powershell
python -m unittest discover -s tests -v
git add ros_depends_ws/src/ubrobot_interfaces deploy/emos/Dockerfile tests/cortex_navigation
git commit -m "feat: define navigation capability action"
```

### Task 3: Implement pure goal and command-safety policy

**Files:**
- Create: `ros_depends_ws/src/ubrobot_navigation/package.xml`
- Create: `ros_depends_ws/src/ubrobot_navigation/setup.py`
- Create: `ros_depends_ws/src/ubrobot_navigation/setup.cfg`
- Create: `ros_depends_ws/src/ubrobot_navigation/resource/ubrobot_navigation`
- Create: `ros_depends_ws/src/ubrobot_navigation/ubrobot_navigation/__init__.py`
- Create: `ros_depends_ws/src/ubrobot_navigation/ubrobot_navigation/policy.py`
- Create: `ros_depends_ws/src/ubrobot_navigation/test/test_policy.py`

**Step 1: Write failing pure-Python tests**

Cover:

- target whitespace is trimmed;
- empty and greater-than-128-character targets are rejected;
- timeout must be finite and within `[1.0, 300.0]`;
- NaN/Inf velocity is replaced with zero;
- velocity is clamped to `0.05 m/s` linear and `0.20 rad/s` angular;
- a lease is valid only when active and heartbeat age is at most 0.25 seconds;
- raw command age greater than 0.25 seconds returns zero.

Representative API:

```python
goal = validate_goal(target=" chair ", timeout_sec=30.0)
assert goal.target == "chair"

safe = sanitize_twist(
    linear_x=0.2,
    linear_y=-0.2,
    angular_z=1.0,
    lease_fresh=True,
    command_fresh=True,
)
assert safe == (0.05, -0.05, 0.20)
```

**Step 2: Verify tests fail**

Run:

```bash
python3 -m pytest ros_depends_ws/src/ubrobot_navigation/test/test_policy.py -q
```

Expected: import failure.

**Step 3: Implement the minimal immutable policy types and functions**

Keep ROS imports out of `policy.py`. Use `math.isfinite`, frozen dataclasses,
and explicit constants. Do not add configuration frameworks or generic plugin
systems.

**Step 4: Run policy and repository tests**

Expected: all PASS.

**Step 5: Commit**

```powershell
git add ros_depends_ws/src/ubrobot_navigation
git commit -m "feat: add deterministic navigation safety policy"
```

### Task 4: Add the command-lease guard

**Files:**
- Create: `ros_depends_ws/src/ubrobot_navigation/ubrobot_navigation/cmd_vel_guard.py`
- Create: `ros_depends_ws/src/ubrobot_navigation/test/test_cmd_vel_guard.py`
- Modify: `ros_depends_ws/src/ubrobot_navigation/setup.py`
- Modify: `ros_depends_ws/src/ubrobot_navigation/package.xml`

**Step 1: Write failing node tests with a fake clock**

Test the guard state machine independently from DDS:

1. raw non-zero command without a lease produces zero;
2. fresh matching lease plus fresh raw command is forwarded after clamping;
3. expired heartbeat produces zero on the next 50 ms guard tick;
4. expired raw command produces zero;
5. lease identifier change invalidates the prior command;
6. NaN/Inf produces zero and an error state;
7. lease revocation emits at least three zero samples.

**Step 2: Verify tests fail**

Expected: `CmdVelGuardState` is missing.

**Step 3: Implement the state machine, then the ROS node**

ROS contract:

- subscribe `/navigation/raw_cmd_vel` (`geometry_msgs/Twist`);
- subscribe `/navigation/command_lease` (`std_msgs/String`), where an empty
  string revokes authority;
- publish `/cmd_vel` (`geometry_msgs/Twist`) every 50 ms;
- publish zero unless both inputs are fresh;
- use best-effort volatile QoS for high-rate velocity data and reliable QoS for
  the lease;
- log lease transitions, never every velocity sample.

Register console script:

```python
"cmd_vel_guard = ubrobot_navigation.cmd_vel_guard:main"
```

**Step 4: Run tests and build the overlay**

Expected: policy tests, node tests, and `colcon build` PASS.

**Step 5: Commit**

```powershell
git add ros_depends_ws/src/ubrobot_navigation
git commit -m "feat: gate navigation velocity with a short-lived lease"
```

### Task 5: Add the `NavigateToObject` capability server

**Files:**
- Create: `ros_depends_ws/src/ubrobot_navigation/ubrobot_navigation/navigate_to_object_server.py`
- Create: `ros_depends_ws/src/ubrobot_navigation/ubrobot_navigation/downstream_goal.py`
- Create: `ros_depends_ws/src/ubrobot_navigation/test/test_downstream_goal.py`
- Modify: `ros_depends_ws/src/ubrobot_navigation/setup.py`
- Modify: `ros_depends_ws/src/ubrobot_navigation/package.xml`

**Step 1: Write failing lifecycle tests**

Use a fake downstream action adapter; do not require Kompass in unit tests.
Prove:

- invalid goals are rejected before a lease is acquired;
- only one outer goal can run;
- downstream rejection revokes the lease and returns FAILED;
- feedback is translated into `phase`, `distance_error`, and
  `orientation_error`;
- outer cancellation calls downstream cancellation, waits for acknowledgement,
  revokes the lease, and returns CANCELLED;
- timeout performs the same cancellation sequence and returns TIMED_OUT;
- an exception revokes the lease in `finally`;
- normal success revokes the lease before returning SUCCEEDED.

**Step 2: Verify tests fail**

Expected: lifecycle module missing.

**Step 3: Implement a pure lifecycle coordinator**

Keep ROS goal handles behind a small adapter protocol so failure paths can be
tested synchronously and deterministically.

**Step 4: Implement the ROS Action server**

The node serves `/ubrobot/navigation/navigate_to_object`, uses
`kompass_interfaces/action/TrackVisionTarget` as a client to
`/track_vision_target`, and publishes a random opaque lease ID at 10 Hz only
while the outer goal owns command authority. Never publish `/cmd_vel` here.

On shutdown, revoke the lease. On client cancellation, do not report completion
until downstream cancellation is acknowledged or a bounded cancellation
deadline expires; either way revoke first.

Register console script:

```python
"navigate_to_object_server = ubrobot_navigation.navigate_to_object_server:main"
```

**Step 5: Run tests, build, and commit**

```powershell
git add ros_depends_ws/src/ubrobot_navigation
git commit -m "feat: add cancellable navigation capability action"
```

### Task 6: Compose the navigation capability into EMOS bringup

**Files:**
- Create: `ros_depends_ws/src/emos_bringup/launch/cortex_navigation_bringup.launch.py`
- Modify: `ros_depends_ws/src/emos_bringup/package.xml`
- Create: `tests/cortex_navigation/test_navigation_bringup_contract.py`
- Modify: `deploy/emos/recipes/vision_depth_follower/recipe.py`

**Step 1: Write the failing launch contract**

Assert that the launch starts both `navigate_to_object_server` and
`cmd_vel_guard`, and that the Kompass DriveManager output is remapped from
`/cmd_vel` to `/navigation/raw_cmd_vel`.

**Step 2: Run it and verify failure**

Expected: launch file missing and recipe still exposes `/cmd_vel` directly.

**Step 3: Verify the exact Kompass 0.8.1 remapping API in the development image**

Use Python introspection and a generated launch description. Record the
verified syntax in a code comment and the runbook. Do not guess the API. If a
component-level output remap is unavailable, use the supported launcher ROS
remapping mechanism for only `my_driver`; never globally remap `/cmd_vel` for
the guard or LeKiwi driver.

**Step 4: Add the capability launch**

Compose the validated sensor launch with the two new nodes. Parameterize lease
and raw-command timeouts but default both to `0.25` seconds and guard period to
`0.05` seconds.

**Step 5: Run tests and an overlay launch smoke test**

With no camera or Kompass goal, expected state is: nodes alive, `/cmd_vel`
publishing zero only, and no non-zero sample.

**Step 6: Commit**

```powershell
git add ros_depends_ws/src/emos_bringup deploy/emos/recipes/vision_depth_follower/recipe.py tests/cortex_navigation
git commit -m "feat: compose guarded navigation capability"
```

### Task 7: Add the Cortex navigation recipe and a rollback-safe deployment override

**Files:**
- Create: `deploy/emos/recipes/cortex_navigation/recipe.py`
- Create: `deploy/emos/compose.cortex-navigation.yaml`
- Modify: `deploy/emos/Dockerfile`
- Modify: `deploy/emos/start-stack.sh`
- Create: `tests/cortex_navigation/test_cortex_recipe_contract.py`

**Step 1: Write failing recipe/deployment tests**

Require:

- `Cortex` and `CortexConfig` in the new recipe;
- a bounded `max_planning_steps` and `max_execution_steps`;
- registration/discovery of only the semantic navigation Action;
- no `/cmd_vel`, serial, torque, motor ID, or device path in Cortex tool
  descriptions;
- `EMOS_RECIPE` override selects the new recipe;
- the old `vision_depth_follower` recipe remains in the image;
- Compose override changes recipe selection only and maps no LeKiwi device.

**Step 2: Verify tests fail**

Expected: new recipe and override missing.

**Step 3: Implement the smallest Cortex recipe**

Reuse the existing Vision, Controller, LocalMapper, and DriveManager setup.
Add Cortex with one navigation capability entrypoint. Use the API proven in
Task 1 so Cortex monitors the ROS Action asynchronously and receives feedback.
Use a planner model client configured entirely through environment variables;
do not commit credentials or a machine-specific token.

The tool description must say that it navigates toward one visually detectable
object label, can be cancelled, and may fail when sensors, detection, or
localization are unavailable.

**Step 4: Seed recipes without overwriting host data**

Generalize `start-stack.sh` seeding so the selected recipe is copied only when
missing. Preserve current log and supervisor semantics.

**Step 5: Build and smoke-test the development image**

Expected: Cortex starts, lists exactly the intended navigation capability, and
answers a non-motion question without emitting a navigation Action goal.

**Step 6: Commit**

```powershell
git add deploy/emos tests/cortex_navigation
git commit -m "feat: orchestrate guarded navigation with Cortex"
```

### Task 8: Run cross-container mock and failure-injection tests

**Files:**
- Create: `deploy/emos/test/cortex_navigation_mock_test.py`
- Create: `docs/validation/2026-07-30-cortex-navigation-mock.md`
- Modify: `deploy/emos/README.md`

**Step 1: Add an automated mock-test client**

The client must always cancel its Action goal in `finally`; a local process
timeout alone is forbidden. It records Action feedback, raw commands, guarded
commands, and mock wheel states.

**Step 2: Build SHA-tagged images from a clean archive on the Pi**

Build a new EMOS image and use the existing SHA-tagged LeKiwi image in mock
mode. Verify no `/dev/lekiwi-base` mapping and no running hardware override.

**Step 3: Run the no-goal baseline**

For at least 30 seconds, assert `/cmd_vel` and all mock wheel velocities remain
zero.

**Step 4: Run one bounded mock navigation goal**

Use a static or recorded detection fixture so the test does not depend on the
slow external VLM server. Assert Action feedback arrives and the expected
omnidirectional forward wheel signature appears only while the outer lease is
active.

**Step 5: Inject failures separately**

For each case, assert `/cmd_vel` becomes zero within 300 ms and remains zero:

1. cancel the outer goal;
2. let the outer goal timeout;
3. terminate the Cortex client;
4. terminate the capability server;
5. stop raw command publication;
6. leave a downstream Kompass goal stale while the outer lease is absent;
7. restart the mock LeKiwi driver while no outer lease exists.

Case 3 is expected to be handled by the Action goal timeout plus lease expiry;
it must not recreate the 2026-07-29 stale-command incident.

**Step 6: Stop all Pi services and document evidence**

Stop both containers, verify no related process or port, and verify physical
motor torque remains off. Record image IDs, source SHA, topic rates, Action
results, stop latency, and logs in the validation document.

**Step 7: Commit**

```powershell
git add deploy/emos/test deploy/emos/README.md docs/validation
git commit -m "test: validate Cortex navigation fail-safe behavior"
```

### Task 9: Remove the UBRobot navigation prefix from the primary UI path

**Files:**
- Create: `src/chat_ui/cortex_client.py`
- Create: `tests/cortex_navigation/test_chat_cortex_client.py`
- Modify: `src/chat_ui/pipeline.py:14-35,115`
- Modify: `src/ubrobot/robots/ubrobot.py:284-363`

**Step 1: Write a failing UI client test**

With a fake Cortex Action transport, prove plain text is submitted unchanged,
feedback is streamed, cancellation is propagated, and final text is returned.

**Step 2: Implement `CortexClient` behind a small transport interface**

Do not initialize `Go2Manager` in the primary chat path. Keep ASR/TTS and media
rendering unchanged. The UI's Stop button must cancel the Cortex Action, not
only stop local worker threads.

**Step 3: Retire prefix routing without deleting rollback code**

Mark `Go2Manager.agent_response()` deprecated and remove it from the primary UI
call path. Do not delete navigation research methods in this milestone.

**Step 4: Run tests and commit**

```powershell
python -m unittest discover -s tests -v
git add src/chat_ui src/ubrobot/robots/ubrobot.py tests/cortex_navigation
git commit -m "refactor: route navigation requests through Cortex"
```

### Task 10: Review, document, and prepare a separate hardware plan

**Files:**
- Modify: `README.md`
- Modify: `emos.md`
- Create: `docs/plans/2026-07-30-cortex-navigation-hardware-validation.md`

**Step 1: Run the complete local and container test suites**

Expected: all legacy LeKiwi contracts and new Cortex navigation tests PASS.

**Step 2: Review security and operational boundaries**

Confirm:

- Cortex has no hardware device mapping;
- the navigation capability does not publish final `/cmd_vel`;
- the command guard cannot enable torque;
- only the LeKiwi driver maps `/dev/lekiwi-base`;
- Fast DDS profile and ROS domain match across containers;
- planner credentials come from runtime secrets/environment only;
- old recipe/image tags remain available for rollback.

**Step 3: Write, but do not execute, the hardware validation plan**

The plan must repeat torque-disabled preflight, lifted-wheel direction pulses,
cancel/timeout tests, and a bounded ground move. It must require an operator at
the physical power cutoff and must never infer hardware authorization from mock
success.

**Step 4: Final documentation commit**

```powershell
git add README.md emos.md docs/plans
git commit -m "docs: describe Cortex navigation operations and hardware gate"
```

## Completion criteria

- Plain-language navigation no longer depends on `nav:` in the primary UI.
- Cortex can select only the semantic `NavigateToObject` capability.
- Outer cancellation and timeout cancel the downstream Kompass goal.
- Missing or stale command lease prevents non-zero `/cmd_vel`.
- All injected failures stop mock motion within 300 ms.
- No real device is mapped and no motor torque is enabled by this plan.
- Source commits, SHA-tagged images, test evidence, and rollback instructions
  are recorded before hardware validation is proposed.

