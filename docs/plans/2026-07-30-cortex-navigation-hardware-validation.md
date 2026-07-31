# Cortex Navigation Hardware Validation Plan

> **Status: written, NOT authorized for execution.** Every gate requires
> explicit operator approval in sequence. Mock or lifted-wheel success never
> authorizes the next gate by itself.

**Goal:** Validate on real LeKiwi hardware that the Cortex-orchestrated
semantic navigation capability (`NavigateToObject`) moves the base only while
a valid outer command lease exists, and that every loss-of-authority path
stops the wheels within the verified latency bounds.

**Architecture under test:**

```text
Chat UI / test client -> /cortex_input_command (Cortex Action)
  -> /ubrobot/navigation/navigate_to_object (capability server, owns lease)
  -> /track_vision_target (Kompass) -> /navigation/raw_cmd_vel
  -> cmd_vel_guard (lease + freshness gate) -> /cmd_vel
  -> lekiwi-base-driver (final clamp, 250 ms watchdog, torque lifecycle)
```

**Tech Stack:** ROS 2 Jazzy, Kompass, EMOS Cortex, ros2_control, Feetech
STS3215, Raspberry Pi ARM64, Docker Compose, Fast DDS UDP-only.

---

## Prerequisites (all must hold before Gate 1)

1. **Images rebuilt from one commit.** Build both `ubrobot/emos` and
   `ubrobot/lekiwi-base-driver` on the Pi from a checksummed `git archive` of
   a single `codex/cortex-navigation` commit (or a merge of it). The driver
   image MUST contain commit `850976a` ("keep torque-disabled preflight
   read-only"); do not reuse `0.2.0-rc1-b90fa1c` or earlier, whose preflight
   silently re-enabled torque every cycle.
2. **Mock regression green.** `docs/validation/2026-07-30-cortex-navigation-mock.md`
   scenarios pass against the exact candidate images, re-run if the source
   moved after `e4fca25`.
3. **Physical setup.** Chassis rigidly secured, all three wheels clear of the
   floor and of obstacles; motor power disconnected while positioning.
4. **Two operators.** One at the terminal, one at the independent physical
   motor-power cutoff. The cutoff operator is authoritative for every
   torque-enabled step.
5. **Abort terminal** prepared with `docker stop -t 1 lekiwi-base-driver`.
   The physical cutoff remains the primary stop.
6. **Pi repo drift check.** Verify the Pi checkout matches the recorded
   commit (`git rev-parse HEAD`, checksum the compose files) — the 2026-07-29
   incident was amplified by a stale Pi checkout.

## Global stop conditions

Stop immediately; the cutoff operator removes motor power if any of:

- the chassis is not rigidly secured or any wheel envelope is obstructed;
- USB vendor/product/serial differs from `1a86` / `55d3` / `5A68011386`;
- `/cmd_vel` or `/lekiwi_base_controller/cmd_vel` has an unexpected publisher;
- any wheel moves during a torque-disabled gate;
- any motor is not model `777`, or IDs `8`, `9`, `7` are not all reachable;
- feedback is non-finite, stale, or inconsistent with the commanded wheel;
- a command does not return below `0.08 rad/s` within one second;
- the driver container restarts automatically during a failure test;
- unexpected noise, vibration, heating, cable movement, or chassis instability.

## Standing bus rules (from the 2026-07-30 firmware root-cause)

- STS3215 wheel mode **auto-enables torque on any goal-speed register write,
  even writing 0**. Torque-free state exists only when nothing writes goals.
- Never share the serial bus between diagnostics (`dump_motor_registers.py`,
  `bisect_torque_enable.py`, `force_torque_off.py`) and the running driver;
  stop or pause the driver first.
- Never hand-turn wheels against live torque.
- Never publish directly to the controller input; use `/cmd_vel` or the
  Action interfaces only.
- A client-side process timeout does NOT cancel a server-side Action goal.
  Before any driver (re)start or mode switch, cancel all Action goals (or
  restart the recipe) and verify `/cmd_vel` carries no non-zero data
  (2026-07-29 regression rule).

## Gate 1: Formal services stopped, torque-off read-back

1. Stop `emos`, `lekiwi-base-driver`, and `emos-dashboard`.
2. With motor power ON but the driver stopped (bus free), run
   `deploy/lekiwi-driver/dump_motor_registers.py` once and record:
   `torque_enable=0` and `goal_velocity=0` on IDs `8`, `9`, `7`.
3. If any motor shows torque enabled or a non-zero goal, run
   `force_torque_off.py`, re-dump, and investigate before continuing.

**Pass:** recorded read-back shows torque off and zero goals on all motors.

## Gate 2: Operator and abort readiness

- Cutoff operator in position with hand on the motor-power switch.
- Abort terminal open; both operators rehearse the stop call.

**Pass:** both operators verbally confirm.

## Gate 3: Torque-disabled serial and joint-state preflight

Start the candidate driver with `compose.hardware.yaml` (NO torque-test
override). Verify:

- serial opened at 1000000 baud; motors `8`, `9`, `7` respond as model `777`;
- log shows `torque-disabled preflight mode`;
- with the read-only preflight (`850976a`), the driver performs **no bus
  writes** until shutdown: confirm by 30 s of traffic observation or by the
  absence of any write log;
- ten `/joint_states` samples are finite; at rest only `0` or the `±0.076699
  rad/s` quantization floor, absolute velocity below `0.08 rad/s`;
- no wheel moves for at least 30 seconds. Any motion is an unconditional
  failure.

Start the EMOS candidate with `compose.cortex-navigation.yaml`
(`EMOS_RECIPE=/emos/recipes/cortex_navigation/recipe.py`). Verify the full
no-goal baseline: `/cmd_vel` and all wheel velocities remain zero for 30 s,
and `/navigation/command_lease` carries no non-empty sample.

**Pass:** all of the above recorded.

## Gate 4: Lifted-wheel zero-command torque test

Re-verify Gate 1–3 conditions. Start the driver with
`compose.hardware-torque-test.yaml` (`restart: "no"`). Verify:

- one `motor torque ENABLED with zero command` log;
- `assert_wheels_stationary(150)` passes (activation aborts otherwise);
- no visible rotation for 30 s; absolute wheel velocity below `0.08 rad/s`;
- container stays running; controllers stay active.

**Pass + explicit operator approval** before Gate 5.

## Gate 5: Lifted-wheel direction pulses through the guard

Below the first-test limits (driver clamp `max_raw_velocity=300`): publish
single `/cmd_vel`-level samples via the guarded path is not possible by
design — pulses MUST enter through the semantic Action so the lease path is
exercised. Use the deterministic `/track_vision_target` fixture from the
mock validation (not the slow VLM server) and send one bounded
`NavigateToObject` goal per case, separated by a full stop and a five-second
observation interval:

1. forward (`linear.x` fixture);
2. lateral (`linear.y` fixture);
3. rotation (`angular.z` fixture).

Pass only if wheel signs match the recorded mock signature table, no
uninvolved wheel sustains motion, and velocity returns below `0.08 rad/s`
within one second of each goal ending. Do not fix a mismatch live: stop,
power off, record the required `*_direction` change, rebuild, and repeat
from Gate 3.

**Pass + explicit operator approval** before Gate 6.

## Gate 6: Loss-of-authority tests while lifted

With the fixture goal active and wheels turning, inject each failure
separately; after each, assert `/cmd_vel` and wheel velocities reach zero
within **300 ms** (limiting case from mock: raw-publication loss, 285 ms):

1. cancel the outer `NavigateToObject` goal;
2. let the outer goal time out (short `timeout_sec`);
3. SIGKILL the Action client (no cleanup runs);
4. terminate `navigate_to_object_server` (lease expiry);
5. stop the fixture's raw command publication (freshness timeout);
6. restart the mock-mode-free driver is N/A here — instead restart the EMOS
   recipe while no outer goal exists, then confirm zero.

Re-run the mock regression first if any case exceeds 300 ms; a lifted
failure ends the session at the cutoff.

**Pass + explicit operator approval** before Gate 7.

## Gate 7: Torque off and operator inspection

- `docker stop -t 3 lekiwi-base-driver`; confirm graceful deactivation and
  torque release in the log (SIGINT path, not SIGKILL).
- With the bus free, `dump_motor_registers.py` read-back: `torque_enable=0`,
  `goal_velocity=0` on all three motors.
- Physical inspection: cabling, mounts, wheel envelope.

**Pass + explicit operator approval** before Gate 8.

## Gate 8: Separately authorized ground move, at most 1 cm

Requires its own explicit go/no-go after Gate 7, with the robot on the
floor, area cleared, and the cutoff operator in position.

- **Detection prerequisite:** the VLM server detection rate must be ≥1 Hz
  (the 2026-07-29 open item: torch cu130 vs driver 535 leaves RT-DETR on CPU
  at ~0.1 Hz). If unresolved, DEFER this gate — do not substitute a static
  fixture for a ground move.
- Send one `NavigateToObject` goal for a clearly visible static object ~1 m
  ahead, `timeout_sec=10`. Abort the goal (and cut power if needed) the
  instant any wheel has moved more than approximately 1 cm of chassis
  displacement.
- Verify the watchdog: after the goal ends, all velocities below
  `0.08 rad/s` within one second.

**Pass + explicit operator approval** before Gate 9.

## Gate 9: Final graceful stop and shutdown

1. Cancel/complete all goals; verify `/cmd_vel` silent and lease empty.
2. `docker stop -t 3 lekiwi-base-driver`, then stop the EMOS container.
3. Torque-off read-back (`dump_motor_registers.py`): torque `0`, goals `0`.
4. Remove motor power; disconnect USB only after the driver is stopped.
5. Record evidence: source SHA, image IDs/digests, register dumps,
   joint-state and `/cmd_vel` logs, per-gate timing, operator names, and any
   failed gate. Create
   `docs/validation/<date>-cortex-navigation-hardware-results.md`.
6. Restore formal deployment to mock mode; keep EMOS stopped until the next
   integration phase is explicitly approved.

## Out of scope / explicit non-goals

- No USB-disconnect or SIGKILL test while a non-zero command is active.
- No moving serial-disconnect test until a servo-side communication timeout
  is independently configured and verified.
- No EMOS-dashboard or full chat-UI hardware test in this plan; the UI path
  was validated no-motion only
  (`docs/validation/2026-07-30-chat-cortex-smoke.md`).
- Mock or lifted-wheel success does NOT authorize any subsequent gate;
  ground operation requires its own explicit approval at Gate 8.
