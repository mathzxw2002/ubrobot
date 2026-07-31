# 2026-07-31 Cortex end-to-end mock validation: UI client to mock wheels

## Outcome

**PASS.** One ROS domain connected the production `RosCortexTransport` client,
the real EMOS Cortex component, the semantic navigation capability, the lease
guard, and the mock LeKiwi driver. A Chinese plain-language prompt
("请走到椅子旁边") drove mock wheels with the correct forward signature while
the command lease was active; a mid-execution cancel was acknowledged in
10.3 ms and stopped motion. No container mapped a hardware device; the
downstream Kompass controller remains a deterministic fixture (M2 scope).

This closes the gap between the two 2026-07-30 validations: the UI-to-Cortex
smoke had no robot stack, and the navigation fail-safe mock had no Cortex
or UI.

## Source and images

- Branch: `codex/cortex-e2e-mock`, final commit `2c83d27`
- Git archive SHA-256:
  `694384fc77dc1b9df5a2ed45cd67a160e8277da7d1ad95f6568a205bb3df9e50`
- EMOS image: `ubrobot/emos:e2e-2c83d27` (ID `1bd72a3e015c`)
- LeKiwi image: `ubrobot/lekiwi-base-driver:e2e-feb68fb` (ID `d2b7cf4c698c`;
  driver sources unchanged since `feb68fb`)
- Evidence directory on the Pi:
  `/home/china/ubrobot-builds/e2e-2c83d27-evidence` (result JSON, planner
  request JSONL, per-container logs and inspect output)

Containers (all `Devices=[]`, host network, ROS domain 0, `rmw_fastrtps_cpp`,
read-only UDP-only Fast DDS profile):

- `e2e-model` — deterministic planner fixture (`mock_planner_server.py`)
- `e2e-driver` — LeKiwi `hardware_mode:=mock`, read-only rootfs, caps dropped
- `e2e-bringup` — capability server + `cmd_vel_guard`, `start_sensors:=false`
- `e2e-cortex` — `cortex_navigation` recipe `--cortex-only` against the fixture
- `e2e-client` — `end_to_end_mock_test.py` with the production transport and
  the deterministic TrackVisionTarget fixture

## Results

| Scenario | Evidence | Result |
|---|---|---|
| 3 s no-goal baseline | 61 `/cmd_vel` samples, all zero; no active lease | PASS |
| Navigation "请走到椅子旁边" | prompt reached planner unchanged; tools offered exactly `inspect_component`, `update_parameter`, `send_goal_to__ubrobot_navigation_navigate_to_object` (10 requests); 32 active-lease samples; 11 UI feedback samples; wheel signature `[back 0.0, left -0.6928, right +0.6928]`; final reply "All 1 steps completed." after 3.14 s; 12 zero samples after the 300 ms deadline | PASS |
| Cancel mid-execution | ack in **10.3 ms**; request ended `CortexRequestError: Plan aborted while waiting for async actions.`; lease emptied; 13 zero samples after lease end | PASS |

Wheel signature matches the 2026-07-28/30 mock baselines for +x motion
(one ~0, one negative, one positive).

## Defects found and fixed during this validation

1. **Planner argument encoding crashes EMOS Cortex.** `_parse_tool_args`
   iterates `arguments.items()`; the OpenAI-standard JSON-string arguments
   form raises `'str' object has no attribute 'items'` inside the execute
   callback and aborts the goal. The fixture emulates a vLLM-style server
   returning already-parsed arguments (commit `f1b94a3`). **M3 consequence:
   a real OpenAI endpoint returns string arguments and would hit the same
   upstream crash — the M3 milestone must patch or wrap this path before
   using a real planner.**
2. **Cortex only waits for async actions while the planner answers
   CONTINUE.** The step-confirmation prompt ("Respond EXECUTE, SKIP, ABORT,
   or CONTINUE") mentions the navigation tool name, so a naive
   pattern-matching planner answers with a tool call; Cortex then finalizes
   ~80 ms after dispatch and `_finalize_goal` calls
   `_cancel_all_active_clients()`, killing the just-started navigation goal
   (wheels twitch and stop). The fixture is now confirmation-aware: CONTINUE
   while `[Active Tools Status]` shows a running action, EXECUTE otherwise
   (commit `aa09d85`).
3. **Test-harness bugs** (not product bugs): an inline `PYTHONPATH=...`
   assignment replaced the ROS Python path (`4631e57`); stale lease samples
   from the first goal made the cancel probe fire before the second goal
   existed (`2c83d27`).

## Behavioral notes for the UI milestone (M4)

- Cortex publishes progressive feedback during navigation ("waiting for
  async actions to complete..." at the 0.5 s monitoring interval); the
  completed final text is "All N steps completed."
- Client-visible cancellation returns the last feedback as a
  `CortexRequestError`; the UI should treat that as a clean stop, not a
  failure banner.

## Shutdown state

All five `e2e-*` containers were removed after evidence capture. The formal
`emos` and `lekiwi-base-driver` containers remained stopped throughout
(`Exited (137)` / `Exited (0)` from the previous sessions). Motor power and
USB remained unplugged; no hardware was accessed at any point.
