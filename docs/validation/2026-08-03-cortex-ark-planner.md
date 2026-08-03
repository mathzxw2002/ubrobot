# Cortex Real Planner (Volcengine ARK) Integration Validation (M7 upper layer)

- Date/time: 2026-08-03 12:30–13:00 (+08:00)
- Commits: `3b3b427` (TrackVisionTarget fixture), plus earlier M7 fixes
- Machine role: robot-side host (Raspberry Pi 5) + workstation
- Planner: **Volcengine ARK, checkpoint `glm-5-2-260617`** via
  `planner_relay.py` (HTTP → HTTPS, port 18081)
- API key: injected into the Cortex recipe container environment only;
  key file `/tmp/ark_key.txt` (mode 600) on the Pi, **not** in the
  repository, redacted from this report
- Hardware authority: **false**; LeKiwi motor torque disabled; wheels lifted
  (owner confirmed); no motion occurs

## Stack on the Pi

| Container | Image | Role |
|---|---|---|
| `lekiwi-base-driver` | `0.2.0-rc1-m7-20260803` (rebuilt with BEST_EFFORT cmd_vel adapter) | torque-disabled driver |
| `emos-nav-readonly` | `e2e-2c83d27` | `cortex_navigation_bringup` (NavigateToObject server, cmd_vel_guard, RealSense chain) |
| `emos-cortex-recipe` | `m3-8bd4be9` (has `_parse_tool_args` patch) | `recipe.py --cortex-only`, planner via relay |
| track_vision_fixture | runs inside `emos-nav-readonly` | deterministic TrackVisionTarget server (fake Kompass, 0.04 m/s forward) |
| planner relay | Pi host process | `18081 -> https://ark.cn-beijing.volces.com/api/v3` |

## Validation results

### Planner connectivity

- `GET /v1/models` through the relay with the key → **200** (was 401 without).
- 22 `/v1/chat/completions` calls forwarded over the session, all 200.

### Real LLM planning (Chinese command)

Goal: `/cortex_input_command` task = `请走到椅子旁边`

| Step | Result |
|---|---|
| Cortex accepted | ✅ |
| Plan | ✅ `Plan: send_goal_to__ubrobot_navigation_navigate_to_object` |
| Tool args (LLM-extracted) | ✅ `{"target": "椅子", "timeout_sec": 60}` (first run); `{"target": "chair", "timeout_sec": 60}` (second run — LLM non-determinism, both valid) |
| Execute | ✅ dispatched to `semantic_navigation_capability`, async run |

### Command chain (all the way to the driver controller)

| Hop | Evidence |
|---|---|
| TrackVisionTarget fixture → `/navigation/raw_cmd_vel` | publishes 0.04 forward |
| cmd_vel_guard (lease active) → `/cmd_vel` | **161 non-zero (0.04) samples** |
| lekiwi adapter (BEST_EFFORT) → `/lekiwi_base_controller/cmd_vel` | **161 non-zero samples, synchronized** |
| ros2_control controller | torque disabled → wheels stationary (expected) |

Lease lifecycle: `navigation command lease active` → `revoked` per run.

## Issues found and fixed during this integration

1. **LeKiwi driver image out of date**: the running image's
   `cmd_vel_adapter` subscribed RELIABLE while `cmd_vel_guard` publishes
   BEST_EFFORT → commands never reached the driver. Rebuilt the driver
   image from current source (`0.2.0-rc1-m7-20260803`); repository source
   already carried the BEST_EFFORT adapter.
2. **No TrackVisionTarget server**: `navigate_to_object_server` forwards
   goals to `/track_vision_target`, which had no server in the bare
   bringup. Extracted the deterministic fixture from
   `cortex_navigation_mock_test.py` into
   `deploy/emos/test/track_vision_fixture.py` (committed `3b3b427`).
3. **Recipe container missing overlay lib path**: `LD_LIBRARY_PATH` lacked
   `/opt/emos_overlay/lib`, so `rosidl_typesupport_c` for
   `ubrobot_interfaces` failed to load; fixed in the container environment.

## Operational observations

- Real-LLM orchestration latency is high: each Cortex re-confirmation is a
  full ARK round trip (5–8 s); the fixture goal (~8 s execution) completed
  inside a few confirmation cycles. Total wall time from goal send to
  command output was ~56 s in the 120 s window.
- LLM tool argument extraction works with the `_parse_tool_args` patch
  (ARK returns string-form arguments); both Chinese (`椅子`) and English
  (`chair`) target labels were produced across runs.

## Acceptance

- [x] Real planner (ARK glm-5) drives the full upper-layer chain with
      semantic understanding of the Chinese command.
- [x] Command flow reaches `/cmd_vel` and the lekiwi adapter with non-zero
      velocity while the lease is active.
- [x] No motion (torque disabled), no hardware authority, no credential in
      the repository or this report.
- [x] Deterministic fixture replaces Kompass; real Kompass remains M2 scope.

## Limitations / next steps

1. Real Kompass navigation (not the fixture) is still pending; the current
   fixture publishes a constant forward command.
2. `emos` image `jazzy-7a64982` (production tag) predates the navigation
   stack; a current-image rebuild is pending so `emos-nav-readonly` can
   become the production container.
3. Latency tuning (`CORTEX_MONITORING_INTERVAL_SEC`) and the non-motion /
   cancel trials from the 2026-07-31 validation remain to be re-run on this
   stack.
