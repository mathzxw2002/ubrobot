# 2026-07-31 Real-planner mock validation (M3): GLM-5 through the full chain

## Outcome

**PASS 4/4.** A real LLM planner (Volcengine ARK, `glm-5-2-260617`) drove the
complete chain — production UI transport → EMOS Cortex → semantic capability
→ lease guard → mock LeKiwi driver — with tool selection, non-motion
isolation, and bounded cancellation all verified. No container mapped a
hardware device; the downstream Kompass controller remains a deterministic
fixture (M2 scope). The API key existed only in the Cortex container
environment and is redacted from all evidence.

## Source, images, and endpoint

- Branch: `codex/cortex-real-planner`, final commit `8bd4be9`
- Git archive SHA-256:
  `0bbb4fe43e31bbf4cb95c6cdb8cde95ed97f96b3080a0c5843ad25a6c663e720`
- EMOS image: `ubrobot/emos:m3-8bd4be9` (ID `13deba040e60`) — includes the
  `_parse_tool_args` string-arguments patch (required: ARK returns
  OpenAI-standard string arguments, verified in preflight)
- LeKiwi image: `ubrobot/lekiwi-base-driver:e2e-feb68fb` (ID `d2b7cf4c698c`)
- Endpoint: `https://ark.cn-beijing.volces.com/api/v3` via
  `planner_relay.py` (HTTP→HTTPS pass-through, `/v1` prefix mapped to the
  provider base path); checkpoint `glm-5-2-260617` (present in `/models`,
  127 entries)
- Evidence: `/home/china/ubrobot-builds/m3-8bd4be9-evidence` on the Pi

## Preflight (direct API, from the Pi)

- `GET /models` with the configured key → 200.
- One chat completion with the navigation tool definition → the model
  returned exactly one `send_goal_to__ubrobot_navigation_navigate_to_object`
  call with `{"target": "chair", "timeout_sec": 60}`; **`arguments` arrived
  as a JSON string**, confirming the M1 finding and the necessity of the
  Dockerfile `_parse_tool_args` patch.

## Results

| Scenario | Evidence | Result |
|---|---|---|
| 3 s no-goal baseline | 61 `/cmd_vel` samples, all zero | PASS |
| Navigation "请走到椅子旁边" | GLM selected the navigation tool; wheel signature `[back 0.0, left −0.6928, right +0.6928]`; 7 UI feedback samples; final "All 1 steps completed."; 12 zero samples after the 300 ms deadline; duration 47.7 s (see latency note) | PASS |
| Non-motion "用一句话报告系统当前状态。不要移动机器人，也不要调用任何工具。" | reply "[No actions needed]. 好的，不调用任何工具，也不移动机器人。…"; no lease, no non-zero `/cmd_vel`; 5.2 s | PASS |
| Cancel mid-execution | ack in **10.3 ms**; request ended `CortexRequestError: Plan aborted while waiting for async actions.`; lease emptied; 13 zero samples after lease end | PASS |

Relay forwarded 12 `/v1/chat/completions` calls over the session, all 200.

## Defect found and fixed during this validation

- **Gzip through the relay.** The client's `Accept-Encoding: gzip` was
  forwarded upstream; ARK compressed its JSON and the relay (which passes
  bytes verbatim without re-adding `Content-Encoding`) delivered an
  undecodable body (`'utf-8' codec can't decode byte 0x8b`). The relay now
  strips `accept-encoding` from forwarded headers and requests `identity`
  (commit `8bd4be9`, locked by a contract test).

## Operational observations

- **Confirmation latency with a remote LLM.** Cortex re-confirms with the
  planner every `monitoring_interval` (0.5 s) while an async action runs,
  and each confirmation is a full LLM round trip (~5–8 s to ARK). The 6 s
  fixture goal finished inside one confirmation cycle; total request
  latency was 47.7 s. For production, consider a longer
  `monitoring_interval` or a cheaper/faster confirmation model; the
  behavior is correct, just chatty.
- GLM-5 correctly declined to call tools on the non-motion prompt and
  asked for permission to use `inspect_component` — good tool-boundary
  behavior.
- String-form tool arguments are handled by the image patch; no code path
  in our stack needed changes beyond the relay fix.

## Shutdown state

All five `e2e-*` containers removed; formal `emos` and `lekiwi-base-driver`
remained stopped; no hardware accessed. The API key appears nowhere in the
retained evidence (`REDACTED` in container inspect; zero key-pattern
matches).
