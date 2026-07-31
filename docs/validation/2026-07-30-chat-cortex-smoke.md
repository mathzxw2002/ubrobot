# Chat UI to Cortex no-motion smoke validation

Date: 2026-07-30 (Asia/Shanghai)

## Scope and artifact identity

This validation exercised the production `RosCortexTransport` from a
disposable UI client through `/cortex_input_command` to a disposable EMOS
Cortex. It deliberately did not start sensors, Kompass, a LeKiwi driver, or
physical hardware.

- Source commit: `55826a4` (`fix: load Cortex ROS bindings at runtime`)
- Git archive SHA-256:
  `00a562e0c3b00ff3a31a1bbf9a9440afcb8b5bdd6f76c906e509a811b38fd267`
- Image: `ubrobot/emos:chat-cortex-55826a4`
- Image ID:
  `sha256:5e7a02a437606b2c28f9a8f55a7e8a195adc1e090f127158d18ecbd068bb9d75`
- Raspberry Pi evidence directory:
  `/home/china/ubrobot-builds/task9-ui-55826a4`

The first attempt against source `9a09806` stopped before sending a goal because
the lazy ROS binding factory raised `NameError`. Commit `55826a4` replaced the
function-local class-body aliases with an explicit namespace and added a
regression test for the real import path. The image was then rebuilt from a new
checksummed archive and the validation restarted from the container baseline.

## Isolation and transport

The four temporary containers reported:

```text
task9-bringup   Devices=[]  Privileged=false
task9-cortex    Devices=[]  Privileged=false
task9-ui-client Devices=[]  Privileged=false
task9-model     Devices=[]  Privileged=false
```

The three execution containers used `ubrobot/emos:chat-cortex-55826a4`, host
networking, `ROS_DOMAIN_ID=0`, `rmw_fastrtps_cpp`, and the same read-only
UDP-only Fast DDS profile. Bringup used
`cortex_navigation_bringup.launch.py start_sensors:=false`. The model fixture
had no hardware access and returned deterministic text-only responses.

## Results

The ordinary UI prompt reached the model unchanged:

```text
Report whether orchestration is ready. Do not navigate or call tools.
```

The Action returned these feedback/final values:

```text
Received task. Creating a plan for: Report whether orchestration is ready. Do not navigate or call tools.
[No actions needed]. UI Cortex transport is ready; no motion was requested.
```

The request completed successfully in `0.745 s`. During it, the observer saw
12 `/cmd_vel` samples, all zero, no command-lease samples, and therefore no
active lease.

For both the ordinary prompt and the cancellation probe, the model received
these tools:

```text
inspect_component
update_parameter
send_goal_to__ubrobot_navigation_navigate_to_object
```

Thus the semantic navigation capability was the only Action tool visible to
Cortex; no low-level navigation Action was exposed.

The cancellation probe was:

```text
CANCEL_PROBE: report readiness without navigating or calling tools.
```

The ROS Action cancellation acknowledgement arrived in `0.0104 s`, below the
two-second bound. Across the probe, 102 `/cmd_vel` samples were all zero and no
command lease was observed. EMOS accepted the cancellation but allowed the
already-running model request to return normally afterward. This run therefore
proves bounded UI-to-Action cancellation acknowledgement and motion isolation;
it does not claim that cancellation interrupts an in-flight model HTTP call.

Overall smoke result: `passed=true`.

## Shutdown and retained evidence

The JSON result, container inspections, and logs are retained as:

```text
/home/china/ubrobot-builds/task9-ui-55826a4/task9-chat-cortex-smoke.json
/home/china/ubrobot-builds/task9-ui-55826a4/task9-container-inspect.json
/home/china/ubrobot-builds/task9-ui-55826a4/task9-bringup.log
/home/china/ubrobot-builds/task9-ui-55826a4/task9-cortex.log
/home/china/ubrobot-builds/task9-ui-55826a4/task9-model.log
```

After capture, `task9-ui-client`, `task9-cortex`, `task9-bringup`, and
`task9-model` were stopped and removed. No `task9` container or TCP listener on
port 18080 remained. The formal `emos` and `lekiwi-base-driver` containers
remained stopped (`Exited (137)` and `Exited (0)`, respectively).
