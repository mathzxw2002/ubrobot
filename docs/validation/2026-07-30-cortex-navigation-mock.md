# 2026-07-30 Cortex navigation cross-container mock validation

## Outcome

PASS. The semantic `NavigateToObject` Action drove the independent LeKiwi mock
container only while an outer command lease was active. Every injected failure
made `/cmd_vel` remain or become zero within the required 300 ms. No test
container mapped `/dev/lekiwi-base`, no real driver was started, and final
read-back confirmed torque disabled and goal velocity zero on all three motors.

## Source and images

- Branch: `codex/cortex-navigation`
- Source commit: `e4fca25`
- Clean archive SHA-256:
  `fc1c91b40f258c970276691c0acecfe6e529419518a1b92c61f79ffc1aaee1b7`
- Build/evidence directory:
  `/home/china/ubrobot-builds/task8-e4fca25`
- EMOS image: `ubrobot/emos:task8-e4fca25`
  (`sha256:6351d657b79421f2a9aee19fbce6f0b0e39b94d1871fbb78221aa4e1062a690f`)
- LeKiwi image: `ubrobot/lekiwi-base-driver:task8-e4fca25`
  (`sha256:9b6144b49769f41b52157df9d008db96da382a3fb2c9ac03d8c544f47132723c`)

Both images were built on the Raspberry Pi from the same checksummed archive.
The EMOS test container launched `cortex_navigation_bringup.launch.py` with
`start_sensors:=false`; the LeKiwi container launched only
`hardware_mode:=mock`.

## Safety configuration

- EMOS and LeKiwi: `HostConfig.Devices=[]`, `Privileged=false`.
- LeKiwi mock: read-only root filesystem, all capabilities dropped, no-new-
  privileges, no hardware override.
- Both containers: ROS domain 0, `rmw_fastrtps_cpp`, and the same UDP-only Fast
  DDS profile.
- `/cmd_vel` publisher and subscriber both reported `BEST_EFFORT`; there were
  no incompatible-QoS warnings after commit `e4fca25`.
- The deterministic `/track_vision_target` fixture replaced camera/VLM
  inference. It published a recorded forward command (`linear.x=0.04`) while
  the production outer Action, lease publisher, guard, LeKiwi adapter,
  ros2_control controller, and mock hardware remained in the path.

The first bounded-goal attempt exposed a real integration defect: the guard
published best-effort velocity while the LeKiwi adapter required reliable
delivery. DDS discovered both endpoints but delivered no samples. The adapter
now subscribes with best-effort volatile QoS; a deployment regression test
locks this contract.

## Results

| Scenario | Evidence | Result |
|---|---|---|
| 30.001 s no-goal baseline | 603 `/cmd_vel` and 601 joint samples; every value zero; no lease | PASS |
| Bounded navigation goal | 40 feedback samples; wheel peak `[0, -0.6928, +0.6928]`; result `SUCCEEDED`; stopped in 33.9 ms | PASS |
| Cancel outer goal | Result `CANCELLED`; stopped in 83.1 ms | PASS |
| Outer goal timeout | Result `TIMED_OUT`; stopped in 62.4 ms | PASS |
| Kill Cortex Action client surrogate | Client exited by SIGKILL (`-9`); server timeout revoked authority; stopped 22.2 ms after deadline | PASS |
| Terminate capability server | Raw publisher remained active; lease expired and `/cmd_vel` stopped in 242.5 ms | PASS |
| Stop raw command publication | Lease remained active; raw freshness timeout stopped output in 285.0 ms | PASS |
| Stale downstream goal without outer lease | 21 non-zero raw samples; zero non-zero `/cmd_vel` and wheel samples | PASS |
| Restart mock driver without outer lease | 3.204 s joint-state gap and recovery; 251 `/cmd_vel` samples and all wheel samples zero | PASS |

The raw-publication-loss case is the limiting result at 285.0 ms. It remains
inside the 300 ms requirement but leaves only about 15 ms scheduling margin;
retain the 250 ms freshness limit and 50 ms guard period as explicit safety
parameters in future changes.

## Cancellation and failure semantics

`deploy/emos/test/cortex_navigation_mock_test.py` owns every Action goal in a
`try/finally` and sends a bounded cancellation request in `finally`, then waits
for the result acknowledgement. The client-death case starts that same client
in a subprocess and sends SIGKILL after goal acceptance, deliberately
preventing its cleanup from running. This proves the server-side Action timeout
and lease expiry handle a dead Cortex client; a shell `timeout` is never used as
a substitute for Action cancellation.

The capability-loss case intentionally terminates
`navigate_to_object_server`. Its container log therefore contains the expected
Action execute traceback and process-death record. No unexpected traceback,
QoS incompatibility, real-hardware mode, or torque-enable record occurred.

## Shutdown and physical state

After evidence capture, `task8-emos` and `task8-lekiwi-mock` were stopped and
auto-removed. The formal `emos` and `lekiwi-base-driver` containers remained
stopped throughout. No related ROS process or test listener remained.

With all drivers stopped, `force_torque_off.py` performed an acknowledged
torque-disable write and read-back on the stable device:

```text
motor 8 (back):  torque_enable=0 goal_raw=0 -> OK
motor 7 (left):  torque_enable=0 goal_raw=0 -> OK
motor 9 (right): torque_enable=0 goal_raw=0 -> OK
RESULT: all motors torque OFF, goal 0
```

Mock success does not authorize a real-hardware navigation test. A separate
lifted-wheel and bounded-ground-motion plan remains required.

## Evidence files on the Raspberry Pi

- `task8-*.json`: scenario measurements and pass/fail results
- `task8-container-inspect.json`: image, command, device, privilege, and mount
  evidence captured before shutdown
- `task8-emos.log` and `task8-lekiwi-mock.log`: container logs
- `source.tar`: clean source archive used for both builds
