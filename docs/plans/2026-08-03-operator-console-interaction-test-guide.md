# Operator Console Interaction Test Guide (M7 complete, 2026-08-03)

Goal: exercise the frontend (workstation) with typed commands and verify
three dimensions: interaction feedback, Cortex orchestration, and actual
execution — using both the software path and the real Pi stack.

## 0. Current architecture and test boundary

```
Workstation (frontend)                     Raspberry Pi (real stack)
┌──────────────────────┐        ┌─────────────────────────────────────┐
│ Operator Console      │        │ Robot Edge :8780 (READ-ONLY)        │
│ (Gradio, 7863)        │ ─────► │   telemetry: real ROS graph ✅      │
│   mode A: cortex-mock │        │   commands: REJECTED (authority=off)│
│   mode B: robot-edge  │        ├─────────────────────────────────────┤
│                       │        │ emos-nav-readonly (bringup/guard)   │
│                       │        │ emos-cortex-recipe (Cortex+ARK+     │
│                       │        │   RoboML+Kompass)                   │
│                       │        │ lekiwi driver (torque DISABLED ⚠️)  │
└───────────────────────┘        └─────────────────────────────────────┘
```

Test boundary (honest):

- **Interaction feedback** = frontend. Fully testable now.
- **Cortex orchestration** = Pi (ARK planner + Kompass). Trigger real goals
  on the Pi (`/tmp/send_cortex_goal.py`), observe feedback on Pi logs.
- **Actual execution** = Pi (torque + wheels). Torque is currently DISABLED
  (safe posture). Real motion requires re-enabling torque with the owner
  checklist (see section 4).

## 1. Start the frontend (workstation)

Two modes; run in separate terminals:

```powershell
# Mode A: pure software (no Pi) - interaction + mock orchestration
$env:UBROBOT_CHAT_BACKEND = "cortex-mock"
$env:UBROBOT_CHAT_MEDIA = "off"
$env:UBROBOT_CHAT_VOICE_PROVIDER = "mock"
$env:UBROBOT_CHAT_TLS = "off"
$env:PYTHONPATH = "src;src/chat_ui"
python src/chat_ui/app.py
# open http://127.0.0.1:7863

# Mode B: connect to real Pi telemetry (commands rejected - expected)
$env:UBROBOT_CHAT_BACKEND = "robot-edge"
$env:UBROBOT_EDGE_URL = "http://192.168.18.233:8780"
$env:UBROBOT_EDGE_TOKEN = "operator-token-m7-20260803"   # tokens on Pi /app/config/tokens.json
$env:UBROBOT_EDGE_OPERATOR_ID = "operator"
$env:UBROBOT_CHAT_MEDIA = "off"
$env:UBROBOT_CHAT_VOICE_PROVIDER = "mock"
$env:UBROBOT_CHAT_TLS = "off"
$env:PYTHONPATH = "src;src/chat_ui"
python src/chat_ui/app.py
# open http://127.0.0.1:7863
```

## 2. Interaction test cases (frontend)

| # | Instruction | Expected behavior | Pass criteria |
|---|---|---|---|
| I1 | `导航到前面的椅子` | task.queued -> planning -> running -> succeeded; timeline events; no red error | Timeline shows 5+ feedback steps, task SUCCEEDED |
| I2 | `抓取桌上的杯子` | grasp-like mock plan (approach/align/grasp/retreat) | Step 1/1 grasp phases in timeline |
| I3 | `你好` (non-motion) | "[No actions needed]" reply, no task dispatch | Immediate text reply, task idle |
| I4 | Start `导航到前面的椅子` then `停一下` | normal cancel: task CANCELLED, no new dispatch | Status shows cancelled, reply "已取消" |
| I5 | Start task then click `停止当前任务` | bounded cancel, task cancelled | Same as I4 via button |
| I6 | Click `紧急停止机器人` | safety latch: banner, all tasks blocked | New command rejected until reset; events show critical |
| I7 | During running, type `任务进度怎么样？` | answered from runtime, NO second Cortex dispatch | No duplicate task, status query only |
| I8 | Check sensor panel | 6 channels available + values (resolution, odom x/y/yaw, joints) | Telemetry table + detail blocks render values |

Mode B notes: I1/I2 commands are REJECTED by Robot Edge (hardware authority
disabled) - that is expected and correct. Telemetry (I8) shows REAL values
from the Pi graph (odometry x/y/yaw from lekiwi, camera 640x480, joints).

## 3. Cortex orchestration tests (Pi real path)

Trigger a real goal on the Pi and compare with frontend mock behavior:

```bash
# on the Pi
docker cp /tmp/send_cortex_goal.py emos-cortex-recipe:/tmp/
docker exec -e PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages:/opt/emos_overlay/lib/python3.12/site-packages   -e LD_LIBRARY_PATH=/opt/emos_overlay/lib:/opt/ros/jazzy/lib/aarch64-linux-gnu:/opt/ros/jazzy/lib   emos-cortex-recipe python3 /tmp/send_cortex_goal.py
```

| # | Instruction | Expected feedback sequence | Pass criteria |
|---|---|---|---|
| C1 | `请走到椅子旁边` | Received task -> Plan: send_goal_to_... -> Executing with {target, timeout} -> dispatched -> Post-execution... -> All 1 steps completed | Real ARK plan (target 椅子/chair, timeout 60) |
| C2 | Non-motion: `用一句话报告系统状态，不要调用任何工具` | "[No actions needed]" reply; no tool call | Planner declines tools, no lease |
| C3 | Cancel mid-execution (Ctrl+C / modify script) | Plan aborted / goal cancelled; lease revoked | guard log: lease revoked; no residual motion |

Observe orchestration on the Pi:

```bash
docker logs emos-cortex-recipe 2>&1 | tail -20      # Cortex/planner/Kompass
docker logs emos-nav-readonly 2>&1 | grep lease     # guard lease lifecycle
tail -5 /tmp/planner-relay.log                       # ARK calls
```

## 4. Actual execution tests (requires torque + owner checklist)

Torque is currently DISABLED. To exercise real motion, re-enable with the
hard gate and the owner checklist (cleared area, wheels on floor, power
cable at operator, second observer, speed <= 0.05 m/s):

```bash
# on the Pi, in deploy/lekiwi-driver (hard gate config)
LEKIWI_DRIVER_TAG=0.2.0-rc1-m7-20260803 docker compose   -f compose.yaml -f compose.hardware.yaml -f compose.hardware-torque-test.yaml up -d
```

| # | Scenario | Expected | Pass criteria |
|---|---|---|---|
| E1 | Zero command | wheels stationary, velocities 0 | no drift beyond +-5 mrad |
| E2 | Bounded motion 0.03 m/s x 3 s | ~80 mm odom delta | odom delta 60-95 mm, delta y < 5 mm |
| E3 | Lease expiry during motion | command stops <= ~0.33 s | drift < 15 mm after stop signal |
| E4 | Normal cancel | fast stop ~0.13 s | drift < 8 mm |
| E5 | NavigateToObject (chair) | real Kompass vision track, motion toward chair | goal SUCCEEDED, odom delta > 0.1 m |

After tests, restore safe posture: torque-disabled compose (no torque-test).

## 5. Safety rules

1. Torque disabled unless E-tests are explicitly authorized.
2. Real motion: cleared area, wheels on floor, power cable at operator,
   second observer, speed <= 0.05 m/s, distance <= 0.5 m.
3. Physical E-stop is waived (owner); the power cable is the final cutoff.
4. Mode B commands being rejected is correct behavior (Robot Edge is
   read-only until the M8 command backend lands).
5. Mock evidence is not hardware evidence.
