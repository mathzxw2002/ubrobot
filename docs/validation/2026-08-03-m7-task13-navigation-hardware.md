# M7 Task 13 Navigation Hardware Validation Report

- Date/time: 2026-08-03 15:40-16:10 (+08:00)
- Machine: Raspberry Pi 5 + LeKiwi base + RealSense D435i
- Mobile profile: **lekiwi** (owner-selected)
- Planner: Volcengine ARK glm-5-2-260617; Detection: RoboML rtdetr (chair)
- Final cutoff: power cable at operator (no physical E-stop, owner decision)
- Second observer present; test area cleared; wheels on floor

## Stack (production images)

| Container | Image | Config |
|---|---|---|
| lekiwi-base-driver | 0.2.0-rc1-m7-20260803 | torque ENABLED, restart:no (hard gate) |
| emos-nav-readonly | jazzy-m7-20260803 | cortex_navigation_bringup |
| emos-cortex-recipe | jazzy-m7-20260803 | full recipe: Cortex+ARK+RoboML+Kompass |
| roboml-resp | vlm_server | rtdetr detection |

## Stage 1: torque enable + zero command

- Driver log: `LeKiwi motor torque ENABLED with zero command`
- Controllers active; container healthy; restart:no
- Zero command: velocities all 0.0; positions stable within +-5 mrad noise

## Stage 2: bounded minimal motion (0.03 m/s x 3 s)

- Lease heartbeat + raw 0.03 m/s for 3 s
- cmd_vel forwarded: 60 non-zero samples (full 3 s at 20 Hz)
- odom delta: **+79.4 mm** (expected 90 mm; start/stop inertia accounts)
- delta y ~= 0 (straight line, no lateral drift)

## Stage 3: stop path trials

| Trial | Motion | Stop latency | Result |
|---|---|---|---|
| Lease expiry (heartbeat stopped, raw kept) | 62.0 mm | +10.1 mm drift (~0.33 s) | fail-closed: guard stops on lease timeout |
| Normal cancel (zero cmd + lease revoke) | 62.2 mm | +3.9 mm drift (~0.13 s) | fast stop |

Guard lease timeout 0.25 s -> command stops within ~0.33 s incl. inertia.

## Stage 4: NavigateToObject (real Kompass vision tracking)

- Goal: target=chair, timeout=20 s -> ACCEPTED -> **SUCCEEDED (status 6)**
- Real detection (rtdetr chair) -> VISION_DEPTH controller -> real motion
- odom: (0.232, -0.003) -> (0.431, 0.328): **delta x=+0.199 m, delta y=+0.331 m**
- Robot tracked and approached the chair, turning as commanded by Kompass

## Findings during execution

1. DDS matching timing: publishing lease/raw immediately after node creation
   loses the first messages (subscription discovery not complete). Fix: wait
   for matching (1-2 s) and repeat-publish heartbeats.
2. Guard lease requires continuous heartbeat (lease_timeout_sec=0.25 s).
3. Guard parameters verified loaded: lease_timeout_sec=0.25,
   raw_command_timeout_sec=0.25, guard_period_sec=0.05 (ros2 param get).
   The launch-time "Parameter 'xxx' is not supported" warnings are harmless
   noise: ros2 launch passes every DeclareLaunchArgument to all nodes as ROS
   parameters, and the RealSense nested launch (no explicit parameters)
   warns about the unknown keys. Nodes with explicit `parameters` (the
   guard) are unaffected.


## Re-validation after tracker timestamp fixes (2026-08-03 evening)

The full E-series was re-run after fixing the Kompass tracker timestamps
(process-relative clock + unified initial-time base, commit 1146174) and
the recipe container rebuild (fresh container, Fast DDS profile mounted,
CriticalZoneChecker ready):

| Test | Result |
|---|---|
| E1 zero command | positions within +-4 mrad, velocities 0 |
| E2 bounded motion 0.03 m/s x 3 s | delta x = +78.9 mm, delta y = -0.6 mm |
| E3 lease expiry fail-closed | 60.8 mm motion, +12.0 mm drift (~0.4 s) |
| E4 normal cancel | 62.6 mm motion, +4.0 mm drift (~0.13 s) |
| E5 NavigateToObject chair | SUCCEEDED, delta x = +36.5 mm, delta y = +14.1 mm |

All stop paths worked; torque restored to disabled after the session.

## Acceptance

- [x] Torque enabled under hard gate (restart:no, operator at cutoff)
- [x] Zero command verified (no motion, no drift)
- [x] Bounded motion at owner-approved speed (0.03 m/s, ~8 cm)
- [x] Stop paths: lease expiry fail-closed; normal cancel fast stop
- [x] NavigateToObject succeeded for a nearby real target (chair)
- [x] No injury/incident; robot returned to rest; power cable available
- [ ] Physical E-stop latency: N/A (owner decision: power cable cutoff)

## Limitations

- Max velocity exercised: 0.03-0.05 m/s (adapter limit 0.05 m/s)
- Distance exercised: ~0.4 m max (goal timeout 20 s at Kompass speeds)
- Kompass RobotConfig radius 0.18 m not yet verified against chassis
- Guard parameter warnings (launch args not accepted) need a follow-up fix
  so lease/raw timeouts are explicit
