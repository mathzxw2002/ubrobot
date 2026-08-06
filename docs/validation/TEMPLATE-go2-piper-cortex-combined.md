# Go2+Piper Cortex Combined Acceptance Report (Task 6)

> **Template.** Fill every section truthfully during the hardware run. Do
> not mark items that were not executed. The Go2 navigation stages are
> deferred until the operator releases them; the Piper arm stages run first
> on the stationary dog.

## Metadata

- Date/time: ________
- Commit: ________ (see `git log -1`)
- Image/tag: robot-edge `ubrobot/robot-edge:go2-piper-hardware`; go2-bridge `ubrobot/go2-bridge:<tag>`
- Profile/config hash: ________
- RMW / ROS domain: `rmw_cyclonedds_cpp` / `ROS_DOMAIN_ID=__`
- Operator: ________ ; Observer: ________
- Checklist signed: deploy/robot-edge/checklist/go2-piper-hardware-checklist.md

## Deployment gates (must all pass)

- [ ] `UBROBOT_PLATFORM=go2_piper`, `UBROBOT_GRASP_PLATFORM=go2_piper`
- [ ] `UBROBOT_EDGE_MODE=hardware` + `UBROBOT_EDGE_HARDWARE_AUTHORITY=true`
- [ ] Local stop bound (`/v1/health/ready` reports `local_stop.bound=true`, contact closed)
- [ ] `REMOTE_PERCEPTION_SERVICE_URL` reachable; `/grasp_poses` returns valid poses
- [ ] go2-bridge + emos containers running, same RMW/domain

## Acceptance harness (workstation + hardware)

```bash
python tests/hardware/test_go2_piper_cortex_acceptance.py
# [PASS/FAIL] per gate check and per mutual-exclusion safety check
```

Record the pass/fail table here.

## Staged trials (one factor changed per round)

| Stage | Result | Notes (goal, feedback, stop latency, video ref) |
|---|---|---|
| S1 read-only health | | |
| S2 zero-output / stop | | Piper torque DISABLED |
| S3 low-speed navigation | **DEFERRED** | operator release required |
| S4 stationary pre-grasp | | Piper only, base still |
| S5 light grasp | | Piper only, base still |

## Failure-injection rounds

| Injection | Expected | Observed |
|---|---|---|
| normal cancel | stop within budget | |
| lease loss / expiry | zero /cmd_vel | |
| Console/Edge/Cortex disconnect | fail-closed stop | |
| local E-stop | latch + cancel | |
| physical E-stop | power-off + software latch | |
| remote-perception disconnect | grasp fails closed, no motion | |

## Mutual-exclusion verification

- [ ] GraspObject while navigation lease active -> REJECTED
- [ ] NavigateToObject during grasp -> grasp fail-closed cancellation
- [ ] Base moving (nonzero /cmd_vel) -> GraspObject REJECTED

## Known limitations

- (e.g. Go2 navigation on the real dog deferred; settling window not yet
  enforced; Cortex auto chaining not implemented - Task 5 productization)

## Rollback record

- On failure: `CORTEX_ENABLE_GRASP=false`,
  `UBROBOT_EDGE_HARDWARE_AUTHORITY=false`; no active lease or torque kept.
