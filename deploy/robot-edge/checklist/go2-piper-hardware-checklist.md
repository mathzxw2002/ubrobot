# Go2+Piper Hardware Acceptance Checklist (Task 6)

Sign each item before ANY motion-authority deployment. This checklist is
referenced by `deploy/robot-edge/compose.go2-piper.hardware.yaml`
(`UBROBOT_EDGE_SAFETY_CHECKLIST`). Version-controlled; record the config
hash and git commit in the validation report.

## Preconditions (dock environment, Task 1 verified)

- [ ] `ros:jazzy-ros-base-noble` runs on the Orin NX dock (verified 2026-08-07)
- [ ] RealSense D435IF detected (`rs-enumerate-devices`)
- [ ] `can0` up at 1 Mbps and Piper frames observed (0x2A1..0x2A5)
- [ ] Go2 body reachable (`ping 192.168.123.161` via eth0)

## Go2 base

- [ ] Go2 is standing (operator-verified) and in sport velocity mode
- [ ] Go2 bridge container (`go2-bridge`) is running, CycloneDDS on eth0
- [ ] `/odom`, `/imu`, `/joint_states` published and fresh
- [ ] Local stop / E-stop bound and contact closed
- [ ] Go2 navigation speed caps confirmed (linear <= 0.2 m/s, angular <= 0.5 rad/s)

## Piper arm

- [ ] Piper CAN driver process running
- [ ] **Torque DISABLED** during S1/S2 (read-only + zero-output stages)
- [ ] Remote perception service URL reachable from the dock
- [ ] `/grasp_poses` endpoint returns a valid pose list on a test payload

## People and environment

- [ ] Open area, no people within reachable distance
- [ ] Two-person rule: one operator, one safety observer
- [ ] Physical E-stop within arm's reach and verified by the observer
- [ ] Operator Console connected and authenticated
- [ ] Rollback plan ready: `CORTEX_ENABLE_GRASP=false`,
      `UBROBOT_EDGE_HARDWARE_AUTHORITY=false`, no active lease/torque

## Staged sequence (one factor changed per round)

- [ ] S1 read-only health (no motion)
- [ ] S2 zero-output / stop (Piper torque DISABLED)
- [ ] S3 low-speed navigation **DEFERRED** (operator release required)
- [ ] S4 stationary pre-grasp (Piper only, base still)
- [ ] S5 light grasp (Piper only, base still)

## Signed

- Operator: ______________________  Date: ________
- Observer: ______________________  Date: ________
